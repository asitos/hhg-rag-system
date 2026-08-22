import asyncio
import base64
import uuid
import time
import logging
import numpy as np
import scipy.signal
from sarvamai import AsyncSarvamAI
from sarvamai.types.realtime_audio_input import RealtimeAudioInput
from src.stt.sarvam import SarvamSTT
from src.config import settings

logger = logging.getLogger(__name__)
ACTIVE_SESSIONS = {}

class SarvamStreamSession:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.client = AsyncSarvamAI(api_subscription_key=settings.sarvam_api_key)
        self.ws = None
        self.ws_ctx = None
        self.connected = False
        self.reader_task = None
        
        self.latest_partial = ""
        self.final_transcript = ""
        self.is_final = False
        self.fallback_used = False
        
        self.t_start = None
        self.t_conn = None
        self.t_first_partial = None
        self.t_final = None
        
        # Buffer for fallback
        self.audio_buffer = []
        self.buffer_sr = 16000

    async def connect(self):
        self.t_start = time.perf_counter()
        try:
            self.ws_ctx = self.client.speech_to_text_realtime_streaming.connect(
                language_code="hi-IN",
                model="saaras:v3-realtime",
                stream_type="fast",
                endpointing="vad",
                silence_duration_ms="200", 
                encoding="linear16",
                sample_rate="16000"
            )
            self.ws = await self.ws_ctx.__aenter__()
            self.t_conn = time.perf_counter()
            self.connected = True
            self.reader_task = asyncio.create_task(self._read_loop())
            logger.info(f"[{self.session_id}] Connected in {(self.t_conn - self.t_start)*1000:.2f}ms")
            return True
        except Exception as e:
            logger.error(f"[{self.session_id}] Connect failed: {e}")
            self.fallback_used = True
            return False

    async def _read_loop(self):
        try:
            async for msg in self.ws:
                if hasattr(msg, 'text'):
                    if self.t_first_partial is None and msg.text:
                        self.t_first_partial = time.perf_counter()
                    self.latest_partial = msg.text
                
                if getattr(msg, 'event', '') == 'transcript.final':
                    self.is_final = True
                    self.final_transcript = msg.text
                    self.t_final = time.perf_counter()
                    logger.info(f"[{self.session_id}] Final received: {self.final_transcript}")
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[{self.session_id}] Reader exception: {e}")
            self.fallback_used = True

    async def send_chunk(self, audio_data: tuple):
        sr, y = audio_data
        
        # Ensure mono
        if len(y.shape) > 1:
            y = y.mean(axis=1)
            
        # Resample to 16000 if needed
        if sr != 16000:
            ratio = sr // 16000
            if sr % 16000 == 0:
                y = y[::ratio]
            else:
                # Nearest neighbor for arbitrary rate to avoid FFT ringing artifacts
                indices = np.round(np.linspace(0, len(y) - 1, int(len(y) * 16000 / sr))).astype(int)
                y = y[indices]
            
        # Convert to int16 if float
        if y.dtype in [np.float32, np.float64]:
            y = np.clip(y, -1.0, 1.0)
            y = (y * 32767).astype(np.int16)
        elif y.dtype != np.int16:
            y = y.astype(np.int16)
            
        self.audio_buffer.append(y)
        
        if not self.connected or self.ws is None:
            return
            
        b64_audio = base64.b64encode(y.tobytes()).decode("utf-8")
        try:
            await self.ws.send_realtime_audio_input(RealtimeAudioInput(audio=b64_audio))
        except Exception as e:
            logger.error(f"[{self.session_id}] Send failed: {e}")
            self.fallback_used = True

    async def finalize(self):
        """Wait for final or use fallback if failed"""
        t_start_finalize = time.perf_counter()
        if not self.fallback_used:
            # Wait for VAD final up to 2 seconds
            for _ in range(20): # max 200ms
                if self.is_final:
                    break
                await asyncio.sleep(0.01)
                
            if not self.is_final:
                logger.warning(f"[{self.session_id}] Timed out waiting for final. Using latest partial.")
                self.final_transcript = self.latest_partial
                self.t_final = time.perf_counter()
        
        if self.fallback_used or not self.final_transcript.strip():
            logger.info(f"[{self.session_id}] Triggering REST fallback...")
            # Combine audio buffer
            if not self.audio_buffer:
                return "No audio", {"stt_mode": "streaming", "fallback": True}
                
            full_audio = np.concatenate(self.audio_buffer)
            import tempfile
            import scipy.io.wavfile as wavfile
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                wavfile.write(tmp.name, 16000, full_audio)
                with open(tmp.name, "rb") as f2:
                    audio_bytes = f2.read()
                rest_stt = SarvamSTT()
                txt = await rest_stt.transcribe(audio_bytes)
                self.final_transcript = txt
                self.fallback_used = True
                self.t_final = time.perf_counter()
                
        metrics = {
            "stt_mode": "rest" if self.fallback_used else "streaming",
            "fallback": self.fallback_used,
            "stt_first_partial_ms": (self.t_first_partial - self.t_start)*1000 if self.t_first_partial else 0.0,
            "stt_final_ms": (self.t_final - t_start_finalize)*1000 if self.t_final else 0.0
        }
        return self.final_transcript, metrics

    async def close(self):
        if self.reader_task:
            self.reader_task.cancel()
        if self.ws_ctx:
            try:
                await self.ws_ctx.__aexit__(None, None, None)
            except Exception:
                pass
        self.connected = False

async def get_session(session_id: str) -> SarvamStreamSession:
    if session_id and session_id in ACTIVE_SESSIONS:
        return ACTIVE_SESSIONS[session_id]
        
    session = SarvamStreamSession()
    success = await session.connect()
    # Even if connect fails, we keep session for fallback (it will store audio in buffer)
    ACTIVE_SESSIONS[session.session_id] = session
    return session

async def cleanup_session(session_id: str):
    if session_id in ACTIVE_SESSIONS:
        session = ACTIVE_SESSIONS.pop(session_id)
        await session.close()
