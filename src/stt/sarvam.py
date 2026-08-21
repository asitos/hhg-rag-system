import httpx
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import time
from src.config import settings

logger = logging.getLogger(__name__)

class SarvamSTT:
    """
    Decision Documented: 
    Sarvam AI was selected over ElevenLabs because this project specifically targets
    14 Indic Languages. Sarvam's Saaras v3 model is state-of-the-art for native Indic 
    language automatic speech recognition (ASR) and provides better word error rates (WER) 
    for these regional languages compared to generic global models.
    """
    def __init__(self):
        self.api_key = settings.sarvam_api_key
        self.url = "https://api.sarvam.ai/speech-to-text"
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5))
    def transcribe(self, audio_file_path: str, language_code: str = "unknown") -> tuple[str, float]:
        """
        Transcribes the given audio file using Sarvam AI.
        Returns: (transcribed_text, api_latency_ms)
        """
        if not self.api_key:
            logger.warning("No Sarvam API key provided. Using mock transcription for testing.")
            return "This is a mock transcription.", 10.0
            
        t0 = time.perf_counter()
        
        with open(audio_file_path, "rb") as f:
            files = {"file": (audio_file_path, f, "audio/wav")}
            data = {
                "model": "saaras:v2",
                "language": language_code
            }
            headers = {"api-subscription-key": self.api_key}
            
            with httpx.Client(timeout=10.0) as client:
                response = client.post(self.url, headers=headers, data=data, files=files)
                response.raise_for_status()
                
                latency_ms = (time.perf_counter() - t0) * 1000
                result = response.json()
                return result.get("transcript", ""), latency_ms
