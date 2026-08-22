import httpx
import logging
from src.config import settings

logger = logging.getLogger(__name__)

class SarvamSTT:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.sarvam_api_key
        self.url = "https://api.sarvam.ai/speech-to-text"
        
    async def transcribe(self, audio_bytes: bytes, language: str = "unknown") -> str:
        """
        Transcribes the given audio bytes using Sarvam AI asynchronously.
        """
        if not self.api_key:
            logger.warning("No Sarvam API key provided.")
            return ""
            
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    self.url,
                    headers={"api-subscription-key": self.api_key},
                    files={"file": ("audio.wav", audio_bytes, "audio/wav")},
                    data={"model": "saaras:v3", "language_code": language},
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("transcript", "")
        except httpx.HTTPError as e:
            logger.error(f"Sarvam API HTTP error: {str(e)}")
            return ""
        except Exception as e:
            logger.error(f"Sarvam transcription error: {str(e)}")
            return ""
