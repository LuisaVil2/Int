import base64, os, httpx
from backend.providers.base import TTSProvider
class XTTSProvider(TTSProvider):
    async def synthesize(self,text,voice_id,language):
        with open(voice_id,'rb') as f: ref=base64.b64encode(f.read()).decode('utf-8')
        async with httpx.AsyncClient(timeout=60) as client:
            r=await client.post(os.environ['HF_XTTS_ENDPOINT_URL'],headers={'Authorization':f"Bearer {os.environ['HF_API_TOKEN']}"},json={'inputs':text,'parameters':{'language':language,'speaker_wav_base64':ref}})
            r.raise_for_status(); return r.content
