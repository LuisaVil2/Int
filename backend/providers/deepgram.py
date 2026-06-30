import asyncio, os
from inspect import isawaitable
from deepgram import DeepgramClient, LiveOptions, LiveTranscriptionEvents
from backend.providers.base import ASRProvider, TranscriptSegment
class DeepgramProvider(ASRProvider):
    def __init__(self,model='nova-2-medical'):
        self.model=model; self.client=DeepgramClient(os.environ['DEEPGRAM_API_KEY']); self.connection=None; self.loop=None
    async def start(self,on_transcript):
        self.loop=asyncio.get_running_loop(); self.connection=self.client.listen.live.v('1')
        def handler(_self,result,**_kwargs):
            if not getattr(result,'is_final',False): return
            alt=result.channel.alternatives[0]; text=(alt.transcript or '').strip()
            if not text: return
            words=[getattr(w,'__dict__',{}) for w in (getattr(alt,'words',[]) or [])]
            speaker=getattr(alt.words[0],'speaker',0) if getattr(alt,'words',None) else 0
            lang=getattr(result.channel,'detected_language',None) or 'en'
            conf=float(getattr(alt,'confidence',0.85) or 0.85)
            async def emit():
                r=on_transcript(TranscriptSegment(text,lang,speaker,conf,words))
                if isawaitable(r): await r
            self.loop.call_soon_threadsafe(lambda: asyncio.create_task(emit()))
        self.connection.on(LiveTranscriptionEvents.Transcript, handler)
        self.connection.start(LiveOptions(model=self.model, language='multi', diarize=True, interim_results=False, smart_format=True))
    async def send_audio(self,audio_chunk:bytes):
        if self.connection: self.connection.send(audio_chunk)
    async def close(self):
        if self.connection: self.connection.finish()
