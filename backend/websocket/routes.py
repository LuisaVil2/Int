import asyncio, base64, time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.config.settings import get_settings
from backend.providers.factory import asr_provider, llm_provider, tts_provider
from backend.memory.conversation import ConversationMemory
from backend.glossary.service import GlossaryService
from backend.interpreter.service import InterpreterService, target_language
from backend.confidence.scoring import ConfidenceEngine, ConfidenceInputs
from backend.confidence.emergency import EmergencyClassifier
from backend.qa.buffer import QABuffer, QASegment
from backend.monitoring.metrics import metrics
router=APIRouter()
qa_clients:set[WebSocket]=set()
async def broadcast_qa(payload):
    for client in list(qa_clients):
        try: await client.send_json(payload)
        except Exception: qa_clients.discard(client)
class PipelineSession:
    def __init__(self,ws:WebSocket):
        s=get_settings(); self.ws=ws; self.memory=ConversationMemory(); self.glossary=GlossaryService(); self.interpreter=InterpreterService(llm_provider(),self.glossary); self.tts=tts_provider(); self.asr=asr_provider(); self.confidence=ConfidenceEngine(); self.emergency=EmergencyClassifier(); self.qa=QABuffer(s.qa_delay); self.voice_id=s.raw['voice']['default_reference_wav']; self.started=time.perf_counter()
    async def start(self): await self.asr.start(self.on_transcript)
    async def close(self): await self.asr.close()
    async def on_transcript(self,seg):
        t0=time.perf_counter(); src='en' if seg.detected_language.startswith('en') else 'es'; tgt=target_language(src)
        interp=await self.interpreter.interpret(seg.text,src,tgt,self.memory)
        hits=self.glossary.hits(seg.text+' '+interp.text)
        emerg=self.emergency.classify(seg.text+' '+interp.text)
        score=self.confidence.score(ConfidenceInputs(seg.asr_confidence,interp.llm_confidence,0.95 if hits else 0.85,0.9,0.9,1.0 if hits else 0.8))
        route='qa_review' if emerg['force_qa_review'] else self.confidence.route(score)
        self.memory.add_turn(seg.speaker_id,src,seg.text,interp.text)
        qa_seg=QASegment(seg.text,interp.text,src,tgt,seg.speaker_id,score,route,hits,emerg,self.memory.snapshot(),interp.latency_ms+(time.perf_counter()-t0)*1000)
        self.qa.add(qa_seg)
        await broadcast_qa({'type':'qa_segment','segment':qa_seg.to_dict(),'remaining_buffer_seconds':self.qa.due_delay(qa_seg)})
        if route=='automatic_approval': asyncio.create_task(self.release_after_buffer(qa_seg))
        else: await self.ws.send_json({'type':'qa_required','segment':qa_seg.to_dict(),'remaining_buffer_seconds':self.qa.due_delay(qa_seg)})
    async def release_after_buffer(self,qa_seg):
        released=await self.qa.wait_and_release(qa_seg)
        if released: await self.synthesize_and_send(released)
    async def synthesize_and_send(self,qa_seg):
        audio=await self.tts.synthesize(qa_seg.interpretation,self.voice_id,qa_seg.target_language)
        payload={'type':'audio','segment_id':qa_seg.id,'audio_base64':base64.b64encode(audio).decode('ascii'),'mime_type':'audio/wav','text':qa_seg.interpretation}
        await self.ws.send_json(payload); await broadcast_qa({'type':'segment_released','segment_id':qa_seg.id})
    async def qa_action(self,msg):
        seg=self.qa.action(msg['segment_id'],msg['action'],msg.get('edited_text'))
        await broadcast_qa({'type':'qa_action','segment':seg.to_dict()})
        if msg['action'] in {'approve','corrected'}: await self.synthesize_and_send(seg)
@router.websocket('/ws/audio')
async def audio_ws(ws:WebSocket):
    await ws.accept(); session=None
    try:
        session=PipelineSession(ws); await session.start(); await ws.send_json({'type':'status','status':'connected'})
        while True:
            msg=await ws.receive()
            if msg.get('bytes'):
                metrics.frames+=1; await session.asr.send_audio(msg['bytes']); await ws.send_json({'type':'ack','latency_ms':round((time.perf_counter()-session.started)*1000,2)})
            elif msg.get('text'):
                import json; data=json.loads(msg['text'])
                if data.get('type')=='qa_action': await session.qa_action(data)
    except WebSocketDisconnect: pass
    finally:
        if session: await session.close()
@router.websocket('/ws/qa')
async def qa_ws(ws:WebSocket):
    await ws.accept(); qa_clients.add(ws); await ws.send_json({'type':'status','status':'connected'})
    try:
        while True: await ws.receive_text()
    except WebSocketDisconnect: qa_clients.discard(ws)
