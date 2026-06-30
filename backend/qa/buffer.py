import time, uuid, asyncio
from dataclasses import dataclass, field, asdict
@dataclass
class QASegment:
    original_transcript:str; interpretation:str; detected_language:str; target_language:str; speaker_id:int; confidence:int; route:str; glossary_hits:list; emergency:dict; memory:list; latency_ms:float; timestamp:float=field(default_factory=time.time); id:str=field(default_factory=lambda:str(uuid.uuid4())); state:str='pending'
    def to_dict(self): return asdict(self)
class QABuffer:
    def __init__(self,delay_seconds): self.delay=delay_seconds; self.items={}
    def add(self,s): self.items[s.id]=s; return s
    def due_delay(self,s): return max(0,self.delay-(time.time()-s.timestamp))
    async def wait_and_release(self,s):
        await asyncio.sleep(self.due_delay(s)); return self.items.pop(s.id,None)
    def action(self,segment_id,action,edited_text=None):
        s=self.items[segment_id]; s.state=action
        if edited_text: s.interpretation=edited_text
        return s
