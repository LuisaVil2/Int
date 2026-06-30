from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping
@dataclass(slots=True)
class TranscriptSegment:
    text:str; detected_language:str; speaker_id:int; asr_confidence:float; words:list[dict[str,Any]]
@dataclass(slots=True)
class InterpretationResult:
    text:str; llm_confidence:float; latency_ms:float
class ASRProvider(ABC):
    @abstractmethod
    async def start(self,on_transcript:Callable[[TranscriptSegment], Awaitable[None]|None])->None: ...
    @abstractmethod
    async def send_audio(self,audio_chunk:bytes)->None: ...
    @abstractmethod
    async def close(self)->None: ...
class LLMProvider(ABC):
    @abstractmethod
    async def interpret(self,text:str,source_lang:str,target_lang:str,context:str,glossary:Mapping[str,Any],style_profile:Mapping[str,Any])->InterpretationResult: ...
class TTSProvider(ABC):
    @abstractmethod
    async def synthesize(self,text:str,voice_id:str,language:str)->bytes: ...
