from functools import lru_cache
from pathlib import Path
import os, yaml
class Settings:
    def __init__(self, raw): self.raw=raw
    @property
    def asr_provider(self): return self.raw['providers']['asr']
    @property
    def llm_provider(self): return self.raw['providers']['llm']
    @property
    def tts_provider(self): return self.raw['providers']['tts']
    @property
    def qa_delay(self): return float(self.raw['qa']['delay_seconds'])
    def env(self,name,default=None,required=False):
        v=os.getenv(name,default)
        if required and not v: raise RuntimeError(f'Missing required environment variable {name}')
        return v
@lru_cache
def get_settings():
    p=Path(os.getenv('APP_CONFIG','config.yaml'))
    return Settings(yaml.safe_load(p.read_text()) if p.exists() else {})
