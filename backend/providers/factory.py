from backend.config.settings import get_settings
from backend.providers.deepgram import DeepgramProvider
from backend.providers.deepseek import DeepSeekProvider
from backend.providers.xtts import XTTSProvider
def asr_provider():
    s=get_settings(); name=s.asr_provider
    if name!='deepgram': raise RuntimeError(f'ASR provider {name} is not configured for execution')
    return DeepgramProvider(s.raw['models'].get('deepgram','nova-2-medical'))
def llm_provider():
    s=get_settings(); name=s.llm_provider
    if name!='deepseek': raise RuntimeError(f'LLM provider {name} is not configured for execution')
    return DeepSeekProvider(s.raw['models'].get('deepseek','deepseek-chat'))
def tts_provider():
    s=get_settings(); name=s.tts_provider
    if name!='xtts': raise RuntimeError(f'TTS provider {name} is not configured for execution')
    return XTTSProvider()
