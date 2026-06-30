import os, time
from openai import AsyncOpenAI
from backend.providers.base import LLMProvider, InterpretationResult
class DeepSeekProvider(LLMProvider):
    def __init__(self,model='deepseek-chat'):
        self.model=model; self.client=AsyncOpenAI(api_key=os.environ['DEEPSEEK_API_KEY'], base_url='https://api.deepseek.com')
    async def interpret(self,text,source_lang,target_lang,context,glossary,style_profile):
        start=time.perf_counter()
        system=f'''Eres un intérprete médico profesional certificado. NO eres un traductor literal.
Interpreta de {source_lang} a {target_lang}. Devuelve únicamente la interpretación.
Preserva intención, registro, tono, terminología médica, medicamentos, medidas y expresiones idiomáticas.
Contexto conversacional:\n{context}\nGlosario médico:\n{glossary}\nPerfil de estilo obligatorio:\n{style_profile}'''
        r=await self.client.chat.completions.create(model=self.model,messages=[{'role':'system','content':system},{'role':'user','content':text}],temperature=0.2,max_tokens=300)
        return InterpretationResult((r.choices[0].message.content or '').strip(),0.9,(time.perf_counter()-start)*1000)
