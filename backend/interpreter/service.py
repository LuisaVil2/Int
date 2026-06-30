import json
from pathlib import Path
class InterpreterService:
    def __init__(self,llm,glossary,style_path='style_profile.json'):
        self.llm=llm; self.glossary=glossary; self.style=json.loads(Path(style_path).read_text(encoding='utf-8'))
    async def interpret(self,text,source,target,memory): return await self.llm.interpret(text,source,target,memory.context(),self.glossary.data,self.style)
def target_language(lang): return 'es' if lang.startswith('en') else 'en'
