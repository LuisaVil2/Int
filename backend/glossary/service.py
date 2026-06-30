import json
from pathlib import Path
class GlossaryService:
    def __init__(self,path='glosario_medico.json'):
        self.path=Path(path); self.data=json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {'terminos':[],'reglas_estilo':''}
    def hits(self,text):
        lower=text.lower(); out=[]
        for item in self.data.get('terminos',[]):
            if item.get('origen','').lower() in lower or item.get('destino','').lower() in lower: out.append(item)
        return out
