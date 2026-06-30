import json,time
from pathlib import Path
class LearningRecorder:
    def __init__(self,path='learning/dataset/corrections.jsonl'): self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True)
    def record(self,original_transcript,ai_interpretation,human_correction,confidence_score,speaker,medical_specialty='general'):
        row={'original_transcript':original_transcript,'ai_interpretation':ai_interpretation,'human_correction':human_correction,'confidence_score':confidence_score,'timestamp':time.time(),'speaker':speaker,'medical_specialty':medical_specialty}
        with self.path.open('a',encoding='utf-8') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')
        return row
