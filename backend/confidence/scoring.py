from dataclasses import dataclass
@dataclass(slots=True)
class ConfidenceInputs:
    asr_confidence:float; llm_confidence:float; terminology_certainty:float; conversation_consistency:float; speaker_consistency:float; glossary_match:float
class ConfidenceEngine:
    weights={'asr_confidence':.25,'llm_confidence':.25,'terminology_certainty':.15,'conversation_consistency':.15,'speaker_consistency':.10,'glossary_match':.10}
    def score(self,i): return round(sum(max(0,min(1,getattr(i,k)))*w for k,w in self.weights.items())*100)
    def route(self,score):
        if score>=95: return 'automatic_approval'
        if score>=80: return 'qa_review'
        return 'pause_manual_approval'
