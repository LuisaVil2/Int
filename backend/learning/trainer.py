import json
from collections import Counter
from pathlib import Path
class PromptOptimizationTrainer:
    def __init__(self,dataset_path='learning/dataset/corrections.jsonl',output_path='learning/prompt_context.md'): self.dataset_path=Path(dataset_path); self.output_path=Path(output_path)
    def build_context(self):
        if not self.dataset_path.exists(): return 'No human corrections recorded yet.'
        rows=[json.loads(l) for l in self.dataset_path.read_text(encoding='utf-8').splitlines() if l.strip()]
        pairs=Counter((r['ai_interpretation'],r['human_correction']) for r in rows).most_common(25)
        text='Prompt optimization notes from QA corrections:\n'+'\n'.join(f'Prefer: {b} | Avoid: {a}' for (a,b),_ in pairs)
        self.output_path.parent.mkdir(parents=True,exist_ok=True); self.output_path.write_text(text,encoding='utf-8'); return text
