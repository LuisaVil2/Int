from collections import deque
class ConversationMemory:
    def __init__(self,max_turns=12): self.turns=deque(maxlen=max_turns)
    def add_turn(self,speaker,source_lang,source_text,interpretation): self.turns.append({'speaker':speaker,'source_lang':source_lang,'source_text':source_text,'interpretation':interpretation})
    def context(self):
        return '(inicio de la conversación, sin contexto previo)' if not self.turns else '\n'.join(f"Hablante {t['speaker']} ({t['source_lang']}): {t['source_text']} → {t['interpretation']}" for t in self.turns)
    def snapshot(self): return list(self.turns)
