import re
class EmergencyClassifier:
    TERMS={'stroke':['stroke','derrame cerebral','accidente cerebrovascular'],'cardiac_arrest':['cardiac arrest','paro cardíaco'],'heart_attack':['heart attack','infarto'],'chest_pain':['chest pain','dolor de pecho'],'suicidal_ideation':['suicidal','suicidio','quitarme la vida'],'anaphylaxis':['anaphylaxis','anafilaxia'],'difficulty_breathing':['difficulty breathing','shortness of breath','no puedo respirar','dificultad para respirar'],'seizure':['seizure','convulsión'],'overdose':['overdose','sobredosis'],'hemorrhage':['hemorrhage','hemorragia','bleeding out']}
    def classify(self,text):
        lower=text.lower(); labels=[]
        for label,terms in self.TERMS.items():
            if any(re.search(r'\b'+re.escape(t.lower())+r'\b',lower) for t in terms): labels.append(label)
        return {'is_emergency':bool(labels),'labels':labels,'force_qa_review':bool(labels)}
