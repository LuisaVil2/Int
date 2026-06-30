from backend.confidence.scoring import ConfidenceEngine, ConfidenceInputs
from backend.confidence.emergency import EmergencyClassifier
from backend.memory.conversation import ConversationMemory
from backend.learning.recorder import LearningRecorder
from backend.learning.trainer import PromptOptimizationTrainer

def test_confidence_routes():
    e=ConfidenceEngine(); assert e.score(ConfidenceInputs(1,1,1,1,1,1))==100; assert e.route(96)=='automatic_approval'; assert e.route(85)=='qa_review'; assert e.route(79)=='pause_manual_approval'
def test_emergency_detection():
    r=EmergencyClassifier().classify('The patient has chest pain and difficulty breathing'); assert r['force_qa_review']; assert 'chest_pain' in r['labels']
def test_memory_context():
    m=ConversationMemory(); m.add_turn(1,'en','I have pain','Me duele'); assert 'Hablante 1' in m.context()
def test_learning_jsonl(tmp_path):
    p=tmp_path/'corrections.jsonl'; out=tmp_path/'prompt.md'; LearningRecorder(str(p)).record('a','b','c',82,1,'cardiology'); assert 'Prefer: c' in PromptOptimizationTrainer(str(p),str(out)).build_context()
