import os, json, time
from typing import Dict, Any

BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(BASE, exist_ok=True)
SESSION_FILE = os.path.join(BASE, "session_state.json")

def load_session() -> Dict[str, Any]:
    try:
        return json.load(open(SESSION_FILE, "r", encoding="utf-8"))
    except Exception:
        return {}

def save_session(state: Dict[str, Any]) -> None:
    payload = {
        'page': state.get('page'),
        'current_agent': state.get('current_agent'),
        'chat': state.get('chat'),
        'agent_roles': state.get('agent_roles'),
        'core_auto_rag': state.get('core_auto_rag'),
        'turn_core': state.get('turn_core'),
        'turn_agents': state.get('turn_agents'),
        'core_suggest': state.get('core_suggest'),
        'proc_log': state.get('proc_log'),
        'saved_at': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    }
    json.dump(payload, open(SESSION_FILE,"w",encoding="utf-8"), ensure_ascii=False, indent=2)
