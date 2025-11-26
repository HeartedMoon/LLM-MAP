import os, json, glob, time
from collections.abc import Mapping, Sequence

SESSION_DIR = "sessions"
OUTPUT_DIR = "outputs"
os.makedirs(SESSION_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# def save_session(state:dict, fname:str="latest.json"):
#     """
#     保存当前会话状态
#     """
#     try:
#         path=os.path.join(SESSION_DIR,fname)
#         with open(path,"w",encoding="utf-8") as f:
#             json.dump(state,f,ensure_ascii=False,indent=2)
#     except Exception as e:
#         print("save_session error",e)

def latest_entry_path(agent_name:str)->str:
    """
    获取某个 agent 最近一次输出的 JSON 文件路径
    """
    files=sorted(glob.glob(os.path.join(OUTPUT_DIR,f"{agent_name.replace(' ','_')}*.json")),key=os.path.getmtime,reverse=True)
    return files[0] if files else ""

def list_agent_files()->list:
    """
    列出 outputs 目录下的所有 JSON 文件
    """
    return [os.path.basename(p) for p in glob.glob(os.path.join(OUTPUT_DIR,"*.json"))]

ALLOWED_TOP_KEYS = {
    "chat", "proc_log", "agent_roles", "page", "current_agent",
    "turn_core", "turn_agents", "core_auto_rag", "core_suggest"
}

def _sanitize(obj, depth=0, max_depth=6):
    """把对象递归地转为 JSON 可序列化：基本类型保留；映射/序列递归；其它转字符串。"""
    if depth > max_depth:
        return None
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Mapping):
        out = {}
        for k, v in obj.items():
            out[str(k)] = _sanitize(v, depth+1, max_depth)
        return out
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [_sanitize(v, depth+1, max_depth) for v in obj]
    # 其他不可序列化对象：转成字符串
    try:
        return str(obj)
    except Exception:
        return None

def save_session(ss, path="session_snapshot.json"):
    """只保存允许的键，并且先做 sanitize。"""
    try:
        slim = {k: ss.get(k) for k in ALLOWED_TOP_KEYS}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_sanitize(slim), f, ensure_ascii=False, indent=2)
        return True, path
    except Exception as e:
        return False, str(e)
