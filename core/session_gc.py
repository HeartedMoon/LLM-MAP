# core/session_gc.py
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import json, time, uuid, shutil

ROOT = Path(__file__).resolve().parents[1]
STATE_ROOT = ROOT / "project_state"
SESS_ROOT = STATE_ROOT / "sessions"
SESS_ROOT.mkdir(parents=True, exist_ok=True)

# ---- 会话 ID 获取/初始化 ----
def get_session_id(st_session_state) -> str:
    """
    每个浏览器会话一个 session_id，保存在 st.session_state['session_id']。
    """
    sid = st_session_state.get("session_id")
    if not sid:
        sid = str(uuid.uuid4())
        st_session_state["session_id"] = sid
    return sid

def session_dir(st_session_state) -> Path:
    sid = get_session_id(st_session_state)
    d = SESS_ROOT / sid
    d.mkdir(parents=True, exist_ok=True)
    return d

# ---- 变量表的会话内路径（替换 variable_agent 中的全局路径）----
def variables_path(st_session_state) -> Path:
    return session_dir(st_session_state) / "variables.json"

# ---- 历史结论 JSON 保存 ----
def save_history_json(st_session_state, agent_name: str, payload: Dict[str, Any]) -> Path:
    """
    把本轮 conclusion 落地到会话历史目录，文件名含时间戳和 agent。
    """
    hist_dir = session_dir(st_session_state) / "history"
    hist_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_agent = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in agent_name)[:50] or "Agent"
    fp = hist_dir / f"{ts}_{safe_agent}.json"
    fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return fp

def list_history_files(st_session_state=None) -> List[Path]:
    """列出当前会话产生的全部历史 JSON 文件"""
    sd = (session_dir(st_session_state) / "history").resolve()
    if not sd.is_dir():
        return []
    files = list(sd.glob("*.json"))
    return sorted(p.resolve() for p in files)

def history_name_to_path(st_session_state=None) -> Dict[str, Path]:
    """把“显示名（纯文件名）”映射到“绝对路径”"""
    return {p.name: p for p in list_history_files(st_session_state)}



def resolve_history_paths(names: List[str], st_session_state=None) -> List[str]:
    """把下拉框里选的文件名 -> 绝对路径（字符串）"""
    name_set = set(names or [])
    mapping = history_name_to_path(st_session_state)
    paths = []
    for n in name_set or []:
        # 既兼容“用户传入的是完整路径”的情况，也兼容“只是文件名”的情况
        p = Path(n)
        if p.exists():
            paths.append(str(p.resolve()))
        elif n in mapping:
            paths.append(str(mapping[n].resolve()))
        # 否则丢弃
    return paths

# ---- 心跳 + 垃圾回收 ----
def touch_heartbeat(st_session_state) -> None:
    d = session_dir(st_session_state)
    hb = d / "_heartbeat.json"
    hb.write_text(json.dumps({"last_seen": time.time()}, ensure_ascii=False), encoding="utf-8")

def gc_stale_sessions(max_idle_minutes: int = 20) -> None:
    """
    清理超过 max_idle_minutes 未更新心跳的整个会话目录。
    """
    now = time.time()
    threshold = max_idle_minutes * 60
    for sd in SESS_ROOT.glob("*"):
        try:
            if not sd.is_dir():
                continue
            hb = sd / "_heartbeat.json"
            if not hb.exists():
                # 没有心跳文件的旧目录也清理
                shutil.rmtree(sd, ignore_errors=True)
                continue
            data = json.loads(hb.read_text(encoding="utf-8"))
            last_seen = float(data.get("last_seen", 0))
            if now - last_seen > threshold:
                shutil.rmtree(sd, ignore_errors=True)
        except Exception:
            # 忽略单个目录的错误，继续清理其它
            pass
