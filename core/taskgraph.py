from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set

try:
    import jsonschema  # type: ignore
except Exception:
    jsonschema = None  # type: ignore

TASKGRAPH_SCHEMA: Dict[str, Any] = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "goal": {"type":"string"},
    "kpi": {"type":"object", "additionalProperties": {"type":["number","string"]}},
    "constraints": {"type":"array", "items":{"type":"string"}},
    "subtasks": {
      "type":"array",
      "items": {
        "type":"object",
        "properties": {
          "id":{"type":"string"},
          "desc":{"type":"string"},
          "agent":{"type":"string"},
          "role_text":{"type":"string"},
          "inputs":{"type":"object"},
          "acceptance_criteria":{"type":"array","items":{"type":"string"}},
          "evidence_need":{"type":"boolean"},
          "budget":{"type":"object","properties":{
            "max_tokens":{"type":"integer"},
            "max_calls":{"type":"integer"}
          }, "additionalProperties": True},
          "deps":{"type":"array","items":{"type":"string"}},
          "outfmt_hint":{"type":"string"}
        },
        "required":["id","agent","desc"]
      }
    },
    "blackboard":{"type":"object"},
    "stopping_rule":{"type":"string"}
  },
  "required":["goal","subtasks"]
}

@dataclass
class SubTask:
    id: str
    desc: str
    agent: str
    role_text: str = ""
    inputs: Dict[str, Any] = field(default_factory=dict)
    acceptance_criteria: List[str] = field(default_factory=list)
    evidence_need: bool = False
    budget: Dict[str, Any] = field(default_factory=dict)
    deps: List[str] = field(default_factory=list)
    outfmt_hint: str = ""

@dataclass
class TaskGraph:
    goal: str
    subtasks: List[SubTask]
    kpi: Dict[str, Any] = field(default_factory=dict)
    constraints: List[str] = field(default_factory=list)
    blackboard: Dict[str, Any] = field(default_factory=dict)
    stopping_rule: str = "until_all_done"

def validate_graph(graph: Dict[str, Any]) -> Tuple[bool, str]:
    if jsonschema is None:
        return True, "jsonschema-not-installed"
    try:
        jsonschema.validate(instance=graph, schema=TASKGRAPH_SCHEMA)
        return True, "ok"
    except Exception as e:
        return False, str(e)

def topo_ready(subtasks: List[SubTask], done: Set[str]) -> List[SubTask]:
    ready = []
    for t in subtasks:
        if t.id in done:
            continue
        if all((d in done) for d in (t.deps or [])):
            ready.append(t)
    return ready
