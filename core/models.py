import os, json
def load_model_list()->list:
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "models.json")
    try: return json.load(open(base,"r",encoding="utf-8"))
    except Exception: return []