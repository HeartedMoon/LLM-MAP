import argparse, json
from core.dyn_schema import load_or_propose_schema
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--role", default="")
    ap.add_argument("--outfmt", default="")
    ap.add_argument("--examples", default="")
    args = ap.parse_args()
    shape_text, draft = load_or_propose_schema(args.agent, role_text=args.role, outfmt_hint=args.outfmt, examples=args.examples)
    print("=== SHAPE (for prompt) ===")
    print(shape_text)
    print("\n=== JSON Schema (Draft-07) ===")
    print(json.dumps(draft, ensure_ascii=False, indent=2))
if __name__ == "__main__":
    main()