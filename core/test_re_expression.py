import os
import json, re
def _parse_json_robust(text: str) -> dict:
    import json, re
    if not text: return {}
    s = text.strip()

    # ```json ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, re.IGNORECASE)
    if m: s = m.group(1).strip()
    # <JSON>...</JSON>
    m = re.search(r"<JSON>\s*([\s\S]*?)\s*</JSON>", s, re.IGNORECASE)
    if m: s = m.group(1).strip()

    # 直接 parse
    try:
        j = json.loads(s)
        if isinstance(j, dict): return j
    except Exception:
        pass

    # 扫第一个平衡花括号
    depth = 0; start = -1
    for i, ch in enumerate(s):
        if ch == '{':
            if depth == 0: start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    cand = s[start:i+1]
                    try:
                        j = json.loads(cand)
                        if isinstance(j, dict): return j
                    except Exception:
                        pass
    return {}


reply = '好的，我们来推导碳酸锂 (Li₂CO₃) 的溶度积表达式。\n\n---\n\n### 步骤 1：写出溶解平衡反应\n碳酸锂在水中的溶解反应为： \n\[\n\text{Li}2\text{CO}3 (s) \;\rightleftharpoons\; 2\,\text{Li}^+ (aq) + \text{CO}3^{2-} (aq)\n\]\n\n---\n\n### 步骤 2：定义溶解度\n设碳酸锂的摩尔溶解度为 \(s\) (mol·L⁻¹)。 \n则在平衡时：\n- \([\text{Li}^+] = 2s\) \n- \([\text{CO}3^{2-}] = s\)\n\n---\n\n### 步骤 3：写出溶度积表达式\n溶度积常数 \(K{sp}\) 定义为溶解平衡中离子浓度的乘积（各自浓度的化学计量数次方）：\n\n\[\nK{sp} = [\text{Li}^+]^2 \cdot [\text{CO}3^{2-}]\n\]\n\n代入浓度表达式：\n\n\[\nK{sp} = (2s)^2 \cdot (s) = 4s^3\n\]\n\n---\n\n### 步骤 4：总结\n因此，碳酸锂的溶度积表达式为：\n\n\[\nK{sp} = 4s^3\n\]\n\n其中 \(s\) 为碳酸锂的摩尔溶解度。\n\n---\n\n### 假设条件\n1. 溶液为稀溶液，忽略离子间相互作用（即活度系数≈1）。 \n2. 温度固定（通常为 25\u202f°C），不考虑温度对 \(K{sp}\) 的影响。 \n3. 无其他离子共存效应。 \n\n---\n\n## JSON 结论\n\njson\n{\n  "models": [\n    {\n      "name": "Lithium Carbonate Solubility Product",\n      "formula": "Ksp = [Li+]^2 * [CO3^2-] = 4s^3",\n      "assumptions": "Ideal dilute solution, constant temperature, no common ion effect, activity coefficients ≈ 1"\n    }\n  ],\n  "variables_changed": true,\n  "notes": "定义了新变量 s (摩尔溶解度)，并据此推导出 Ksp 表达式。"\n}\n'
conclusion={}
m = re.search(r"\{[\s\S]*\}\s*$", reply)  # 最后一个 JSON 对象
if m:
    try:
        conclusion = json.loads(m.group(0))
    except Exception:
        pass
n = _parse_json_robust(reply)
# conclusion1 = json.loads(n)
a = "efewfew"
if a:
    print("逻辑判断成立，a不是空串")
else:
    print("a是空串")

b = []
if b:
    print("逻辑判断成立，b不是空表")
else:
    print("b是空表")
print(m)
print(conclusion)
print(n)
# print(conclusion1)
print(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'project_state', 'variables.json'))