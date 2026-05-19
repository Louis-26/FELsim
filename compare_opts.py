import json, re

def find_opt(fpath):
    try:
        with open(fpath, 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception:
        return []
    results = []
    found_doublet = False
    for i, cell in enumerate(nb['cells']):
        src_lines = cell.get('source', [])
        txt = ''.join(src_lines).lower()
        if 'quadrupole doublet' in txt:
            found_doublet = True
        if found_doublet and cell['cell_type'] == 'markdown' and 'optimization' in txt:
            title = src_lines[0].strip() if src_lines else "No Title"
            for j in range(i+1, min(i+10, len(nb['cells']))):
                c = nb['cells'][j]
                if c['cell_type'] == 'code':
                    code_src = ''.join(c.get('source', []))
                    if any(x in code_src for x in ['minimize', 'least_squares']):
                        stdout = ""
                        for out in c.get('outputs', []):
                            if 'text' in out:
                                stdout += "".join(out['text'])
                            elif 'data' in out:
                                for k, v in out['data'].items():
                                    if 'text' in k: stdout += str(v)
                        
                        v_ext = {}
                        for m in re.finditer(r'(\w+)\s*=\s*({\s*[^}]+\s*}|\[\s*[^\]]+\s*\])', code_src):
                            name, val = m.groups()
                            if any(x in name.lower() for x in ['var', 'obj', 'bound', 'x0']):
                                v_ext[name] = val.strip()
                                
                        results.append({'title': title, 'vars': v_ext, 'stdout': stdout, 'method': 'least_squares' if 'least_squares' in code_src else 'minimize'})
                        break
    return results

orig = find_opt('backend_orig/test/beamline_optimization.ipynb')
new = find_opt('backend/test/beamline_optimization.ipynb')

for i in range(max(len(orig), len(new))):
    print(f"CASE {i+1}")
    if i < len(orig):
        o = orig[i]
        print(f"  [ORIG] {o['title']} | Method: {o['method']}")
        print(f"  [ORIG] Context: {o['vars']}")
        print(f"  [ORIG] Stdout: {o['stdout'].strip()[-500:]}")
    if i < len(new):
        n = new[i]
        print(f"  [NEW ] {n['title']} | Method: {n['method']}")
        print(f"  [NEW ] Context: {n['vars']}")
        print(f"  [NEW ] Stdout: {n['stdout'].strip()[-500:]}")
    print("-" * 60)
