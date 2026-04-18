---
description: Summarize store/results for a given scenario (task|domain|class)
argument-hint: [scenario] [experiment]
---

Summarize `store/results/acc-*.txt` files, grouped by CL method, showing best λ per method.

- `$1` = scenario (e.g. `task`, `domain`, `class`) — defaults to `task`
- `$2` = experiment prefix (e.g. `splitMNIST5`) — defaults to `splitMNIST5`

Run this from the repo root:

```bash
cd /Users/marawangamal/Documents/github/continual-learning/store/results && python3 -c "
import glob, os, re, sys

scenario = '${1:-task}'
exp = '${2:-splitMNIST5}'
prefix = f'acc-{exp}-{scenario}'

files = glob.glob(f'{prefix}*.txt')
rows = []
for f in files:
    try:
        acc = float(open(f).read().strip().split()[0])
    except Exception:
        continue
    name = os.path.basename(f).removesuffix('.txt')
    stem = name[len(prefix):].lstrip('-')
    tag = stem.split('--', 1)[-1] if '--' in stem else stem
    rows.append((tag if tag else 'baseline (no-reg)', acc))

def method(t):
    if 'ActMatI' in t: return 'actmat-i'
    if 'ActMatC' in t: return 'actmat-c'
    if 'FIdiag' in t:  return 'ewc'
    if '-SI'    in t:  return 'si'
    if '-OWM'   in t:  return 'owm'
    return 'baseline'

def reg_strength(t):
    m = re.search(r'PReg([0-9.]+(?:[eE][+-]?[0-9]+)?)', t)
    return float(m.group(1)) if m else None

groups = {}
for t, a in rows:
    groups.setdefault(method(t), []).append((t, a))

if not rows:
    print(f'no results for prefix: {prefix}'); sys.exit(0)

print(f'== {exp} / {scenario} ==\n')
order = ['baseline', 'actmat-i', 'actmat-c', 'ewc', 'si', 'owm']
summary = []
for m in order + [k for k in groups if k not in order]:
    g = groups.get(m, [])
    if not g: continue
    g_sorted = sorted(g, key=lambda r: -r[1])
    best_tag, best_acc = g_sorted[0]
    lams = sorted({reg_strength(t) for t, _ in g if reg_strength(t) is not None})
    if lams:
        lam_range = f'[{lams[0]:g}, {lams[-1]:g}] ({len(lams)} pts)'
        best_lam = reg_strength(best_tag)
        best_lam_s = f'{best_lam:g}' if best_lam is not None else '—'
    else:
        lam_range, best_lam_s = '—', '—'
    summary.append((m, best_lam_s, best_acc, lam_range))
    print(f'[{m}]  best: {best_tag:<55s} acc={best_acc:.4f}  λ-range={lam_range}')
    for t, a in sorted(g, key=lambda r: (reg_strength(r[0]) is None, reg_strength(r[0]) or 0)):
        print(f'   {t:<55s} {a:.4f}')
    print()

print('Markdown summary:')
print('| method | best λ | acc | λ range tested |')
print('|--------|--------|-----|----------------|')
for m, bl, ba, lr in summary:
    print(f'| {m} | {bl} | {ba:.4f} | {lr} |')
"
```

After running, summarize the best-per-method table back to the user in a short markdown table (include the `λ range tested` column from the script output).
