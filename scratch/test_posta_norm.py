import re

def normalize_posta_name(raw_val, default_idx=0):
    if raw_val is None:
        return f"Posta {['A','B','C','D'][default_idx % 4]}"
    s = str(raw_val).strip().upper()
    if not s or s in ('NAN', 'NONE'):
        return f"Posta {['A','B','C','D'][default_idx % 4]}"
    
    cleaned = re.sub(r'POSTA', '', s).strip()
    
    if 'D' in cleaned or s.endswith('D') or '4' in cleaned: return 'Posta D'
    if 'C' in cleaned or s.endswith('C') or '3' in cleaned: return 'Posta C'
    if 'B' in cleaned or s.endswith('B') or '2' in cleaned: return 'Posta B'
    if 'A' in cleaned or s.endswith('A') or '1' in cleaned: return 'Posta A'
    
    return f"Posta {['A','B','C','D'][default_idx % 4]}"

tests = ['a', 'A', 'posta a', 'POSTA A', 'Posta-A', 'b', 'B', 'posta b', 'POSTA B', 'Posta-B', 'c', 'C', 'posta c', 'POSTA C', 'Posta-C', 'd', 'D', 'posta d', 'POSTA D', 'POSTA-D', '1', '2', '3', '4']
for t in tests:
    print(f"{t:<10} -> {normalize_posta_name(t)}")
