import glob
import os
import re

def analyze_all_files():
    views = sorted(glob.glob('views/*.py'))
    other_files = ['app.py', 'global_state.py', 'config.py'] + sorted(glob.glob('algorithms/*.py'))
    all_files = views + other_files
    
    file_info = {}
    
    for fpath in all_files:
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract markdown strings
        md_matches = re.findall(r'st\.markdown\(\s*(?:r|f|rf|fr)?("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"[^"\n]*"|\'[^\'\n]*\')', content)
        expanders = re.findall(r'st\.expander\(\s*(?:r|f|fr|rf)?["\'](.*?)["\']', content)
        docstrings = re.findall(r'"""([\s\S]*?)"""', content)
        
        file_info[fpath] = {
            'content': content,
            'md_matches': md_matches,
            'expanders': expanders,
            'docstrings': docstrings
        }
    
    print("=== SUMMARY OF LOADED FILES ===")
    for k, v in file_info.items():
        print(f"{k:35}: {len(v['md_matches'])} markdown calls, {len(v['expanders'])} expanders, {len(v['docstrings'])} docstrings")

if __name__ == '__main__':
    analyze_all_files()
