import glob
import os
import re

def detailed_metrics():
    views = sorted(glob.glob('views/*.py'))
    algorithms = sorted(glob.glob('algorithms/*.py'))
    root_files = ['app.py', 'global_state.py', 'config.py']
    all_files = views + algorithms + root_files
    
    total_lines = 0
    total_docstring_lines = 0
    total_markdown_lines = 0
    total_comment_lines = 0
    total_code_lines = 0
    
    file_stats = []
    
    for fpath in all_files:
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.splitlines()
            
        doc_lines = 0
        in_doc = False
        md_lines = 0
        comment_lines = 0
        code_lines = 0
        
        # docstrings & markdown detection
        # Match st.markdown blocks
        md_blocks = re.findall(r'st\.(?:markdown|info|caption|warning|success|error)\(\s*(?:r|f|fr|rf)?("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"[^"\n]*"|\'[^\'\n]*\')', content)
        md_text_total_lines = sum(b.count('\n') + 1 for b in md_blocks)
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if stripped.count('"""') == 2 or stripped.count("'''") == 2:
                    doc_lines += 1
                else:
                    in_doc = not in_doc
                    doc_lines += 1
            elif in_doc:
                doc_lines += 1
            elif stripped.startswith('#'):
                comment_lines += 1
            else:
                code_lines += 1
                
        total_lines += len(lines)
        total_docstring_lines += doc_lines
        total_comment_lines += comment_lines
        total_code_lines += code_lines
        total_markdown_lines += md_text_total_lines
        
        file_stats.append({
            'file': fpath,
            'total': len(lines),
            'doc': doc_lines,
            'comment': comment_lines,
            'md_lines': md_text_total_lines,
            'code': code_lines
        })
        
    print(f"Total Lines: {total_lines}")
    print(f"Docstring Lines: {total_docstring_lines}")
    print(f"Comment Lines: {total_comment_lines}")
    print(f"UI Markdown / Info Lines: {total_markdown_lines}")
    print(f"Executable Code Lines: {total_code_lines}")
    print(f"Information/Text Ratio: {(total_docstring_lines + total_comment_lines + total_markdown_lines) / total_lines * 100:.1f}%")
    
    with open('scratch/detailed_metrics.txt', 'w', encoding='utf-8') as out:
        out.write("FILE BREAKDOWN:\n")
        for s in file_stats:
            out.write(f"{s['file']:35}: Total {s['total']:4} | MD: {s['md_lines']:4} | Doc: {s['doc']:3} | Comm: {s['comment']:3} | Code: {s['code']:4}\n")

if __name__ == '__main__':
    detailed_metrics()
