import os
import ast
import tokenize
import io

def analyze_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    lines = content.splitlines()
    total_lines = len(lines)
    
    blank_lines = set()
    for i, line in enumerate(lines, 1):
        if not line.strip():
            blank_lines.add(i)
            
    # Token analysis for comments
    comment_lines = set()
    tokens = list(tokenize.generate_tokens(io.StringIO(content).readline))
    for tok in tokens:
        toktype, tokval, (srow, scol), (erow, ecol), line = tok
        if toktype == tokenize.COMMENT:
            # If the line up to the comment character has only whitespace, it's a pure comment line
            if line[:scol].strip() == '':
                comment_lines.add(srow)
                
    # AST analysis for docstrings and multiline string constants (markdown/HTML/text)
    docstring_lines = set()
    multiline_text_lines = set()
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            # Standalone docstring expressions (module, class, function docstrings or standalone string expressions)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                    for l in range(node.lineno, node.end_lineno + 1):
                        if l not in blank_lines and l not in comment_lines:
                            docstring_lines.add(l)
            # Other multiline strings (like in st.markdown("""..."""), HTML/CSS definitions)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if hasattr(node, 'lineno') and hasattr(node, 'end_lineno') and node.end_lineno > node.lineno:
                    for l in range(node.lineno + 1, node.end_lineno + 1):
                        if l not in blank_lines and l not in comment_lines and l not in docstring_lines:
                            multiline_text_lines.add(l)
    except Exception as e:
        print(f"Error parsing AST for {filepath}: {e}")
        
    # Standard Code (SLOC) = Total Lines - Blank Lines - Comment Lines - Standalone Docstring Lines
    standard_code_lines = set(range(1, total_lines + 1)) - blank_lines - comment_lines - docstring_lines
    
    # Pure Logic Code (excl. multiline embedded markdown/HTML/text)
    pure_logic_lines = standard_code_lines - multiline_text_lines
    
    return {
        'total': total_lines,
        'blank': len(blank_lines),
        'comment': len(comment_lines),
        'docstring': len(docstring_lines),
        'multiline_text': len(multiline_text_lines),
        'code_standard': len(standard_code_lines),
        'code_pure_logic': len(pure_logic_lines)
    }

def main():
    groups = {
        'Algoritmalar (algorithms/)': ['algorithms/' + f for f in sorted(os.listdir('algorithms')) if f.endswith('.py')],
        'Görünümler (views/)': ['views/' + f for f in sorted(os.listdir('views')) if f.endswith('.py')],
        'Uygulama & Durum (App / Config / State)': ['app.py', 'config.py', 'global_state.py']
    }
    
    grand_total = {'total': 0, 'blank': 0, 'comment': 0, 'docstring': 0, 'multiline_text': 0, 'code_standard': 0, 'code_pure_logic': 0}
    
    for group_name, files in groups.items():
        print(f"\n{'='*105}")
        print(f" {group_name.upper()}")
        print(f"{'='*105}")
        header = f"{'Dosya Adı':<35} | {'Toplam':>7} | {'Boş':>6} | {'Yorum(#)':>9} | {'Docstring':>10} | {'Net Kod':>9} | {'(Salt Mantık)':>13}"
        print(header)
        print("-" * len(header))
        
        group_total = {'total': 0, 'blank': 0, 'comment': 0, 'docstring': 0, 'multiline_text': 0, 'code_standard': 0, 'code_pure_logic': 0}
        
        for filepath in files:
            res = analyze_file(filepath)
            for k in group_total:
                group_total[k] += res[k]
                grand_total[k] += res[k]
            
            filename = os.path.basename(filepath)
            print(f"{filename:<35} | {res['total']:>7} | {res['blank']:>6} | {res['comment']:>9} | {res['docstring']:>10} | {res['code_standard']:>9} | {res['code_pure_logic']:>13}")
            
        print("-" * len(header))
        print(f"{'GRUP TOPLAMI':<35} | {group_total['total']:>7} | {group_total['blank']:>6} | {group_total['comment']:>9} | {group_total['docstring']:>10} | {group_total['code_standard']:>9} | {group_total['code_pure_logic']:>13}")

    print(f"\n{'='*105}")
    print(f" GENEL PROJE ÖZETİ")
    print(f"{'='*105}")
    print(f"{'GENEL TOPLAM':<35} | {grand_total['total']:>7} | {grand_total['blank']:>6} | {grand_total['comment']:>9} | {grand_total['docstring']:>10} | {grand_total['code_standard']:>9} | {grand_total['code_pure_logic']:>13}")
    print(f"{'='*105}")

if __name__ == '__main__':
    main()
