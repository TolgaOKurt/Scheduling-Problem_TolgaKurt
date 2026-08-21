import glob
import re

def summarize_tab_structures():
    files = sorted(glob.glob('views/tab*.py'))
    
    with open('scratch/tab_structure_summary.txt', 'w', encoding='utf-8') as out:
        for fpath in files:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            out.write("="*80 + "\n")
            out.write(f"FILE: {fpath}\n")
            out.write("="*80 + "\n")
            
            # Find markdown headers (##, ###, ####)
            headers = re.findall(r'st\.markdown\(.*?([#]{2,4}\s+[^\n]+)', content)
            out.write("HEADERS:\n")
            for h in headers:
                out.write(f"  {h}\n")
                
            # Find card titles
            cards = re.findall(r'<div class="card-title"[^>]*>(.*?)</div>', content)
            out.write("\nCARD TITLES:\n")
            for c in cards:
                out.write(f"  - {c.strip()}\n")
                
            # Find colored alert boxes
            alerts = re.findall(r'<div style="background-color:\s*#[a-fA-F0-9]+[^>]*>.*?<b>(.*?)</b>', content)
            out.write("\nALERT BOX LABELS:\n")
            for a in alerts:
                out.write(f"  - {a.strip()}\n")
                
            out.write("\n\n")

if __name__ == '__main__':
    summarize_tab_structures()
    print("Done")
