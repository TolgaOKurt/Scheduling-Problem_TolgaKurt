import glob
import os
import re
from collections import defaultdict

def extract_texts():
    files = sorted(glob.glob('views/*.py') + ['app.py', 'global_state.py', 'config.py'])
    all_texts = []
    
    for fpath in files:
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # extract st.markdown / st.info / st.caption / st.success / st.warning strings
        matches = re.finditer(r'st\.(?:markdown|info|caption|write|success|warning|error)\(\s*(?:r|f|fr|rf)?("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"[^"\n]*"|\'[^\'\n]*\')', content)
        for m in matches:
            raw = m.group(1)
            if raw.startswith(('"""', "'''")):
                clean = raw[3:-3]
            elif raw.startswith(('"', "'")):
                clean = raw[1:-1]
            else:
                clean = raw
            clean = clean.strip()
            if len(clean) > 30 and not clean.startswith(('<div class="metric-card"', '<svg', 'style=')):
                all_texts.append({
                    'file': fpath,
                    'text': clean,
                    'line': content[:m.start()].count('\n') + 1
                })
                
    return all_texts

def find_duplicates(texts):
    normalized = defaultdict(list)
    for item in texts:
        norm_txt = re.sub(r'<[^>]+>', ' ', item['text'])
        norm_txt = re.sub(r'\s+', ' ', norm_txt).strip()
        if len(norm_txt) > 35:
            normalized[norm_txt].append(item)
            
    exact_dups = {k: v for k, v in normalized.items() if len(v) > 1}
    
    topics = {
        "1. 4 Posta 3 Vardiya & Çelik Tesis Yapısı Tanımı": [
            "4 posta", "3 vardiya", "00:00-08:00", "08:00-16:00", "16:00-24:00", "Posta A", "Posta B", "Posta C", "Posta D", "Sürekli döküm", "Yüksek fırın"
        ],
        "2. Sert Kısıtlar (Hard Constraints / Kanuni Zorunluluklar)": [
            "HC1", "HC2", "HC3", "Sert Kısıt", "Hard Constraint", "11 saat dinlenme", "günde en fazla 1", "min_workers_per_shift"
        ],
        "3. Yumuşak Kısıtlar ve Ağırlıklar (Soft Constraints - SC1, SC2, SC3, SC4, SC5)": [
            "SC1", "SC2", "SC3", "SC4", "SC5", "Yumuşak Kısıt", "Soft Constraint", "1000", "500", "100", "50", "20", "Gece vardiyası", "Hafta sonu", "Ardışık"
        ],
        "4. Matematiksel Amaç Fonksiyonu (min Z = ...)": [
            "min Z", "Amaç Fonksiyonu", "Objective Function", "Z =", "Ceza Puanı", "Toplam Ceza"
        ],
        "5. NSP / Yöneylem Araştırması (Operations Research) Tanımı & Tarihçesi": [
            "Operations Research", "Yöneylem", "Hemşire Çizelgeleme", "Nurse Scheduling", "NP-Hard", "kombinatoryal patlama", "Dantzig", "1960"
        ],
        "6. Çözücü Algoritma Teorileri (Lokal Arama, Stokastik, Popülasyon, ILP, vb.)": [
            "Lokal Arama", "Yerel Arama", "Popülasyon", "Deterministik", "Stokastik", "Metasezgisel", "Global Optimum", "Yerel Minimum", "Doğrusal Programlama"
        ],
        "7. Adillik & Eşitlik Metrikleri (Gini Katsayısı, Standart Sapma, Shannon Entropisi)": [
            "Gini", "Shannon", "Entropi", "Standart Sapma", "Adillik", "Varyans", "Adil Dağılım", "Eşitsizlik"
        ],
        "8. Ön-Analiz / Tahmini Maliyet Rozeti (CPU / Ceza Çağrısı)": [
            "Tahmini Ceza", "Birim Maliyet", "Saf CPU", "Değerlendirme", "Ön-Analiz"
        ],
        "9. Vardiya Kodları & Tablo Lejantı (0: Boş/İzin, 1: Sabah, 2: Akşam, 3: Gece)": [
            "Vardiya 1", "Vardiya 2", "Vardiya 3", "Sabah (08:00", "Akşam (16:00", "Gece (00:00", "0: İzin"
        ]
    }
    
    with open('scratch/redundancy_report.txt', 'w', encoding='utf-8') as out:
        out.write(f"TOPLAM METİN BLOKU SAYISI: {len(texts)}\n")
        out.write(f"BİREBİR VEYA ÇOK YAKIN KOPYA METİNLER ({len(exact_dups)} Küme):\n")
        for k, v in exact_dups.items():
            out.write(f"\n--- [Tekrar Sayısı: {len(v)}] Dosyalar: {[x['file'] + ':' + str(x['line']) for x in v]} ---\n")
            out.write(f"Metin: {k}\n")
            
        out.write("\n" + "="*80 + "\n")
        out.write("TEMATİK / ANLAMSAL BİLGİ TEKRARLARI (KONU BAZLI DAĞILIM):\n")
        out.write("="*80 + "\n")
        for topic, keywords in topics.items():
            out.write(f"\n==================================================\n")
            out.write(f"📌 {topic}\n")
            out.write(f"==================================================\n")
            occurrences = defaultdict(list)
            for item in texts:
                matched_kw = [kw for kw in keywords if kw.lower() in item['text'].lower()]
                if matched_kw:
                    occurrences[item['file']].append((matched_kw, item['line'], item['text']))
            for f, occ in occurrences.items():
                out.write(f"\n📂 {f} (Bu dosyada {len(occ)} farklı yerde geçiyor):\n")
                for o in occ:
                    # clean up snippet
                    snip = " ".join(o[2].split())
                    out.write(f"   • [Satır {o[1]}]: {snip[:160]}...\n")

if __name__ == '__main__':
    texts = extract_texts()
    find_duplicates(texts)
    print("Report generated successfully.")
