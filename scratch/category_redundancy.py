import glob
import re

def analyze_specific_redundancies():
    files = {}
    for fpath in glob.glob('views/*.py') + glob.glob('algorithms/*.py') + ['app.py', 'global_state.py']:
        with open(fpath, 'r', encoding='utf-8') as f:
            files[fpath] = f.read()

    categories = {
        "1. Çelik Sektörü 4 Posta 3 Vardiya Çalışma Sistemi & Dinamikleri": {
            "keywords": ["4 posta", "3 vardiya", "00:00-08:00", "08:00-16:00", "16:00-24:00", "Sürekli döküm", "Yüksek fırın", "Haddehane"],
            "description": "Çelik sektöründe kesintisiz üretimin (7/24) 4 posta 3 vardiya sistemi ile nasıl yürüdüğü, postaların dinlenme/çalışma döngüleri ve neden esnek çizelgeleme gerektiği bilgisi."
        },
        "2. Sert Kısıtların (Hard Constraints - HC1..HC5) Açıklamaları": {
            "keywords": ["HC1", "HC2", "HC3", "HC4", "HC5", "günde en fazla 1", "11 saat dinlenme", "4 MYK", "Kritik MYK", "asgari kadro"],
            "description": "Kanuni ve operasyonel sert kısıtlar (Günde maks 1 vardiya, vardiyalar arası min 11 saat dinlenme, 4 kritik MYK sertifikası şartı, haftalık izin zorunluluğu) birden çok sekmede tekrar tekrar açıklanıyor."
        },
        "3. Yumuşak Kısıtlar ve Ceza Puanları (SC1..SC5 / min Z Formülasyonu)": {
            "keywords": ["SC1", "SC2", "SC3", "SC4", "SC5", "1000", "500", "100", "50", "20", "Gece ardışık", "Hafta sonu tatili", "İzin talepleri", "İş yükü dengesi", "min Z"],
            "description": "Yumuşak kısıtların ceza katsayıları (1000, 500, 100, 50, 20 puan) ve matematiksel amaç fonksiyonu formülü (min Z) hem teorik sekmelerde hem de her çözücünün kendi sekmesinde listeleniyor."
        },
        "4. Algoritmaların Teorik Çalışma Prensipleri & 'Ne Yapar / Nasıl Çalışır' Kartları": {
            "keywords": ["Nasıl Çalışır", "Algoritma Dinamikleri", "Lokal Arama", "Komşuluk Yapısı", "Mutasyon", "Çaprazlama", "Sıcaklık Azaltma", "Fermon Güncelleme", "Parçacık Hızı", "MIP Gap", "Arama Uzayı"],
            "description": "Her bir algoritmanın (Greedy, CSP, ILP, Hill Climbing, SA, GA, Memetic, Tabu Search, VNS, PSO, ACO, CP-SAT) teorik tanımı ve mekanizması hem kendi sekmesindeki info kartlarında hem de Sekme Karşılaştırma (tab_comparison) teorik matrisinde/tablolarında yineleniyor."
        },
        "5. Ön-Analiz / Tahmini Ceza Değerlendirme Çağrısı (Evaluator CPU Maliyeti) Rozetleri ve Formülü": {
            "keywords": ["render_evaluator_cost_badge", "render_preanalysis_prediction_card", "Tahmini Ceza Değerlendirme", "Birim Maliyet", "Saf CPU"],
            "description": "Her iteratif metasezgiselin çalıştırma butonu üstünde ceza hesaplama formülü (örneğin ~10,000 çağrı x 0.015 ms = saf CPU süresi) ve formül açıklaması ortak fonksiyon ve benzer metinlerle 8+ sekmede yer alıyor."
        },
        "6. Eşitlik / Adillik Metrikleri (Gini Katsayısı, Shannon Entropisi, Standart Sapma)": {
            "keywords": ["Gini", "Shannon", "Entropi", "Standart Sapma", "Adillik", "Varyans", "Adil Dağılım"],
            "description": "İş yükü ve gece vardiyası dağılımının ne kadar adil olduğunu ölçen Gini, Shannon ve Standart Sapma metriklerinin ne anlama geldiği, nasıl yorumlanacağı neredeyse tüm çözücü sekmelerinin sonuç alanında ve karşılaştırma sekmesinde tekrarlanıyor."
        },
        "7. Vardiya Kodları & Çizelge Lejant Açıklaması (0=İzin, 1=Sabah, 2=Akşam, 3=Gece)": {
            "keywords": ["0: İzin", "1: Sabah", "2: Akşam", "3: Gece", "08:00-16:00", "16:00-24:00", "00:00-08:00", "Vardiya 1", "Vardiya 2", "Vardiya 3"],
            "description": "Vardiya matrislerinin altındaki lejant / açıklama metinleri ve renk kodları her çizelge görselleştirmesinde tekrar ediliyor."
        },
        "8. Çözücü Durum / Başarı / Kısıt Uygunluk Mesajları (Standart UI Şablonları)": {
            "keywords": ["Çözücünün Çalışmayı Bitirme Nedeni", "Sert Kısıt Uygunluğu", "%100 GEÇERLİ ÇÖZÜM", "Otomatik Kısıt Kurtarma Bildirimi"],
            "description": "Her çözücünün sonuç ekranındaki bildirim kutuları, kısıt başarı yüzdeleri ve bitirme nedenleri birebir aynı format ve benzer kalıplarla tekrarlanıyor."
        }
    }

    results = {}
    for cat_name, cat_data in categories.items():
        found_in = []
        for fpath, content in files.items():
            matches = [kw for kw in cat_data['keywords'] if kw.lower() in content.lower()]
            if len(matches) >= 2 or (len(matches) >= 1 and len(cat_data['keywords']) <= 3):
                # count occurrences
                count = sum(content.lower().count(kw.lower()) for kw in matches)
                found_in.append((fpath, count, matches))
        results[cat_name] = {
            'description': cat_data['description'],
            'found_in': found_in
        }
        
    with open('scratch/category_redundancy_summary.txt', 'w', encoding='utf-8') as out:
        for cat_name, data in results.items():
            out.write(f"\n{'='*70}\n{cat_name}\n{'='*70}\n")
            out.write(f"Açıklama: {data['description']}\n")
            out.write(f"Bulunduğu Dosya Sayısı: {len(data['found_in'])}\n")
            out.write("Dosyalar:\n")
            for fpath, cnt, kw in sorted(data['found_in'], key=lambda x: x[1], reverse=True):
                out.write(f"  - {fpath:35} (Eşleşme sıklığı: {cnt:3} | Anahtar Kelimeler: {kw[:4]})\n")

if __name__ == '__main__':
    analyze_specific_redundancies()
    print("Category redundancy summary generated.")
