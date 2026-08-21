"""
================================================================================
  VIEWS/TAB_METAHEURISTIC_GUIDE.PY - METASEZGİSEL PROBLEM ADAPTASYON REHBERİ
================================================================================
  Bu sekme, Metasezgisel Algoritmaların (Metaheuristics) problemden bağımsız
  (problem-agnostic / black-box) çalışma prensiplerini, No Free Lunch (NFL)
  teoremini ve herhangi bir optimizasyon problemine adım adım nasıl adapte
  edileceğini detaylandıran kapsamlı teorik ve metodolojik rehberdir.
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np


def render_tab_metaheuristic_guide():
    """Metasezgisel Problemden Bağımsız Çatı ve Adaptasyon Rehberi sekmesini çizer."""
    
    st.markdown("## 🧩 Metasezgisel Algoritmalar: Problemden Bağımsız Çalışma & Adaptasyon Rehberi")
    st.markdown("""
    **Metasezgisel Algoritmalar (Metaheuristics)**; belirli bir probleme özel sezgisel kuralların 
    (örneğin en yakın komşuya git veya en çok kazandıranı çantaya at gibi) ötesinde, 
    **her türlü zorlu optimizasyon problemine uygulanabilen yüksek seviyeli, problemden bağımsız (problem-agnostic)** 
    arama stratejileridir.
    """)

    # ==============================================================================
    # 1. BÖLÜM: PROBLEM-AGNOSTIC PRENSİBİ VE KARA KUTU MODELİ
    # ==============================================================================
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0fdf4 0%, #e0f2fe 100%); border: 1px solid #bae6fd; border-radius: 12px; padding: 1.4rem; margin-bottom: 1.8rem;">
        <h4 style="color: #0369a1; margin-top: 0; margin-bottom: 0.6rem;">📌 Temel Soru: Metasezgiseller Problemin Ne Olduğunu Bilmeden Nasıl Çalışır?</h4>
        <p style="margin-bottom: 0.6rem; font-size: 1rem; color: #1e293b; line-height: 1.6;">
            Matematiksel kesin çözücüler (ILP / Simplex / Branch & Bound); problemin içindeki her bir eşitliği, eşitsizliği (<i>Ax ≤ b</i>) ve katsayıyı tam olarak bilmek zorundadır. 
            Buna karşılık metasezgisel algoritmalar problemi bir <b>"Kara Kutu Değerlendirici (Black-Box Evaluator)"</b> olarak görür.
        </p>
        <p style="margin-bottom: 0; font-size: 0.95rem; color: #334155;">
            🔍 Algoritma; çözdüğü şeyin bir <i>çelik fabrikası vardiyası</i> mı, bir <i>uçak rotası</i> mı yoksa bir <i>finansal portföy</i> mü olduğunu bilmez. 
            Yalnızca kendisine sunulan adaya bir <b>değişim operatörü (perturbation/move)</b> uygular ve kara kutudan dönen <b>sayısal kalite puanına (Fitness / Cost Z)</b> göre yönünü belirler!
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""<div class="card-box" style="border-left: 5px solid #2563eb; height: 100%;">
<div class="card-title" style="color: #1e40af;">🎯 Kesin Çözücüler (Exact Solvers - ILP / CP)</div>
<p class="card-text"><b>Matematiksel Beyaz Kutu (White-Box):</b></p>
<ul class="card-text" style="font-size: 0.92rem; padding-left: 1.2rem;">
    <li>Problemin tüm kısıtlarını doğrusal veya tamsayı denklemler halinde açıkça ister.</li>
    <li>Matematiksel modelde en ufak bir kural değiştiğinde (örneğin karesel ceza eklendiğinde) çözücünün tüm yapısı çöker veya yeniden kurulmalıdır.</li>
    <li><b>Avantaj:</b> Kanıtlanabilir küresel optimumu (Optimal Guarantee) ve alt sınırı (Lower Bound) bulur.</li>
    <li><b>Dezavantaj:</b> Problem boyutu büyüdükçe NP-Hard kombinatoryal patlama nedeniyle kilitlenir.</li>
</ul>
</div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""<div class="card-box" style="border-left: 5px solid #10b981; height: 100%;">
<div class="card-title" style="color: #065f46;">🚀 Metasezgisel Çözücüler (Metaheuristics)</div>
<p class="card-text"><b>Evrensel Arama Motoru (Black-Box Engine):</b></p>
<ul class="card-text" style="font-size: 0.92rem; padding-left: 1.2rem;">
    <li>Problemin iç kısıt matematiğiyle ilgilenmez; sadece <code>f(S) -> Maliyet</code> arayüzünü çağırır.</li>
    <li>Doğrusal olmayan (non-linear), süreksiz, simülasyon tabanlı veya karmaşık kurallar doğrudan amaç fonksiyonuna gömülebilir.</li>
    <li><b>Avantaj:</b> Milyarlarca olası kombinasyon içeren devasa uzaylarda saniyeler içinde mükemmele yakın pratik çözümler üretir.</li>
    <li><b>Dezavantaj:</b> Bulunan çözümün matematiksel olarak %100 mutlak en iyi olduğunu kanıtlayamaz.</li>
</ul>
</div>""", unsafe_allow_html=True)

    st.divider()

    # ==============================================================================
    # 2. BÖLÜM: HER PROBLEME ADAPTE EDİLEBİLİR Mİ? (NO FREE LUNCH TEOREMİ)
    # ==============================================================================
    st.markdown("### 🌐 Her Optimizasyon Problemine Adapte Edilebilir mi?")
    st.markdown("""
    **Evet!** Bir optimizasyon probleminin metasezgisellerle çözülebilmesi için **3 temel şartın** sağlanması yeterlidir:
    """)

    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("""<div class="metric-card" style="text-align: left; padding: 1rem;">
<h5 style="color: #1d4ed8; margin-top:0;">1️⃣ Çözüm Kodlanabilir Olmalı</h5>
<p style="font-size: 0.9rem; color: #475569; margin: 0;">
Problemin bir adayı bilgisayar hafızasında bir vektör, dizi, matris veya permütasyon (Encoding) olarak ifade edilebilmelidir.
</p>
</div>""", unsafe_allow_html=True)

    with p2:
        st.markdown("""<div class="metric-card" style="text-align: left; padding: 1rem;">
<h5 style="color: #047857; margin-top:0;">2️⃣ Kalite Ölçülebilir Olmalı</h5>
<p style="font-size: 0.9rem; color: #475569; margin: 0;">
Herhangi bir aday çözüm verildiğinde, onun ne kadar iyi veya kötü olduğunu belirten sayısal bir skor (Fitness / Cost Z) hesaplanabilmelidir.
</p>
</div>""", unsafe_allow_html=True)

    with p3:
        st.markdown("""<div class="metric-card" style="text-align: left; padding: 1rem;">
<h5 style="color: #b45309; margin-top:0;">3️⃣ Komşu Üretilebilir Olmalı</h5>
<p style="font-size: 0.9rem; color: #475569; margin: 0;">
Mevcut bir çözümden küçük bir değişiklik (Swap, Flip, Insert, Crossover) ile yeni bir aday çözüm türetilebilmelidir.
</p>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # NO FREE LUNCH KUTUSU
    st.markdown("""
    <div style="background-color: #fffbeb; border: 1px solid #fef3c7; border-left: 6px solid #f59e0b; border-radius: 8px; padding: 1.2rem; margin-bottom: 1.5rem;">
        <h4 style="color: #b45309; margin-top: 0; margin-bottom: 0.5rem;">⚖️ No Free Lunch (NFL) Teoremi (Wolpert & Macready, 1997)</h4>
        <p style="font-size: 0.95rem; color: #78350f; margin-bottom: 0.4rem; line-height: 1.5;">
            <b>"Tüm olası optimizasyon problemleri üzerinde ortalama alındığında, hiçbir metasezgisel algoritma diğerinden üstün değildir; hatta rastgele aramadan bile daha iyi değildir!"</b>
        </p>
        <p style="font-size: 0.92rem; color: #92400e; margin-bottom: 0;">
            💡 <b>Bu Teorem Bize Ne Söyler?</b> Genel geçer bir algoritmayı körü körüne alıp bir probleme doğrudan atarsanız mucize bekleyemezsiniz. 
            Bir algoritmanın üstün başarı göstermesinin tek yolu; <u>arama operatörlerini ve kısıt yapılarını problemin alan bilgisine (Domain Knowledge) göre özel olarak tasarlamaktır</u>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ==============================================================================
    # 3. BÖLÜM: METASEZGİSEL ADAPTASYONUN 5 EVRENSEL DİREĞİ (THE 5 PILLARS)
    # ==============================================================================
    st.markdown("### 🏛️ Yeni Bir Probleme Metasezgisel Uyarlamanın 5 Evrensel Temel Direği")
    st.markdown("""
    Herhangi bir yeni optimizasyon problemini (Lojistik, Fabrika Planlama, Finans, Oyun Yapay Zekası, Biyoinformatik) 
    metasezgisel algoritmalarla çözmek için aşağıdaki **5 adımlı mimari formül** uygulanır:
    """)

    tab_p1, tab_p2, tab_p3, tab_p4, tab_p5 = st.tabs([
        "1. Temsil (Representation)",
        "2. Amaç Fonksiyonu (Objective)",
        "3. Komşuluk & Hareket (Move)",
        "4. Kısıt Yönetimi (Constraints)",
        "5. Algoritmik Mekanizma (Engine)"
    ])

    with tab_p1:
        st.markdown("#### 🧬 1. Direk: Çözüm Temsili ve Kodlama (Solution Encoding)")
        st.markdown("""
        Problemin doğasına uygun bir veri yapısı seçilir:
        """)
        enc_df = pd.DataFrame([
            {"Temsil Türü": "Permütasyon Dizisi", "Format": "[3, 1, 4, 2, 5]", "Uygun Problemler": "Gezgin Satıcı (TSP), Akış Tipi Çizelgeleme (Flow-Shop), Sıralama", "Örnek Anlam": "Şehirlerin veya makinelerin ziyaret sırası."},
            {"Temsil Türü": "İkili (Binary Bitstring)", "Format": "[1, 0, 1, 1, 0]", "Uygun Problemler": "0/1 Sırt Çantası (Knapsack), Küme Kapsama (Set Covering)", "Örnek Anlam": "1: Nesne seçildi / alındı, 0: Alınmadı."},
            {"Temsil Türü": "Ayrık Matris / Tablo", "Format": "W × D Matrisi (Değerler in {0,1,2,3})", "Uygun Problemler": "Vardiya Çizelgeleme (NSP), Ders Programı (Timetabling)", "Örnek Anlam": "Satır işçiyi, sütun günü, hücre vardiyayı (G, A, N, OFF) gösterir."},
            {"Temsil Türü": "Sürekli Reel Vektör", "Format": "[0.42, -1.85, 3.14]", "Uygun Problemler": "Yapay Sinir Ağı Ağırlıkları, Roket Yörünge Optimizasyonu", "Örnek Anlam": "Sonsuz reel uzaydaki koordinat değerleri."}
        ])
        st.table(enc_df)

    with tab_p2:
        st.markdown("#### 🎯 2. Direk: Amaç ve Uygunluk Fonksiyonu (Objective Function / Fitness)")
        st.markdown("""
        Bir çözümün ne kadar kaliteli olduğunu tek bir sayıya indirgeyen skaler formülasyondur:
        * **Minimizasyon Problemleri:** Maliyet, gecikme, kısıt cezaları veya enerji harcaması minimize edilir ($Z \\to \\min$).
        * **Maksimizasyon Problemleri:** Kâr, verimlilik veya kapsama maksimize edilir ($Z \\to \\max$).
        * **Ağırlıklı Toplam Modeli (Weighted Sum):** Birden fazla çelişen hedef ağırlık katsayılarıyla ($w_i$) birleştirilir:
        """)
        st.latex(r"Z(S) = \sum_{j=1}^{M} w_j \cdot \text{Ceza}_j(S)")

    with tab_p3:
        st.markdown("#### 🔄 3. Direk: Komşuluk & Değişim Operatörleri (Neighborhood Operators)")
        st.markdown("""
        Mevcut çözüm $S$'den yeni bir aday $S'$ üretmek için problemin geometrisine uygun mikro operatörler tasarlanır:
        """)
        col_op1, col_op2 = st.columns(2)
        with col_op1:
            st.markdown("""
            * **2-Opt / Swap (Takas):** İki elemanın pozisyonunu yer değiştirir (TSP ve NSP'de temel operatör).
            * **Bit-Flip / 1-Opt:** Bir elemanın durumunu değiştirir ($0 \\leftrightarrow 1$).
            * **Insert / Shift (Araya Ekleme):** Bir elemanı yerinden çıkarıp araya sokar.
            """)
        with col_op2:
            st.markdown("""
            * **Inversion (Ters Çevirme):** Bir alt bloğu ters yüz eder.
            * **Crossover (Çaprazlama):** İki ebeveynin iyi parçalarını birleştirerek çocuk üretir.
            * **Multi-Neighborhood (VNS):** Mikro ($N_1$), Mezo ($N_2$) ve Makro ($N_3$) hiyerarşik operatörler.
            """)

    with tab_p4:
        st.markdown("#### 🛡️ 4. Direk: Kısıt Yönetim Stratejisi (Constraint Handling)")
        st.markdown("""
        Gerçek hayatta kısıtları ihlal eden geçersiz adaylarla karşılaşıldığında uygulanan 4 temel strateji:
        """)
        st.markdown("""
        1. **Ölüm Cezası / Reddetme (Death Penalty / Reject):** Sert kısıtları bozan adayı derhal çöpe atıp ceza fonksiyonunu dahi çağırmamak (Bizim projemizdeki `check_swap_feasibility` yaklaşımı).
        2. **Onarım Operatörü (Repair Mechanism):** Bozulan kuralı akıllı bir düzeltme adımıyla otomatik onarmak.
        3. **Ceza Fonksiyonu (Penalty Method):** İhlal miktarını büyük bir katsayıyla ($M$) çarpıp amaç fonksiyonuna eklemek: $Z' = Z + M \\cdot \\text{İhlal}$.
        4. **Geçerli Uzayda Kalma (Feasible-Preserving Moves):** Operatörleri öyle tasarlamak ki, hiçbir hamle geçerli uzayın dışına çıkamasın.
        """)

    with tab_p5:
        st.markdown("#### ⚙️ 5. Direk: Algoritmaya Özgü Mekanizmaların Entegrasyonu")
        st.markdown("""
        Her metasezgisel algoritma bu 4 temel direğin üzerine kendi özgün felsefesini ekler:
        """)
        st.markdown("""
        * **Simulated Annealing (Tavlama):** Başlangıç sıcaklığı $T_0$, soğuma katsayısı $\\alpha$ ve Metropolis kabul kuralı $\\exp(-\\Delta Z / T)$.
        * **Tabu Search (Tabu Arama):** Yasaklı nitelik tanımı (örn: `(işçi, gün)` çifti), Tabu Tenure $L$ ve Aspirasyon kriteri.
        * **Genetik / Memetik Algoritma:** Popülasyon boyutu, turnuva seçimi, ebeveyn çaprazlaması ve Lamarckian yerel arama derinliği.
        * **Değişken Komşuluk (VNS):** Sarsma (Shaking) ile yerel çukurlardan sıçrama ve sistematik komşuluk değiştirme.
        * **Parçacık Sürü (PSO):** İki çözüm arasındaki fark takas dizisi ($A \\ominus B$), atalet ağırlığı $w$ ve bilişsel/sosyal çekim ($c_1, c_2$).
        * **Karınca Kolonisi (ACO):** Kararların bırakacağı feromon matrisi $\\tau$ ve yerel cazibe $\\eta$ dengesi.
        """)
    st.divider()

    st.markdown("""
    <div style="text-align: center; margin-top: 1.2rem; padding: 1.2rem; background-color: #f8fafc; border-radius: 8px; border: 1px dashed #cbd5e1;">
        <span style="color: #334155; font-size: 1rem; font-weight: 600;">
            👉 Bir sonraki sekmelerde (Sekme 7'den Sekme 14'e kadar), yukarıda açıklanan bu evrensel adaptasyon prensiplerinin 
            <b>Hill Climbing, Simulated Annealing, Genetik Algoritma, Memetik Algoritma, Tabu Search, VNS, Discrete PSO ve ACO</b> 
            çözücülerimiz üzerinde canlı olarak nasıl çalıştığını adım adım deneyimleyebilirsiniz!
        </span>
    </div>
    """, unsafe_allow_html=True)
