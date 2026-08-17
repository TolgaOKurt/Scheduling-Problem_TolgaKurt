"""
================================================================================
  VIEWS/TAB1_NSP_INTRO.PY - SEKME 1: HEMŞİRE ÇİZELGELEME PROBLEMİ (NSP/NRP)
================================================================================
"""
import streamlit as st

def render_tab1():
    """Sekme 1 içeriğini çizer: NSP tanımı, tarihçesi, kısıt mimarisi ve NP-Hard analizi."""
    st.markdown("## 📖 Hemşire Çizelgeleme Problemi (NSP / NRP) Nedir?")
    st.markdown("""
    **Hemşire Çizelgeleme Problemi (Nurse Scheduling Problem - NSP)** veya uluslararası literatürdeki adıyla 
    **Hemşire Nöbet Planlama Problemi (Nurse Rostering Problem - NRP)**; personelin belirli bir planlama periyodunda 
    (haftalık, iki haftalık veya aylık), yasal mevzuatlara, kurumsal ihtiyaçlara ve çalışan tercihlerine uygun olarak 
    vardiyalara **en uygun (optimal)** şekilde atanmasını amaçlayan klasik ve zorlu bir **Kombinatoryal Optimizasyon** problemidir.
    """)

    # 1. ÇİZELGELEME HİYERARŞİSİ VE KAVRAMSAL BAĞLAM
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0fdf4 0%, #e0f2fe 100%); border: 1px solid #bae6fd; border-radius: 10px; padding: 1.2rem; margin-bottom: 1.5rem;">
        <h4 style="color: #0369a1; margin-top: 0; margin-bottom: 0.5rem;">📌 Çizelgeleme Literatüründe NSP'nin Yeri: Timetabling vs. Rostering</h4>
        <p style="margin-bottom: 0.5rem; font-size: 0.98rem; color: #1e293b;">
            Yöneylem araştırmasında çizelgeleme problemleri temelde üç ana kategoride incelenir:
        </p>
        <ul style="margin-bottom: 0; font-size: 0.95rem; color: #334155;">
            <li><b>Timetabling (Zaman Çizelgeleme):</b> Etkinliklerin, derslerin veya sınavların belirli zaman dilimlerine ve mekanlara atanması (Örn: Üniversite ders programı).</li>
            <li><b>Staff Rostering / NSP (Personel Nöbet Çizelgeleme):</b> 7/24 kesintisiz hizmet veren dinamik organizasyonlarda, bireylerin yasal sınırlar, uzmanlıklar ve ergonomik kurallar doğrultusunda vardiyalara atanması.</li>
            <li><b>Job-Shop / Flow-Shop Sequencing (İş Sıralama):</b> Üretim hatlarında işlerin makinelere geçiş sıralaması ve sürelerinin optimizasyonu.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""<div class="card-box">
<div class="card-title">🎯 Problemin Temel Amacı ve Optimizasyon Boyutları</div>
<p class="card-text">
NSP, sadece matematiksel bir atama problemi olmayıp aynı zamanda <b>insan kaynakları yönetimi, iş sağlığı-güvenliği ve kurumsal verimlilik</b> kesişiminde yer alır:
</p>
<ul class="card-text" style="padding-left: 1.2rem;">
    <li><b>Operasyonel Kapsama (Coverage):</b> 7/24 kesintisiz çalışan her vardiyada (Gündüz, Akşam, Gece) hizmetin aksamaması için yeterli sayıda ve doğru nitelikte personelin hazır bulunması.</li>
    <li><b>Yasal ve Ergonomik Güvence:</b> 4857 Sayılı İş Kanunu ve İSG mevzuatlarına uyularak aşırı çalışma, yetersiz dinlenme ve tükenmişlik riskinin önlenmesi.</li>
    <li><b>Çalışan Memnuniyeti ve Adalet:</b> Nöbet yüklerinin (özellikle gece ve hafta sonu) çalışanlar arasında eşit paylaştırılması ve kişisel izin taleplerinin karşılanması.</li>
    <li><b>Sektörler Arası Genellenebilirlik:</b> NSP için geliştirilen matematiksel modeller; sağlık, havacılık, enerji şebekeleri ve çelik fabrikaları gibi tüm vardiyalı sanayi kollarına doğrudan uyarlanabilir.</li>
</ul>
</div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""<div class="card-box">
<div class="card-title">⏳ Tarihsel Gelişim ve Literatür</div>
<p class="card-text" style="font-size: 0.95rem;">
NSP'nin yöneylem araştırması literatüründeki dönüm noktaları:
</p>
<ul style="padding-left: 1.1rem; font-size: 0.92rem; color: #334155;">
    <li><b>1950'ler (İlk Adımlar):</b> Doğrusal programlama ve basit sezgisel kurallarla bilgisayar destekli ilk çalışan çizelgeleme denemeleri.</li>
    <li><b>1976 (Matematiksel Temeller):</b> <i>D.M. Warner</i> ve <i>Ahuja</i> tarafından NSP'nin modern Karmaşık Tamsayılı Doğrusal Programlama (MILP) modellerinin yayınlanması.</li>
    <li><b>1990'lar (Metasezgiseller):</b> Problemin NP-Hard yapısı nedeniyle Genetik Algoritmalar, Tavlama Benzetimi ve Tabu Arama gibi yaklaşımların yaygınlaşması.</li>
    <li><b>2000+ (Çok Amaçlı ve Hibrit Yaklaşımlar):</b> <i>Burke vd. (2004)</i> öncülüğünde çalışan tercihleri ile operasyonel maliyetleri eşzamanlı optimize eden çok amaçlı (Pareto) ve hibrit çözücüler (Matheuristics).</li>
</ul>
</div>""", unsafe_allow_html=True)

    st.divider()

    # 2. KISIT YAPISI: SERT VE YUMUŞAK KISITLAR
    st.markdown("### ⚖️ Kısıt Mimarisi: Sert (Hard) vs. Yumuşak (Soft) Kısıtlar")
    st.markdown("""
    Optimizasyon modellerinde kısıtlar iki temel sınıfa ayrılır. Bu ayrım, bir çizelgenin **"uygulanabilir (feasible)"** olup olmadığını ve **"çözüm kalitesini"** belirler:
    """)

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #dc2626; background-color: #fff8f8;">
<div class="card-title" style="color: #991b1b;">
<span class="badge-hard">ZORUNLU</span> Sert Kısıtlar (Hard Constraints)
</div>
<p style="color: #7f1d1d; font-size: 0.95rem; font-weight: 600;">
Çözümün geçerli (feasible) sayılması için %100 oranında sağlanmalıdır. Tek bir ihlal dahi planı uygulanamaz kılar.
</p>
<ul style="font-size: 0.93rem; color: #1e293b;">
    <li><b>Kesintisiz Minimum Vardiya İhtiyacı:</b> Her vardiyada (Gündüz, Akşam, Gece) sahada bulunması zorunlu asgari personel sayısının tam karşılanması.</li>
    <li><b>Çakışma Yasağı (Tek Vardiya / Gün):</b> Bir çalışanın aynı gün içinde yalnızca 1 göreve (Gündüz, Akşam, Gece veya İzin) atanabilmesi.</li>
    <li><b>Kritik Ehliyet ve Nitelik Zorunluluğu:</b> Kritik operasyonlarda (ör. potacı, vinç operatörü, yoğun bakım uzmanı) zorunlu sertifikalı personelin eksiksiz bulunması.</li>
    <li><b>11 Saat Kesintisiz Dinlenme Kuralı:</b> Gece vardiyasından (24:00 - 08:00) çıkan personelin aynı günün sabahı (08:00 - 16:00) gündüz vardiyasına yazılamaması (Gece + Ertesi Gün Gündüz ≤ 1 kuralı).</li>
    <li><b>Zorunlu Hafta Tatili (Kayan 7 Günlük Pencere):</b> İSG standartları gereği herhangi bir kayan 7 günlük aralıkta en fazla 6 gün kesintisiz çalışma yapılabilmesi; en az 1 gün dinlenme (OFF) hakkının garanti edilmesi.</li>
</ul>
</div>""", unsafe_allow_html=True)

    with col_c2:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #2563eb; background-color: #f8faff;">
<div class="card-title" style="color: #1e40af;">
<span class="badge-soft">ESNEK</span> Yumuşak Kısıtlar (Soft Constraints)
</div>
<p style="color: #1e3a8a; font-size: 0.95rem; font-weight: 600;">
İhlali durumunda plan yasal olarak geçerlidir; ancak amaç fonksiyonuna ceza puanı eklenerek çizelgenin kalitesi düşer.
</p>
<ul style="font-size: 0.93rem; color: #1e293b;">
    <li><b>Kişisel İzin Talepleri:</b> Çalışanların önceden bildirdiği mazeret ve özel izin tercihlerine azami düzeyde uyulması.</li>
    <li><b>Sirkadiyen Ritim Uyumu:</b> Biyolojik saat sağlığı için vardiya geçişlerinin ileri yönde (Gündüz → Akşam → Gece → İzin) ilerlemesi; ters yönlü ani geçişlerin önlenmesi.</li>
    <li><b>Adil Gece Nöbeti Dağılımı:</b> Gece vardiyası yükünün tüm personel arasında homojen ve dengeli paylaştırılması.</li>
    <li><b>Kıdem ve Usta Dengesi:</b> Her vardiyada tecrübe ortalamasını korumak adına en az bir kıdemli usta/uzman ile yeni personelin eşleştirilmesi.</li>
    <li><b>Takım ve Posta Bütünlüğü:</b> Ekip uyumunu korumak için aynı postadaki personelin mümkün olduğunca aynı vardiyada birlikte görev yapması.</li>
</ul>
</div>""", unsafe_allow_html=True)

    st.divider()

    # 3. NP-HARD KARMAŞIKLIK VE KOMBİNATUVAR PATLAMA
    st.markdown("### 🧩 NP-Hard Karmaşıklık ve Arama Uzayının Boyutu")

    np_col1, np_col2 = st.columns([3, 2])

    with np_col1:
        st.markdown("""
        NSP, Hesaplamalı Karmaşıklık Teorisinde **NP-Hard (Non-deterministic Polynomial-time Hard)** sınıfına aittir.
        
        <i>N</i> çalışan, <i>D</i> gün ve <i>K</i> olası vardiya durumu (<i>K</i> = 4: Gündüz, Akşam, Gece, İzin) içeren bir modelde olası tüm çizelgelerin kombinatoryal arama uzayı boyutu:
        """, unsafe_allow_html=True)
        st.latex(r"|\Omega| = K^{N \times D} = 4^{N \times D}")
        st.markdown("""
        Personel veya gün sayısı doğrusal olarak arttığında, incelenmesi gereken olası çizelge sayısı **üstel (eksponansiyel)** olarak patlar. 
        Bu durum, **Kaba Kuvvet (Brute-Force)** yöntemleri ile en iyi çözümü aramanın imkansız olduğunu gösterir.
        """)

    with np_col2:
        st.markdown("""
        <div style="background-color: #ffffff; border: 2px solid #e2e8f0; border-radius: 10px; padding: 1.2rem; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
            <div style="font-weight: 700; color: #0f172a; margin-bottom: 0.6rem; font-size: 1.05rem;">📊 Örnek Boyut Karşılaştırması</div>
            <div style="font-size: 0.9rem; color: #334155; line-height: 1.6;">
                • <b>24 İşçi × 14 Gün:</b> Toplam 336 karar hücresi.<br>
                • <b>Olası Çözüm Sayısı:</b> 4<sup>336</sup> ≈ 3.75 × 10<sup>202</sup><br>
                • <i>(Karşılaştırma: Gözlemlenebilir evrendeki toplam atom sayısı yaklaşık 10<sup>80</sup>'dir).</i><br>
                • Saniyede 1 milyar çizelge test eden bir süper bilgisayarın tüm uzayı taraması evrenin yaşından trilyonlarca kat daha uzun sürer.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 4. OPTİMİZASYON YÖNTEMLERİ VE İŞ AKIŞI
    st.markdown("### 🔄 NSP Optimizasyon Süreci: 4 Aşamalı Karar Mekanizması")

    flow_col1, flow_col2, flow_col3, flow_col4 = st.columns(4)

    with flow_col1:
        st.markdown("""
        <div style="background: #ffffff; border-top: 4px solid #3b82f6; border-radius: 8px; padding: 1rem; box-shadow: 0 2px 6px rgba(0,0,0,0.04); height: 100%;">
            <div style="font-weight: bold; color: #1d4ed8; font-size: 0.95rem;">1. Girdi & Talep Analizi</div>
            <p style="font-size: 0.85rem; color: #475569; margin-top: 0.4rem;">
                Kadro büyüklüğü, vardiya ihtiyaçları, yetkinlikler ve kişisel izin talepleri sisteme tanımlanır.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with flow_col2:
        st.markdown("""
        <div style="background: #ffffff; border-top: 4px solid #ef4444; border-radius: 8px; padding: 1rem; box-shadow: 0 2px 6px rgba(0,0,0,0.04); height: 100%;">
            <div style="font-weight: bold; color: #b91c1c; font-size: 0.95rem;">2. Sert Kısıt Filtreleme</div>
            <p style="font-size: 0.85rem; color: #475569; margin-top: 0.4rem;">
                Yasal ve operasyonel kuralları ihlal eden çözümler elenerek yalnızca geçerli (feasible) çözüm uzayı belirlenir.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with flow_col3:
        st.markdown("""
        <div style="background: #ffffff; border-top: 4px solid #f59e0b; border-radius: 8px; padding: 1rem; box-shadow: 0 2px 6px rgba(0,0,0,0.04); height: 100%;">
            <div style="font-weight: bold; color: #b45309; font-size: 0.95rem;">3. Ceza Minimizasyonu</div>
            <p style="font-size: 0.85rem; color: #475569; margin-top: 0.4rem;">
                MILP, CSP veya Metasezgiseller (Genetik, SA, Tabu, Memetik) ile yumuşak kısıt ceza fonksiyonu (min Z) optimize edilir.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with flow_col4:
        st.markdown("""
        <div style="background: #ffffff; border-top: 4px solid #10b981; border-radius: 8px; padding: 1rem; box-shadow: 0 2px 6px rgba(0,0,0,0.04); height: 100%;">
            <div style="font-weight: bold; color: #047857; font-size: 0.95rem;">4. Çizelge & KPI Raporu</div>
            <p style="font-size: 0.85rem; color: #475569; margin-top: 0.4rem;">
                Nihai vardiya matrisi, iş yükü dengesi, talep karşılama oranları ve analitik grafikler kullanıcıya sunulur.
            </p>
        </div>
        """, unsafe_allow_html=True)
