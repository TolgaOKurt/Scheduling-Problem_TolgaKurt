"""
================================================================================
  VIEWS/TAB1_NSP_INTRO.PY - SEKME 1: HEMŞİRE ÇİZELGELEME PROBLEMİ (NSP/NRP)
================================================================================
"""
import streamlit as st

def render_tab1():
    """Sekme 1 içeriğini çizer: NSP tanımı, tarihçesi ve kısıt kartları."""
    st.markdown("## 📖 Hemşire Çizelgeleme Problemi (NSP / NRP) Nedir?")
    
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""<div class="card-box">
<div class="card-title">📌 Temel Tanım ve Yöneylem Araştırması Bağlamı</div>
<p class="card-text">
<b>Hemşire Çizelgeleme Problemi (Nurse Scheduling Problem - NSP)</b> veya diğer bilinen adıyla <b>Hemşire Nöbet Planlama Problemi (Nurse Rostering Problem - NRP)</b>, yöneylem araştırmasında (Operations Research) hemşirelerin belirli bir zaman periyodunda (haftalık veya aylık) vardiyalara <b>en uygun (optimal)</b> şekilde atanmasını hedefleyen bir kombinatoryal optimizasyon problemidir.
</p>
<p class="card-text">
Problemin temel amacı, hastanenin veya kurumun operasyonel ihtiyaçlarını eksiksiz karşılarken yasal mevzuatlara uyulmasını sağlamak ve çalışanların kişisel tercihlerini maksimum düzeyde göz önünde bulundurmaktır.
</p>
<p class="card-text">
Hemşire çizelgeleme problemine geliştirilen çözüm yöntemleri ve matematiksel modeller, sadece sağlık sektöründe kalmayıp <b>kısıt tabanlı vardiya planlaması gerektiren tüm sanayi ve hizmet sektörlerinde</b> doğrudan uygulanmaktadır.
</p>
</div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""<div class="card-box">
<div class="card-title">⏳ Tarihsel Gelişim ve Karmaşıklık</div>
<ul>
<li><b>1950'ler:</b> Bilgisayar destekli ilk çalışan çizelgeleme araştırmalarının ortaya çıkışı.</li>
<li><b>1976:</b> NSP'nin modern matematiksel formülasyonuyla yöneylem araştırması literatürüne girdiği iki temel akademik yayının yapılması.</li>
<li><b>NP-Hard Karmaşıklığı:</b> Hemşire ve gün sayısı arttıkça olası kombinasyon sayısı katlanarak (eksponansiyel) artar. Bu durum problemin tam çözümlerinin (MILP) yanı sıra Sezgisel (Heuristic) ve Meta-sezgisel yöntemlerle (Genetik Algoritmalar, Tabu Arama) çözülmesini gerektirir.</li>
</ul>
</div>""", unsafe_allow_html=True)

    st.divider()

    st.markdown("### ⚖️ Kısıt Yapısı: Sert (Hard) vs. Yumuşak (Soft) Kısıtlar")
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #dc2626; background-color: #fff5f5;">
<div class="card-title" style="color: #991b1b;">
<span class="badge-hard">ZORUNLU</span> Sert Kısıtlar (Hard Constraints)
</div>
<p><b>Çözümün "Geçerli" (Feasible) sayılması için HİÇBİR ŞEKİLDE İHLAL EDİLEMEZ.</b></p>
<ul>
<li><b>Minimum Vardiya İhtiyacı:</b> Her vardiyada (örneğin gece nöbetinde) yasal ve operasyonel olarak bulunması gereken minimum uzman personel sayısı.</li>
<li><b>Çakışma Yasağı:</b> Bir çalışanın aynı gün ve saat diliminde birden fazla vardiyaya atanamaması.</li>
<li><b>Yasal Çalışma Süresi Sınırı:</b> Haftalık maksimum yasal çalışma saatinin (ör. 40-48 saat) aşılamaması.</li>
<li><b>Kesintisiz Dinlenme Süresi:</b> Gece vardiyasından çıkan personelin hemen ertesi sabah vardiyasına yazılamaması (Zorunlu 11 saat dinlenme kuralı).</li>
<li><b>Zorunlu Hafta Tatili (Kayan 7 Günlük Pencere / İSG Standardı):</b> İş Sağlığı ve Güvenliği (İSG) gereği <b>herhangi bir kayan 7 günlük periyotta en fazla 6 gün kesintisiz çalışma</b> yapılabilir. Takvim haftalarında yaşanabilecek 12 gün aralıksız çalışma ve iş kazası riskini önlemek için modelimiz 7 günlük kayan pencerede en az 1 gün zorunlu izin (OFF) kuralını sert kısıt olarak uygular.</li>
</ul>
</div>""", unsafe_allow_html=True)
        
    with col_c2:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #2563eb; background-color: #eff6ff;">
<div class="card-title" style="color: #1e40af;">
<span class="badge-soft">ESNEK</span> Yumuşak Kısıtlar (Soft Constraints)
</div>
<p><b>İhlali durumunda plan geçerlidir fakat "Çözüm Kalitesi ve Memnuniyet Skoru" düşer.</b></p>
<ul>
<li><b>Kişisel İzin Talepleri:</b> Çalışanların önceden sunduğu özel izin ve vardiya tercihlerine uyulması.</li>
<li><b>Adil Nöbet Dağılımı:</b> Gece ve hafta sonu nöbetlerinin tüm personel arasında eşit ve adil paylaştırılması.</li>
<li><b>Ardışık Çalışma Düzeni:</b> Nöbet günlerinin parça parça değil, toplu halde veya dengeli verilmesi.</li>
<li><b>Ekip Uyum Tercihleri:</b> Belirli kıdemli ve tecrübesiz personelin aynı vardiyada eşleştirilmesi.</li>
</ul>
</div>""", unsafe_allow_html=True)

    st.markdown("### 🧩 NP-Hard Karmaşıklık Neden Önemlidir?")
    st.info("""
    **Matematiksel Kombinasyon Patlaması:** 20 personelin 30 günlük bir ay boyunca 3 farklı vardiyaya atanması durumunda olası tüm çizelgelerin sayısı **4^600 ≈ 10^361** seviyesindedir. 
    Bu sayı evrendeki toplam atom sayısından kat kat büyüktür. Bu sebeple kaba kuvvet (Brute-Force) ile en iyi çözümü aramak imkansızdır. Yöneylem araştırmacıları bu problemi çözmek için **Tamsayılı Programlama (MILP)** ve **Meta-sezgisel Algoritmalar** kullanırlar.
    """)
