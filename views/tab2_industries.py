"""
================================================================================
  VIEWS/TAB2_INDUSTRIES.PY - SEKME 2: SANAYİ VE ENDÜSTRİYEL UYGULAMALAR
================================================================================
"""
import streamlit as st

def render_tab2():
    """Sekme 2 içeriğini çizer: Elektrik, Ulaşım ve Çelik Sanayisi Uygulamaları."""
    st.markdown("## 🏭 Sekme 2: NSP Modelinin Sanayi ve Endüstri Sektörlerindeki Kullanımı")
    st.markdown("""
    Hemşire Çizelgeleme Problemi (NSP) için geliştirilen matematiksel kısıt optimizasyonu ve yöneylem araştırması modelleri, 
    yalnızca sağlık sektöründe sınırlı kalmayıp **7/24 kesintisiz operasyon gerektiren, yüksek iş güvenliği riski barındıran 
    ve vardiyalı personel istihdam eden tüm kritik sanayi kollarında** doğrudan uygulanmaktadır.
    """)

    # 1. SEKTÖR DETAY KARTLARI (3 KOLON)
    ind_col1, ind_col2, ind_col3 = st.columns(3)

    with ind_col1:
        st.markdown("""
        <div class="card-box" style="height: 100%;">
            <div class="card-title">⚡ 1. Elektrik & Enerji Sektörü</div>
            <p style="font-size: 0.92rem; color: #334155; margin-bottom: 0.8rem;">
                <b>Uygulama Alanı:</b> Nükleer, termik ve hidroelektrik santralleri, TEİAŞ Yük Tevzi ve SCADA kontrol merkezleri.
            </p>
            <hr style="border-color: #e2e8f0; margin: 0.6rem 0;">
            <p style="color: #dc2626; font-weight: bold; font-size: 0.92rem; margin-bottom: 0.3rem;">🚨 Sert Kısıtlar (Hard):</p>
            <ul style="font-size: 0.88rem; color: #1e293b; padding-left: 1.1rem; margin-bottom: 0.8rem;">
                <li>Kontrol odasında 7/24 lisanslı nükleer/elektrik başmühendisi bulundurma zorunluluğu.</li>
                <li>Acil müdahale ve yüksek gerilim arıza-bakım ekiplerinin hazır beklemesi.</li>
                <li>Maksimum kesintisiz konsantrasyon süresi sınırlamaları.</li>
            </ul>
            <p style="color: #2563eb; font-weight: bold; font-size: 0.92rem; margin-bottom: 0.3rem;">🎯 Yumuşak Kısıtlar (Soft):</p>
            <ul style="font-size: 0.88rem; color: #1e293b; padding-left: 1.1rem; margin-bottom: 0.8rem;">
                <li>Yüksek riskli santral ünitelerindeki nöbetlerin personel arasında adil rotasyonu.</li>
                <li>Hafta sonu ve bayram vardiyalarının dengelenmesi.</li>
            </ul>
            <div style="background-color: #fef2f2; border-left: 3px solid #ef4444; padding: 6px 10px; border-radius: 4px; font-size: 0.82rem; color: #991b1b;">
                <b>Kritik Risk:</b> Frekans çökmesi, iletim hattı arızaları ve geniş çaplı elektrik kesintisi (Blackout).
            </div>
        </div>
        """, unsafe_allow_html=True)

    with ind_col2:
        st.markdown("""
        <div class="card-box" style="height: 100%;">
            <div class="card-title">✈️ 2. Ulaşım, Havacılık & Lojistik</div>
            <p style="font-size: 0.92rem; color: #334155; margin-bottom: 0.8rem;">
                <b>Uygulama Alanı:</b> Havayolları (Kokpit/Kabin), Demiryolları (Makinist/Dispeçer), Denizcilik & Limanlar.
            </p>
            <hr style="border-color: #e2e8f0; margin: 0.6rem 0;">
            <p style="color: #dc2626; font-weight: bold; font-size: 0.92rem; margin-bottom: 0.3rem;">🚨 Sert Kısıtlar (Hard):</p>
            <ul style="font-size: 0.88rem; color: #1e293b; padding-left: 1.1rem; margin-bottom: 0.8rem;">
                <li><b>Yorgunluk Yönetimi (FTL):</b> SHGM, FAA ve EASA regülasyonları gereği zorunlu uçuş ve blok dinlenme süreleri.</li>
                <li>Uçak tipi sertifikasyonu (Type Rating) ve rota/meydan yetkinlik eşleşmesi.</li>
                <li>Maksimum ardışık uçuş görev süresi sınırları.</li>
            </ul>
            <p style="color: #2563eb; font-weight: bold; font-size: 0.92rem; margin-bottom: 0.3rem;">🎯 Yumuşak Kısıtlar (Soft):</p>
            <ul style="font-size: 0.88rem; color: #1e293b; padding-left: 1.1rem; margin-bottom: 0.8rem;">
                <li>Konaklamalı (yatı) seferlerde tercih edilen şehir ve otel eşleşmeleri.</li>
                <li>Görev intikali (Deadheading) sürelerinin ve uçuş saatlerinin kıdeme göre dağıtımı.</li>
            </ul>
            <div style="background-color: #fef2f2; border-left: 3px solid #ef4444; padding: 6px 10px; border-radius: 4px; font-size: 0.82rem; color: #991b1b;">
                <b>Kritik Risk:</b> Yorgunluk kaynaklı kaza (Fatigue Risk), sefer iptalleri ve zincirleme gecikmeler.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with ind_col3:
        st.markdown("""
        <div class="card-box" style="height: 100%;">
            <div class="card-title">🏗️ 3. Çelik & Ağır Metal Sanayi</div>
            <p style="font-size: 0.92rem; color: #334155; margin-bottom: 0.8rem;">
                <b>Uygulama Alanı:</b> Entegre çelik tesisleri, yüksek fırınlar, sürekli döküm hatları ve haddehaneler.
            </p>
            <hr style="border-color: #e2e8f0; margin: 0.6rem 0;">
            <p style="color: #dc2626; font-weight: bold; font-size: 0.92rem; margin-bottom: 0.3rem;">🚨 Sert Kısıtlar (Hard):</p>
            <ul style="font-size: 0.88rem; color: #1e293b; padding-left: 1.1rem; margin-bottom: 0.8rem;">
                <li><b>Kesintisiz 7/24 Proses:</b> Yüksek fırınlar soğutulamaz; 3 vardiya (08-16, 16-24, 24-08) eksiksiz dolmalıdır.</li>
                <li>MYK sertifikalı tavan vinci operatörü, potacı ve döküm uzmanı gereksinimi.</li>
                <li>4857 Sayılı İş Kanunu Md. 69 (Gece en fazla 7.5 saat fiili çalışma).</li>
            </ul>
            <p style="color: #2563eb; font-weight: bold; font-size: 0.92rem; margin-bottom: 0.3rem;">🎯 Yumuşak Kısıtlar (Soft):</p>
            <ul style="font-size: 0.88rem; color: #1e293b; padding-left: 1.1rem; margin-bottom: 0.8rem;">
                <li>4 Posta sistemi ile takım bütünlüğünün korunması.</li>
                <li>Usta-çırak tecrübe dengesi ve sirkadiyen saat yönü vardiya dönüşü.</li>
            </ul>
            <div style="background-color: #fef2f2; border-left: 3px solid #ef4444; padding: 6px 10px; border-radius: 4px; font-size: 0.82rem; color: #991b1b;">
                <b>Kritik Risk:</b> Yüksek fırın donması (milyonlarca dolarlık refrakter hasarı) ve sıvı maden kazaları.
            </div>
        </div>
        """, unsafe_allow_html=True)
