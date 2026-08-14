"""
================================================================================
  VIEWS/TAB2_INDUSTRIES.PY - SEKME 2: SANAYİ VE ENDÜSTRİYEL UYGULAMALAR
================================================================================
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def render_tab2():
    """Sekme 2 içeriğini çizer: Elektrik, Ulaşım ve Çelik Sanayisi Karşılaştırması."""
    st.markdown("## 🏭 NSP Modelinin Sanayi ve Endüstri Sektörlerindeki Kullanımı")
    st.write("""
    Hemşire çizelgeleme probleminde geliştirilen matematiksel kısıt modelleri, 
    7/24 esasına göre çalışan, yüksek iş güvenliği gerektiren ve vardiyalı personelin bulunduğu sanayi kollarında doğrudan uygulanmaktadır.
    """)

    ind_col1, ind_col2, ind_col3 = st.columns(3)

    with ind_col1:
        st.markdown("""<div class="card-box">
<div class="card-title">⚡ 1. Elektrik & Enerji Sektörü</div>
<p><b>Uygulama Alanı:</b> Nükleer, termik, hidroelektrik santralleri ve elektrik şebeke kontrol merkezleri.</p>
<hr style="border-color: #e2e8f0;">
<p><b style="color: #dc2626;">Sert Kısıtlar:</b></p>
<ul>
<li>Kontrol odasında 7/24 lisanslı nükleer/elektrik mühendisi bulundurma zorunluluğu.</li>
<li>Acil müdahale ve arıza-bakım ekiplerinin minimum sayıda hazırda tutulması.</li>
</ul>
<p><b style="color: #2563eb;">Yumuşak Kısıtlar:</b></p>
<ul>
<li>Tehlikeli santral birimlerindeki nöbetlerin personel arasında adil rotasyonu.</li>
<li>Hafta sonu vardiyalarının dengelenmesi.</li>
</ul>
</div>""", unsafe_allow_html=True)

    with ind_col2:
        st.markdown("""<div class="card-box">
<div class="card-title">✈️ 2. Ulaşım & Lojistik Sektörü</div>
<p><b>Uygulama Alanı:</b> Havayolları (Pilot/Kabin), Demiryolları (Makinist), Kara ve Deniz Lojistiği.</p>
<hr style="border-color: #e2e8f0;">
<p><b style="color: #dc2626;">Sert Kısıtlar:</b></p>
<ul>
<li><b>Yorgunluk Yönetimi (Fatigue Management):</b> Uluslararası havacılık ve ulaştırma yasaları gereği zorunlu uçuş/sürüş arası dinlenme saatleri.</li>
<li>Makinist ve pilotların belirli araç tipi ve rota lisanslarının bulunması.</li>
</ul>
<p><b style="color: #2563eb;">Yumuşak Kısıtlar:</b></p>
<ul>
<li>Ekip üyelerinin konaklamalı nöbetlerde tercih ettikleri şehirler.</li>
<li>Uçuş saatlerinin kıdem sırasına göre dağıtımı.</li>
</ul>
</div>""", unsafe_allow_html=True)

    with ind_col3:
        st.markdown("""<div class="card-box">
<div class="card-title">🏗️ 3. Çelik & Metal Sanayi</div>
<p><b>Uygulama Alanı:</b> Entegre çelik tesisleri, yüksek fırınlar, haddehaneler ve dökümhaneler.</p>
<hr style="border-color: #e2e8f0;">
<p><b style="color: #dc2626;">Sert Kısıtlar:</b></p>
<ul>
<li><b>Kesintisiz Üretim:</b> Yüksek fırınlar soğutulmaz; 3 vardiya (08-16, 16-24, 24-08) kesintisiz dolmalıdır.</li>
<li>Aşırı sıcak ve ağır iş sınıfı nedeniyle günlük maksimum çalışma süresi ve İSG sınırları.</li>
</ul>
<p><b style="color: #2563eb;">Yumuşak Kısıtlar:</b></p>
<ul>
<li>Ağır iş temposu sonrası üst üste gece vardiyası yazılmaması.</li>
<li>Vardiya takımlarının sabit tutularak ekip uyumunun artırılması.</li>
</ul>
</div>""", unsafe_allow_html=True)

    st.divider()

    st.markdown("### 📊 Sektörler Arası Kısıt Karşılaştırma Matrisi")

    df_sector = pd.DataFrame({
        "Sektör": ["Sağlık (NSP)", "Elektrik & Enerji", "Ulaşım & Lojistik", "Çelik & Metal Sanayi"],
        "7/24 Kesintisiz Zorunluluk": ["Yüksek (Acil/Servis)", "Kritik (Santral/Şebeke)", "Çok Yüksek (Seferler)", "Kritik (Yüksek Fırın)"],
        "Temel Sert Kısıt": ["Uzman Hemşire Oranı / Nöbet Saati", "Lisanslı Operatör / Nöbetçi Mühendis", "Zorunlu Dinlenme Saati (FAA/İSG)", "Max Çalışma Süresi & İSG Standartları"],
        "Temel Yumuşak Kısıt": ["İzin Tercihi / Adil Gece Nöbeti", "Santral Birim Rotasyonu", "Yatı Nöbeti & Rota Tercihi", "Vardiya Takım Uyum Tercihi"],
        "İhlal Sonucu": ["Hizmet Aksaması / Tıbbi Hata", "Elektrik Kesintisi / Santral Arızası", "Kaza Riski / Yasal Cezalar", "İş Kazası / Yüksek Fırın Hasarı"]
    })

    st.dataframe(df_sector, width="stretch", hide_index=True)

    st.markdown("### 📈 Sektörlere Göre Vardiya Hassasiyet Analizi")
    
    fig_radar = go.Figure()
    categories = ['Yasal Kısıt Sertliği', 'Yorgunluk & İSG Riski', '7/24 Kesintisiz Çalışma', 'Esnek İzin Yönetimi', 'Kıdem/Yetkinlik Eşleme']

    fig_radar.add_trace(go.Scatterpolar(
        r=[5, 4, 5, 4, 4],
        theta=categories,
        fill='toself',
        name='Sağlık (NSP)',
        line_color='#dc2626'
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=[4, 4, 5, 3, 5],
        theta=categories,
        fill='toself',
        name='Elektrik & Enerji',
        line_color='#d97706'
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=[5, 5, 4, 3, 4],
        theta=categories,
        fill='toself',
        name='Ulaşım & Lojistik',
        line_color='#2563eb'
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=[4, 5, 5, 2, 3],
        theta=categories,
        fill='toself',
        name='Çelik & Metal Sanayi',
        line_color='#059669'
    ))

    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 5], tickfont=dict(color="#475569")),
            bgcolor="#ffffff"
        ),
        paper_bgcolor="#f8fafc",
        font=dict(color="#0f172a", size=13),
        showlegend=True,
        height=450
    )

    st.plotly_chart(fig_radar, width="stretch")
