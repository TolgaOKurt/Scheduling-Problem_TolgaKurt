"""
================================================================================
  VIEWS/TAB2_INDUSTRIES.PY - SEKME 2: SANAYİ VE ENDÜSTRİYEL UYGULAMALAR
================================================================================
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def render_tab2():
    """Sekme 2 içeriğini çizer: Elektrik, Ulaşım, Çelik Sanayisi Karşılaştırması ve Radar Analizi."""
    st.markdown("## 🏭 NSP Modelinin Sanayi ve Endüstri Sektörlerindeki Kullanımı")
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

    st.divider()

    # 2. KARŞILAŞTIRMA MATRİSİ
    st.markdown("### 📊 Sektörler Arası Kısıt ve Dinamik Karşılaştırma Matrisi")
    st.markdown("""
    Aşağıdaki matris, sağlık sektörü ile 7/24 vardiyalı ağır sanayi kolları arasındaki operasyonel kısıtları, 
    yasal dayanakları ve olası ihlal sonuçlarını özetlemektedir:
    """)

    df_sector = pd.DataFrame({
        "Sektör": [
            "Sağlık (NSP / NRP)",
            "Elektrik & Enerji",
            "Ulaşım & Havacılık",
            "Çelik & Ağır Metal"
        ],
        "Tipik Vardiya Düzeni": [
            "3x8 veya 2x12 (Klinik / Acil)",
            "3x8 (3 Vardiya Kesintisiz)",
            "Dinamik Uçuş Blokları (FTL)",
            "3x8 (4 Posta Sürekli Sistem)"
        ],
        "7/24 Kesintisiz Zorunluluk": [
            "Yüksek (Acil / Yoğun Bakım)",
            "Kritik (Santral / Şebeke)",
            "Çok Yüksek (Sefer & Hatlar)",
            "Hayati (Yüksek Fırın Duruş Yasağı)"
        ],
        "Temel Sert Kısıt (Hard)": [
            "Uzman Hemşire Oranı / 11 Saat Dinlenme",
            "Lisanslı Operatör / Nöbetçi Başmühendis",
            "Zorunlu Dinlenme Saati (FAA/SHGM/İSG)",
            "Max 7.5s Gece / MYK Kritik Ehliyetler"
        ],
        "Temel Yumuşak Kısıt (Soft)": [
            "Kişisel İzin Talebi / Adil Gece Nöbeti",
            "Santral Ünite Rotasyonu / Bayram Dengesi",
            "Yatı Nöbeti & Rota Tercihi / Kıdem",
            "Posta Bütünlüğü / Usta-Çırak Dengesi"
        ],
        "Kritik Risk & İhlal Sonucu": [
            "Hizmet Aksaması / Tıbbi Hata Riski",
            "Şebeke Frekans Çökmesi / Blackout",
            "Uçuş Emniyeti Riski / Ağır Yasal Ceza",
            "Yüksek Fırın Donması / Ağır İş Kazası"
        ],
        "Yasal & Regülatif Dayanak": [
            "Sağlık Bak. Yönetmelikleri & İş K.",
            "EPDK & Nükleer Düzenleme Kurumu",
            "SHGM / ICAO / FAA FTL Kuralları",
            "4857 SK Md. 69, 6331 SK İSG & TİS"
        ]
    })

    st.dataframe(df_sector, width="stretch", hide_index=True)

    st.divider()

    # 3. RADAR GRAFİĞİ VE ANALİTİK YORUM
    st.markdown("### 📈 Sektörlere Göre Vardiya ve Kısıt Hassasiyet Analizi")

    r_col1, r_col2 = st.columns([3, 2])

    with r_col1:
        fig_radar = go.Figure()
        categories = [
            'Yasal Kısıt Sertliği',
            'Yorgunluk & İSG Riski',
            '7/24 Kesintisiz Çalışma',
            'Esnek İzin Yönetimi',
            'Kıdem / Yetkinlik Eşleme'
        ]

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
            font=dict(color="#0f172a", size=12),
            showlegend=True,
            legend=dict(orientation="h", y=1.2, x=0.1),
            height=430,
            margin=dict(l=40, r=40, t=40, b=30)
        )

        st.plotly_chart(fig_radar, width="stretch", key="t2_fig_radar")

    with r_col2:
        st.markdown("""
        <div style="background-color: #ffffff; border: 1px solid #cbd5e1; border-radius: 10px; padding: 1.3rem; box-shadow: 0 4px 12px rgba(0,0,0,0.04); height: 100%;">
            <h4 style="color: #1e40af; margin-top: 0; margin-bottom: 0.8rem; font-size: 1.1rem;">💡 Radar Analizi Yorumu ve Çıkarımlar</h4>
            <div style="font-size: 0.9rem; color: #334155; line-height: 1.65;">
                <p>Radar grafiğinde yer alan 1-5 puanlık değerlendirme skalası, her sektörün optimizasyon modelindeki ağırlık önceliklerini yansıtır:</p>
                <ul style="padding-left: 1.1rem; margin-bottom: 0.6rem;">
                    <li><b>Yorgunluk & İSG Riski:</b> Ağır metal ve havacılık sektörlerinde zirvededir (5/5). Tek bir yorgunluk hatası telafisi imkansız can ve mal kayıplarına yol açabilir.</li>
                    <li><b>7/24 Kesintisiz Süreç:</b> Çelik sanayisinde yüksek fırınların sönme riski nedeniyle kesinti toleransı 0'dır (5/5).</li>
                    <li><b>Kıdem ve Yetkinlik:</b> Enerji kontrol merkezlerinde ve yoğun bakım birimlerinde sertifikalı uzman gereksinimi en yüksek seviyededir.</li>
                    <li><b>Esnek İzin Yönetimi:</b> Sağlıkta kişisel izin talepleri nispeten yüksek esneklik gösterirken, ağır sanayide posta bütünlüğü nedeniyle izinler daha katı rotasyonla yönetilir.</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
