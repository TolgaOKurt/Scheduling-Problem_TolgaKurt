"""
================================================================================
  VIEWS/TAB16_MULTI_OBJECTIVE.PY - SEKME 16: ÇOK AMAÇLI OPTİMİZASYON (MOO) & PARETO ANALİZİ
================================================================================
  Bu modül; Yöneylem Araştırması (OR) ve Çizelgeleme literatüründe tek amaçlı skaler
  yaklaşımların ötesine geçen Çok Amaçlı Optimizasyon (Multi-Objective Optimization - MOO),
  Pareto Hakimiyeti (Pareto Dominance), NSGA-II mimarisi ve çatışan amaçlar arasındaki
  ödünleşim (Trade-off) teorisini interaktif ve matematiksel olarak sunar.
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from algorithms.worker_manager import generate_worker_profiles
from algorithms.penalty_calculator import calculate_full_penalties


def render_tab16(params):
    """Sekme 16 içeriğini çizer: Çok Amaçlı Optimizasyon (MOO) ve Pareto Analizi Rehberi."""
    st.markdown("## 🎯 Sekme 16: Çok Amaçlı Optimizasyon (Multi-Objective Optimization) & Pareto Analizi")
    
    st.markdown("""
    <div style="background-color: #eff6ff; border-left: 5px solid #2563eb; border-radius: 8px; padding: 16px 20px; margin-bottom: 22px; color: #1e40af; line-height: 1.65;">
        <h4 style="margin-top: 0; color: #1d4ed8; font-size: 1.15rem;">🌐 Tek Amaçtan Çok Amaca: Sanayide Gerçekçi Karar Verme Paradigması</h4>
        Klasik çizelgeleme modelleri tüm ceza puanlarını tek bir skaler amaç fonksiyonunda toplar (<i>min Z = &sum; w<sub>i</sub> &times; Ceza<sub>i</sub></i>). 
        Ancak gerçek endüstriyel dünyada amaçlar <b>birbiriyle doğrudan çatışır</b>: İşletme maliyetini düşürmek personelin dinlenme hakkını zorlaştırırken, 
        çalışanların tüm kişisel izin taleplerini karşılamak takım bütünlüğünü ve usta dengesini bozar. 
        <b>Çok Amaçlı Optimizasyon (Multi-Objective Optimization - MOO)</b>, tek bir "yapay" en iyi çözüm yerine karar vericiye 
        <b>Pareto Optimal Seçenekler Yelpazesi (Pareto Frontier)</b> sunan ileri düzey yöneylem disiplinidir.
    </div>
    """, unsafe_allow_html=True)

    # ==============================================================================
    # 1. BÖLÜM: NEDEN ÇOK AMAÇLI OPTİMİZASYON? (TEK AMAÇLI AĞIRLIKLI TOPLAMIN ÇIKMAZLARI)
    # ==============================================================================
    st.markdown("### 🧩 1. Neden Tercih Edilir? Tek Amaçlı (Ağırlıklı Toplam) Yaklaşımın Çıkmazları")
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="card-box" style="border-top: 5px solid #ef4444; height: 100%;">
            <div class="card-title" style="color: #b91c1c;">⚠️ Tek Amaçlı Skalerleştirmenin (Weighted Sum) Kısıtları</div>
            <ul style="font-size: 0.92rem; color: #334155; line-height: 1.65;">
                <li><b>Ağırlık Seçiminin Keyfiliği (Arbitrary Weights):</b> Ceza katsayıları (<i>w<sub>posta</sub>=15, w<sub>izin</sub>=40 vb.</i>) genellikle sübjektif seçilir. Ağırlıktaki %10'luk bir değişim tamamen farklı bir çizelge üretebilir.</li>
                <li><b>İçbükey Olmayan (Non-Convex) Bölgeleri Kaçırma:</b> Matematiksel olarak ağırlıklı toplam yöntemi, Pareto yüzeyinin içbükey (non-convex) kısımlarındaki mükemmel çözümleri <i>asla bulamaz</i> (Duality Gap problemi).</li>
                <li><b>Baskın Amaç Sorunu:</b> Katsayısı yüksek olan tek bir kural (örn: Kıdemli Usta cezası), diğer tüm insani tercihleri ve adalet kriterlerini gölgede bırakabilir.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card-box" style="border-top: 5px solid #10b981; height: 100%;">
            <div class="card-title" style="color: #047857;">🏆 Çok Amaçlı Optimizasyonun (MOO) Getirdiği Üstünlükler</div>
            <ul style="font-size: 0.92rem; color: #334155; line-height: 1.65;">
                <li><b>Kör Ağırlık Belirleme Zorunluluğu Yoktur:</b> Algoritma çalışmadan önce katsayı tahmin etmeye gerek kalmaz; çözücü amaçlar arasındaki doğal dengeyi kendi keşfeder.</li>
                <li><b>Karar Destek Sistemi (DSS) Gücü:</b> Yöneticiye tek bir çizelgeyi dikte etmek yerine, <i>"Hangi amaçtan ne kadar taviz verilirse ne kazanılır?"</i> analizi (Trade-off Menüsü) sunar.</li>
                <li><b>Uyuşmazlık Yönetimi (Şirket vs. Çalışan):</b> Fabrika yönetimi ile işçi sendikası arasındaki müzakereleri şeffaf, veriye dayalı ve matematiksel bir uzlaşma zeminine oturtur.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ==============================================================================
    # 2. BÖLÜM: ÇELİK FABRİKASINDA 2 ÇATIŞAN ANA EKSEN (MATEMATİKSEL FORMÜLASYON)
    # ==============================================================================
    st.markdown("### 🧮 2. Çelik Tesisinde Çatışan 2 Temel Amaç Fonksiyonunun Matematiksel Ayrımı")
    st.markdown("Vardiya problemimizde 6 yumuşak ceza unsuru, doğası gereği birbiriyle rekabet eden **2 ana vektörel amaca** ayrılır:")

    m_col1, m_col2 = st.columns(2)

    with m_col1:
        st.markdown("""
        <div style="background-color: #f8fafc; border: 2px solid #3b82f6; border-radius: 10px; padding: 18px; height: 100%;">
            <h4 style="color: #1d4ed8; margin-top: 0;">🏭 1. Amaç: İşletme & Üretim Verimliliği (f₁)</h4>
            <p style="font-size: 0.92rem; color: #1e293b; line-height: 1.6;">
                Fabrika yönetiminin önceliği: Üretim kesintisiz aksın, ekipler bölünmesin, usta eksikliği yüzünden döküm durmasın.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"""
        f_1(X) = w_{\text{posta}} \cdot \text{Posta\_Dev}(X) + w_{\text{usta}} \cdot \text{No\_Usta}(X) + \text{FourPosta\_Pen}(X)
        """)
        st.caption("📌 **Kapsam:** Posta takım bütünlüğü, aktif vardiyalarda Kıdemli Usta varlığı ve 4-posta günlük blok dinlenme kuralı.")

    with m_col2:
        st.markdown("""
        <div style="background-color: #fdf2f8; border: 2px solid #ec4899; border-radius: 10px; padding: 18px; height: 100%;">
            <h4 style="color: #be185d; margin-top: 0;">👨‍🏭 2. Amaç: Çalışan Memnuniyeti, Adalet & Ergonomi (f₂)</h4>
            <p style="font-size: 0.92rem; color: #1e293b; line-height: 1.6;">
                Çalışanların önceliği: İzin taleplerim karşılansın, gece nöbetleri adil dağılsın, sirkadiyen ritmim bozulmasın.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.latex(r"""
        f_2(X) = w_{\text{izin}} \cdot \text{Pref\_Viol}(X) + w_{\text{gece}} \cdot \text{Night\_Imb}(X) + w_{\text{work}} \cdot \text{Work\_Imb}(X) + w_{\text{sirk}} \cdot \text{Circadian}(X)
        """)
        st.caption("📌 **Kapsam:** Kişisel izin talepleri, gece nöbeti adaleti (L1), toplam iş yükü dengesi (L1) ve sirkadiyen dönüşler.")

    st.markdown("""
    <div style="background-color: #f8fafc; border: 1.5px solid #cbd5e1; border-radius: 8px; padding: 14px; margin-top: 15px; text-align: center;">
        <span style="font-size: 1.1rem; font-weight: 700; color: #0f172a;">
            Vektörel Amaç Modeli: &nbsp;&nbsp; 
            <span style="color: #2563eb;">min <b>F</b>(<i>X</i>) = [ <i>f</i><sub>1</sub>(<i>X</i>), <i>f</i><sub>2</sub>(<i>X</i>) ]<sup>T</sup></span>
            &nbsp;&nbsp; s.t. &nbsp;&nbsp; <i>X</i> &isin; &Omega; (Tüm Sert Kısıtlar %100 Sağlanmalıdır)
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ==============================================================================
    # 3. BÖLÜM: PARETO HAKİMİYETİ (DOMINANCE) VE ÖN CEPHE (FRONTIER) TEORİSİ
    # ==============================================================================
    st.markdown("### 📐 3. Pareto Hakimiyeti (Pareto Dominance) ve Ön Cephe Teorisi")
    
    p_col1, p_col2 = st.columns([1.4, 1.6])

    with p_col1:
        st.markdown("""
        <div class="card-box" style="border-left: 4px solid #8b5cf6;">
            <div class="card-title" style="color: #6d28d9;">⚖️ Pareto Üstünlüğü (Dominance) Nedir?</div>
            <p style="font-size: 0.92rem; color: #334155; line-height: 1.65;">
                İki geçerli çizelge çözümü <b>A</b> ve <b>B</b> olsun. 
                A çözümü B çözümüne <b>Pareto Baskındır (A &prec; B)</b> ancak ve ancak şu iki şart aynı anda sağlanırsa:
            </p>
            <ol style="font-size: 0.9rem; color: #1e293b; line-height: 1.65; padding-left: 1.2rem;">
                <li><b>Hiçbir amaçta daha kötü olmamalıdır:</b><br>
                    <i>f</i><sub>1</sub>(A) &le; <i>f</i><sub>1</sub>(B) &nbsp; ve &nbsp; <i>f</i><sub>2</sub>(A) &le; <i>f</i><sub>2</sub>(B)</li>
                <li><b>En az bir amaçta kesinlikle daha iyi olmalıdır:</b><br>
                    <i>f</i><sub>1</sub>(A) &lt; <i>f</i><sub>1</sub>(B) &nbsp; veya &nbsp; <i>f</i><sub>2</sub>(A) &lt; <i>f</i><sub>2</sub>(B)</li>
            </ol>
            <p style="font-size: 0.88rem; color: #6b7280; margin-top: 8px;">
                💡 <b>Pareto Optimal Çözüm:</b> Çözüm uzayında kendisine baskın gelen başka <i>hiçbir çözümün bulunmadığı</i> çizelgelerdir.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with p_col2:
        st.markdown("""
        <div class="card-box" style="border-left: 4px solid #059669;">
            <div class="card-title" style="color: #047857;">📈 Pareto Ön Cephesi (Pareto Frontier) & Diz Noktası (Knee Point)</div>
            <p style="font-size: 0.92rem; color: #334155; line-height: 1.65;">
                Tüm Pareto optimal çözümlerin 2 boyutlu amaç uzayında oluşturduğu sınıra <b>Pareto Ön Cephesi (Pareto Frontier)</b> denir.
            </p>
            <ul style="font-size: 0.9rem; color: #1e293b; line-height: 1.65; padding-left: 1.2rem;">
                <li><b>Ödünleşim (Trade-off):</b> Bu cephe üzerinde bir amaç ancak diğer amaç kötüleştirilerek iyileştirilebilir.</li>
                <li><b>Uç Nokta 1 (İşletme Aşırı):</b> Posta bütünlüğü kusursuzdur (0 ceza) ancak çalışan izinlerinin çoğu reddedilmiştir.</li>
                <li><b>Uç Nokta 2 (Çalışan Aşırı):</b> Herkesin izni verilmiştir ancak postalar paramparça bölünmüştür.</li>
                <li><b>Diz Noktası (Knee Point):</b> Eğrinin büküldüğü, her iki taraftan da minimum feragatle maksimum faydanın elde edildiği <b>Altın Denge Noktasıdır</b>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ==============================================================================
    # 4. BÖLÜM: ÇOK AMAÇLI OPTİMİZASYON ÇÖZÜM METODOLOJİLERİ (NSGA-II & ε-CONSTRAINT)
    # ==============================================================================
    st.markdown("### 🔬 4. Çok Amaçlı Problemleri Çözen Algoritmik Mimariler")

    a_col1, a_col2, a_col3 = st.columns(3)

    with a_col1:
        st.markdown("""
        <div class="card-box" style="border-top: 4px solid #6366f1; height: 100%;">
            <div class="card-title" style="color: #4338ca;">🧬 1. NSGA-II (Deb et al., 2002)</div>
            <p style="font-size: 0.88rem; color: #334155; line-height: 1.6;">
                Çok amaçlı optimizasyonun dünyadaki altın standardı genetik algoritmasıdır.
            </p>
            <ul style="font-size: 0.85rem; color: #475569; line-height: 1.55; padding-left: 1.1rem;">
                <li><b>Hızlı Baskınlık Sıralaması (Fast Non-dominated Sort):</b> Popülasyonu katman katman (Front 1, Front 2...) Pareto rütbelerine ayırır.</li>
                <li><b>Kalabalıklaşma Mesafesi (Crowding Distance):</b> Çözümlerin Pareto cephesine homojen yayılmasını sağlar; çeşitliliği korur.</li>
                <li><b>Elitizm:</b> En iyi ön cephe bireylerini sonraki nesle doğrudan aktarır.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with a_col2:
        st.markdown("""
        <div class="card-box" style="border-top: 4px solid #0ea5e9; height: 100%;">
            <div class="card-title" style="color: #0369a1;">🎯 2. ε-Kısıt Yöntemi (ε-Constraint)</div>
            <p style="font-size: 0.88rem; color: #334155; line-height: 1.6;">
                Matematiksel çözücülerle (MILP / CP-SAT) kesin Pareto noktaları üretme tekniğidir.
            </p>
            <ul style="font-size: 0.85rem; color: #475569; line-height: 1.55; padding-left: 1.1rem;">
                <li><b>Tek Amacı Optimize Et:</b> <i>min f<sub>1</sub>(X)</i></li>
                <li><b>Diğer Amacı Kısıta Çevir:</b> <i>f<sub>2</sub>(X) &le; &epsilon;<sub>k</sub></i></li>
                <li><b>Grid Tarama:</b> &epsilon; değeri adım adım kaydırılarak MILP defalarca çözülür; içbükey dahil tüm kesin Pareto noktaları yakalanır.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with a_col3:
        st.markdown("""
        <div class="card-box" style="border-top: 4px solid #f59e0b; height: 100%;">
            <div class="card-title" style="color: #b45309;">🧩 3. MOEA/D (Ayrıştırma Tabanlı)</div>
            <p style="font-size: 0.88rem; color: #334155; line-height: 1.6;">
                Çok amaçlı problemi eşzamanlı çözülen <i>N</i> adet tek amaçlı alt probleme böler.
            </p>
            <ul style="font-size: 0.85rem; color: #475569; line-height: 1.55; padding-left: 1.1rem;">
                <li><b>Ağırlık Vektörleri:</b> Çözüm uzayını homojen bölen yön vektörleri tanımlanır.</li>
                <li><b>Komşuluk Bilgi Paylaşımı:</b> Her alt problem komşu alt problemlerin bulduğu bilgileri kullanarak Pareto cephesini çok hızlı örer.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ==============================================================================
    # 5. BÖLÜM: 3 TEMEL YÖNETİMSEL SENARYO KARŞILAŞTIRMASI
    # ==============================================================================
    st.markdown("### 👔 5. Pareto Cephesinden 3 Yönetimsel Strateji Seçeneği")

    s_col1, s_col2, s_col3 = st.columns(3)

    with s_col1:
        st.markdown("""
        <div style="background-color: #eff6ff; border: 1.5px solid #3b82f6; border-radius: 8px; padding: 14px; height: 100%;">
            <div style="font-weight: 700; color: #1d4ed8; font-size: 1rem; margin-bottom: 6px;">🏭 1. Strateji: İşletme Öncelikli (Managerial)</div>
            <p style="font-size: 0.85rem; color: #1e40af; line-height: 1.55;">
                • <b>Öncelik:</b> Posta bütünlüğü %100 korunur, her vardiyada usta kesinlikle bulunur.<br>
                • <b>Bedel:</b> Çalışanların izin taleplerinin %40'ı karşılanamaz, gece nöbeti adaleti esner.<br>
                • <b>Kullanım:</b> Yüksek üretim kotalı acil sipariş dönemleri.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with s_col2:
        st.markdown("""
        <div style="background-color: #fefce8; border: 1.5px solid #eab308; border-radius: 8px; padding: 14px; height: 100%;">
            <div style="font-weight: 700; color: #a16207; font-size: 1rem; margin-bottom: 6px;">⭐ 2. Strateji: Diz Noktası (Knee Point - Dengeli)</div>
            <p style="font-size: 0.85rem; color: #713f12; line-height: 1.55;">
                • <b>Öncelik:</b> Her iki taraftan da minimum tavizle maksimum doyum.<br>
                • <b>Kazanım:</b> Postalar %92 oranında bir arada kalırken, izinlerin %88'i karşılanır.<br>
                • <b>Kullanım:</b> Standart fabrika işletme dönemi için ideal altın oran.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with s_col3:
        st.markdown("""
        <div style="background-color: #fdf2f8; border: 1.5px solid #ec4899; border-radius: 8px; padding: 14px; height: 100%;">
            <div style="font-weight: 700; color: #be185d; font-size: 1rem; margin-bottom: 6px;">👨‍👩‍👧‍👦 3. Strateji: Çalışan Öncelikli (Union / Human-Centric)</div>
            <p style="font-size: 0.85rem; color: #9d174d; line-height: 1.55;">
                • <b>Öncelik:</b> Tüm izin talepleri karşılanır, gece nöbetleri kusursuz eşit dağılır.<br>
                • <b>Bedel:</b> Posta takımları bölünür, farklı posta elemanları aynı vardiyada kaynaşır.<br>
                • <b>Kullanım:</b> Bayram, tatil ve düşük üretim bakım periyotları.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ==============================================================================
    # 6. BÖLÜM: BİLİMSEL ÖZET MATRİSİ (TEK AMAÇLI vs. ÇOK AMAÇLI)
    # ==============================================================================
    st.markdown("### 📋 6. Karşılaştırma Matrisi: Tek Amaçlı (Single-Objective) vs. Çok Amaçlı (Multi-Objective)")
    
    comp_df = pd.DataFrame({
        "Karşılaştırma Boyutu": [
            "Amaç Fonksiyonu Yapısı",
            "Ağırlık İhtiyacı (Weights)",
            "Üretilen Çözüm Sayısı",
            "İçbükey Olmayan (Non-convex) Bölgeler",
            "Karar Vericinin Rolü",
            "Hesaplama Karmaşıklığı",
            "En Uygun Olduğu Alan"
        ],
        "Tek Amaçlı Optimizasyon (Weighted Sum)": [
            "Skaler: min Z = Σ wi · fi(x)",
            "Çözüm öncesi keyfi ağırlık belirlenmelidir",
            "Yalnızca 1 tek çözüm noktası",
            "❌ Duality gap nedeniyle yakalayamaz",
            "Pasif (Solver ne bulduysa kabul edilir)",
            "Daha Hızlı (Tek arama ekseni)",
            "Önceliklerin net ve tartışmasız olduğu durumlar"
        ],
        "Çok Amaçlı Optimizasyon (MOO / Pareto)": [
            "Vektörel: min F(x) = [f1(x), f2(x)]ᵀ",
            "Ağırlık gerekmez, amaçlar bağımsız yarışır",
            "Birbirine üstünlüğü olmayan Çözümler Kümesi (Frontier)",
            "✅ Tüm içbükey ve ayrık Pareto bölgelerini yakalar",
            "Aktif (Menüden işletme politikasına uygun çizelgeyi seçer)",
            "Daha Kapsamlı (Pareto cephesini örmek gerekir)",
            "Şirket & Sendika / Maliyet & Ergonomi uyuşmazlıkları"
        ]
    })

    st.dataframe(comp_df, hide_index=True, use_container_width=True)
