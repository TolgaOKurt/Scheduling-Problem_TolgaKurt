"""
================================================================================
  VIEWS/TAB3_STEEL_MODEL.PY - SEKME 3: ÇELİK SANAYİ KISITLARI VE MILP MODELİ
================================================================================
"""
import streamlit as st

def render_tab3():
    """Sekme 3 içeriğini çizer: 5 Sert, 5 Yumuşak kısıt ve MILP matematiksel formülasyonu."""
    st.markdown("## 🏗️ Çelik ve Metal Sanayisi Çizelgeleme Kısıtları & Matematiksel Modeli")
    st.write("""
    Çelik sanayisinde yüksek fırınlar ve döküm hatları 7/24 kesintisiz çalışır. 
    Aşağıda **4857 Sayılı İş Kanunu**, **6331 Sayılı İSG Kanunu** ve Türkiye çelik tesisleri **Toplu İş Sözleşmeleri (TİS)** esas alınarak hazırlanan 5 Sert ve 5 Yumuşak kısıt listelenmiştir:
    """)

    col_steel_hard, col_steel_soft = st.columns(2)

    # --- Sol Sütun: Sert Kısıtlar (Hard Constraints) ---
    with col_steel_hard:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #dc2626; background-color: #fff5f5;">
<div class="card-title" style="color: #991b1b;"><span class="badge-hard">ZORUNLU</span> Sert Kısıtlar (Hard Constraints)</div>
<p style="margin-bottom: 1rem;"><b>Üretim emniyeti ve yasal zorunluluk olup KESİNLİKLE İHLAL EDİLEMEZ:</b></p>
<div class="constraint-list-item" style="border-left: 4px solid #dc2626; background: #ffffff; padding: 1rem; margin-bottom: 1rem; border-radius: 6px;">
<b>1) Kesintisiz Üretim (Fırın Duruş Yasağı):</b><br>
Yüksek fırınlar ve döküm hatları soğutulamaz. Her 3 vardiyada <i>(Gündüz 08-16, Akşam 16-24, Gece 24-08)</i> sahada olması gereken minimum kadro sayısı eksiksiz karşılanmalıdır.
</div>
<div class="constraint-list-item" style="border-left: 4px solid #dc2626; background: #ffffff; padding: 1rem; margin-bottom: 1rem; border-radius: 6px;">
<b>2) Kritik Sertifika ve Ehliyet Zorunluluğu (MYK):</b><br>
Çelikhane ve dökümhanelerde her vardiyada en az <b>1 Tavan Vinci Operatörü</b>, <b>1 Potacı</b>, <b>1 Sıcak Metal Döküm Uzmanı</b> ve <b>1 Gaz İzleme Sorumlusu</b> bulunması zorunludur.
</div>
<div class="constraint-list-item" style="border-left: 4px solid #dc2626; background: #ffffff; padding: 1rem; margin-bottom: 1rem; border-radius: 6px;">
<b>3) 4857 Sayılı Kanun Gece Çalışması Sınırı (Madde 69):</b><br>
Gece vardiyasında (24:00 - 08:00) fiili çalışma süresi <b>7.5 saati aşamaz</b> (0.5 saat zorunlu yemek/ara dinlenme verilir).
</div>
<div class="constraint-list-item" style="border-left: 4px solid #dc2626; background: #ffffff; padding: 1rem; margin-bottom: 1rem; border-radius: 6px;">
<b>4) Vardiyalar Arası Minimum 11 Saat Dinlenme:</b><br>
Gece vardiyasından çıkan bir işçi, fizyolojik dinlenme zorunluluğu gereği <b>en az 11 saat kesintisiz dinlenmeden</b> ertesi vardiyaya yazılamaz.
</div>
<div class="constraint-list-item" style="border-left: 4px solid #dc2626; background: #ffffff; padding: 1rem; margin-bottom: 1rem; border-radius: 6px;">
<b>5) Zorunlu Hafta Tatili (Kayan 7 Günlük Pencere & İSG Standardı - Madde 46):</b><br>
Sabit takvim haftalarında yaşanabilecek <b>12 gün kesintisiz çalışma ve iş kazası riskini engellemek amacıyla</b> modelimizde İSG standartlarına uygun olarak <b>herhangi bir kayan 7 günlük periyotta en fazla 6 gün kesintisiz çalışma</b> yapılabilir. İşçinin 7 günlük kayan pencerede en az 1 gün zorunlu dinlenme (OFF) hakkı sert kısıt olarak garanti edilir.
</div>
</div>""", unsafe_allow_html=True)

    # --- Sağ Sütun: Yumuşak Kısıtlar (Soft Constraints) ---
    with col_steel_soft:
        st.markdown("""<div class="card-box" style="border-top: 5px solid #2563eb; background-color: #eff6ff;">
<div class="card-title" style="color: #1e40af;"><span class="badge-soft">ESNEK</span> Yumuşak Kısıtlar (Soft Constraints)</div>
<p style="margin-bottom: 1rem;"><b>Çözüm kalitesini, adaletini, çalışan memnuniyetini ve verimliliği artırır:</b></p>
<div class="constraint-list-item" style="border-left: 4px solid #2563eb; background: #ffffff; padding: 1rem; margin-bottom: 1rem; border-radius: 6px;">
<b>6) Sabit Posta / Takım Bütünlüğü (4 Posta Sistemi):</b><br>
Ekip uyumunu korumak amacıyla A, B, C ve D postalarındaki çalışanların vardiyalarda takım halinde birlikte görev yapması.
</div>
<div class="constraint-list-item" style="border-left: 4px solid #2563eb; background: #ffffff; padding: 1rem; margin-bottom: 1rem; border-radius: 6px;">
<b>7) Usta - Çırak (Kıdem) Dengesi:</b><br>
Vardiyalarda tecrübe ortalamasını yüksek tutmak için her vardiyada kıdemli ustalar ile yeni elemanların dengeli eşleştirilmesi.
</div>
<div class="constraint-list-item" style="border-left: 4px solid #2563eb; background: #ffffff; padding: 1rem; margin-bottom: 1rem; border-radius: 6px;">
<b>8) Adil Gece ve Bayram Nöbeti Dağılımı:</b><br>
Gece vardiyalarının, hafta sonu nöbetlerinin ve resmi bayram nöbetlerinin tüm personel arasında adil paylaştırılması.
</div>
<div class="constraint-list-item" style="border-left: 4px solid #2563eb; background: #ffffff; padding: 1rem; margin-bottom: 1rem; border-radius: 6px;">
<b>9) Sirkadiyen Ritim Uyumlu Vardiya Dönüşü:</b><br>
Vardiya döngülerinin biyolojik ritme uygun olarak saat yönünde <i>(Gündüz 08-16 ➔ Akşam 16-24 ➔ Gece 24-08)</i> ilerlemesi.
</div>
<div class="constraint-list-item" style="border-left: 4px solid #2563eb; background: #ffffff; padding: 1rem; margin-bottom: 1rem; border-radius: 6px;">
<b>10) Kişisel İzin ve Mazeret Taleplerine Uyum:</b><br>
Çalışanların önceden sunduğu özel mazeret izinlerine ve vardiya değişim (swap) taleplerine azami oranla uyulması.
</div>
</div>""", unsafe_allow_html=True)

    st.divider()

    # --- MATEMATİKSEL OPTİMİZASYON MODELİ VE FORMÜLASYONU (MILP) ---
    st.markdown("## 📐 Çelik Sanayisi Vardiya Optimizasyonu Matematiksel Formülasyonu (MILP)")
    st.write("Yukarıda tanımlanan kısıtların Yöneylem Araştırması (Karışık Tamsayılı Programlama - MILP) denklem kümelerine dönüştürülmüş matematiksel modeli:")

    m_col1, m_col2 = st.columns(2)

    with m_col1:
        st.markdown("### 1. Kümeler, İndeksler ve Karar Değişkenleri")
        st.markdown(r"""
        **Kümeler & İndeksler:**
        - $I = \{1, 2, \dots, N\}$: Personel kümesi ($i \in I$)
        - $T = \{1, 2, \dots, D\}$: Planlama periyodundaki günler ($t \in T$)
        - $K = \{1, 2, 3, 0\}$: Vardiya türleri ($1$: Gündüz 08-16, $2$: Akşam 16-24, $3$: Gece 24-08, $0$: İzin/OFF)
        - $S = \{\text{Vinç}, \text{Potacı}, \text{Döküm Uzmanı}, \text{Gaz Sorumlusu}\}$: Kritik yetkinlik kümeleri
        - $I_s \subseteq I$: $s$ sertifikasına sahip personel alt kümesi
        
        **İkili Karar Değişkeni (Binary Decision Variable):**
        """)
        st.latex(r"""
        x_{i,t,k} = \begin{cases} 1, & \text{eğer } i \text{ çalışanı } t \text{ gününde } k \text{ vardiyasına atanırsa} \\ 0, & \text{aksi halde} \end{cases}
        """)

    with m_col2:
        st.markdown("### 2. Amaç Fonksiyonu (Objective Function)")
        st.write("Yumuşak kısıtlardan sapma cezalarının **Minimizasyonu**:")
        st.latex(r"""
        \min Z = w_1 \sum_{i \in I} (d_i^+ + d_i^-) + w_2 \sum_{i \in I} \sum_{t \in T} P_{i,t} \cdot x_{i,t,0} + w_3 \sum_{i \in I} \sum_{t \in T} s_{i,t}^{\text{Sirkadiyen}}
        """)
        st.caption("Burada $w_1, w_2, w_3$ sapma ağırlık parametreleri, $d_i^+, d_i^-$ adil nöbet sapmaları, $P_{i,t}$ kişisel izin talebi ihlal cezasıdır.")

    st.divider()

    st.markdown("### 3. Matematiksel Sert Kısıt Denklem Kümeleri (Hard Constraints)")

    eq_col1, eq_col2 = st.columns(2)

    with eq_col1:
        st.markdown("**1. Tek Vardiya / Gün Çakışma Yasağı:**")
        st.latex(r"\sum_{k \in K} x_{i,t,k} = 1, \quad \forall i \in I, \forall t \in T")

        st.markdown("**2. Kesintisiz Üretim Minimum Kadro İhtiyacı (Sert Kısıt 1):**")
        st.latex(r"\sum_{i \in I} x_{i,t,k} \ge R_{t,k}, \quad \forall t \in T, \forall k \in \{1, 2, 3\}")

        st.markdown("**3. Kritik Sertifika ve Ehliyet Zorunluluğu (Sert Kısıt 2):**")
        st.latex(r"\sum_{i \in I_s} x_{i,t,k} \ge REQ_{s,k}, \quad \forall t \in T, \forall k \in \{1, 2, 3\}, \forall s \in S")

    with eq_col2:
        st.markdown("**4. Gece Vardiyası Sonrası Gündüz Yasağı (11 Saat Dinlenme - Sert Kısıt 3 & 4):**")
        st.latex(r"x_{i,t,3} + x_{i,t+1,1} \le 1, \quad \forall i \in I, \forall t \in \{1, \dots, D-1\}")

        st.markdown("**5. Zorunlu Hafta Tatili Kuralı (Kayan 7 Günlük Pencere - Sert Kısıt 5):**")
        st.latex(r"\sum_{\tau=t}^{t+6} x_{i,\tau,0} \ge 1, \quad \forall i \in I, \forall t \in \{1, \dots, D-6\}")

    st.markdown("### 4. Matematiksel Yumuşak Kısıt Denklem Kümeleri (Soft Constraints)")

    seq_col1, seq_col2 = st.columns(2)

    with seq_col1:
        st.markdown("**6. Gece Nöbeti Adil Dağılım Denklemi (Yumuşak Kısıt 8):**")
        st.latex(r"\sum_{t \in T} x_{i,t,3} - \bar{y}^N = d_i^+ - d_i^-, \quad \forall i \in I")

    with seq_col2:
        st.markdown("**7. Sirkadiyen Ritim / Saat Yönünde Vardiya Geçişi (Yumuşak Kısıt 9):**")
        st.latex(r"x_{i,t,2} + x_{i,t+1,1} \le 1 + s_{i,t}^{\text{Sirkadiyen}}, \quad \forall i \in I, \forall t \in \{1, \dots, D-1\}")
