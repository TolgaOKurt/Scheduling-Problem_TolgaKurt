"""
================================================================================
  VIEWS/TAB3_STEEL_MODEL.PY - SEKME 3: ÇELİK SANAYİ KISITLARI VE MILP MODELİ
================================================================================
"""
import streamlit as st
import pandas as pd

def render_tab3():
    """Sekme 3 içeriğini çizer: Çelik sanayi dinamikleri, 5 Sert / 5 Yumuşak kısıt ve MILP matematiksel modeli."""
    st.markdown("## 🏗️ Sekme 3: Çelik ve Metal Sanayisi Çizelgeleme Kısıtları & Matematiksel Modeli (MILP)")
    st.markdown("""
    Entegre demir-çelik tesisleri; yüksek fırınlar, çelikhaneler, sürekli döküm hatları ve haddehaneleriyle 
    **dünyanın en zorlu, tehlikeli ve kesintisiz (7/24/365)** çalışma ortamlarından biridir. 
    Bu sektörde vardiya çizelgelemesi; yalnızca personel planlaması değil, **doğrudan can güvenliği, proses sürekliliği ve milyonlarca dolarlık üretim emniyeti** meselesidir.
    """)

    # ==============================================================================
    # 1. ÇELİK SANAYİSİNİN ÖZEL OPERASYONEL DİNAMİKLERİ (GENİŞ İNCELEME)
    # ==============================================================================
    st.markdown("### 🏭 1. Çelik ve Metal Sanayisinin Özel Operasyonel Dinamikleri")
    
    st_col1, st_col2 = st.columns(2)

    with st_col1:
        st.markdown("""
        <div class="card-box" style="border-left: 5px solid #d97706; background-color: #fffbeb;">
            <div class="card-title" style="color: #92400e;">🔥 1.1. Termal Süreçler & Yüksek Fırın Duruş Yasağı</div>
            <p style="font-size: 0.92rem; color: #1e293b; line-height: 1.6;">
                Yüksek fırınlarda demir cevherinin sıvı pik demire dönüştürülmesi <b>1500°C - 1600°C</b> sıcaklıkta gerçekleşir. 
                Bu fırınlar <b>15 ila 20-25 yıllık kampanya ömürleri</b> boyunca sürekli sıcak tutularak işletilir; kısa süreli planlı bakımlarda dahi oda sıcaklığına soğutulmaz, hava girişi izole edilerek <b>uyutma (banking)</b> yöntemiyle sıcaklığı korunur.
            </p>
            <ul style="font-size: 0.88rem; color: #334155; padding-left: 1.1rem; margin-bottom: 0.4rem;">
                <li><b>Refrakter Astar Donması:</b> Fırında nöbetçi kadro eksikliği veya kontrolsüz enerji kesintisi nedeniyle fırın plansız durursa, dip haznedeki binlerce tonluk sıvı metal donar ve refrakter astar parçalanır.</li>
                <li><b>Maddi Hasar Boyutu:</b> Bir yüksek fırının kontrolsüz donması tesis için <b>10-50 Milyon Dolar</b> doğrudan hasar ve 3 ila 6 ay süren üretim duruşu anlamına gelir.</li>
                <li><b>Kesintisiz Kadro Zorunluluğu:</b> Her üç vardiya (Gündüz 08-16, Akşam 16-24, Gece 24-08) sahada kritik sertifikalı kadroyla %100 doldurulmak zorundadır.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card-box" style="border-left: 5px solid #2563eb; background-color: #eff6ff;">
            <div class="card-title" style="color: #1e40af;">👥 1.2. 4 Posta Sürekli Çalışma Sistemi (A, B, C, D)</div>
            <p style="font-size: 0.92rem; color: #1e293b; line-height: 1.6;">
                Çelik sanayisinde 7/24 kesintisiz üretimi sağlamak için çalışan kadrosu geleneksel olarak <b>4 Postaya (A, B, C, D)</b> ayrılır.
            </p>
            <ul style="font-size: 0.88rem; color: #334155; padding-left: 1.1rem; margin-bottom: 0.4rem;">
                <li><b>Çalışma Prensibi:</b> Herhangi bir günde 3 posta sahada aktif görev yapar (Gündüz, Akşam, Gece); 4. posta ise haftalık zorunlu dinlenme (OFF) iznindedir.</li>
                <li><b>Takım Bütünlüğü Önemi:</b> Birlikte çalışan posta üyelerinin refleksleri, iletişim dili ve acil durum uyumu sabittir. Postanın bölünerek farklı vardiyalara dağıtılması iş kazası riskini belirgin şekilde artırır.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with st_col2:
        st.markdown("""
        <div class="card-box" style="border-left: 5px solid #dc2626; background-color: #fef2f2;">
            <div class="card-title" style="color: #991b1b;">🪪 1.3. Kritik MYK Sertifika & Ehliyet Zorunlulukları</div>
            <p style="font-size: 0.92rem; color: #1e293b; line-height: 1.6;">
                Ağır sanayide her çalışan her operasyonu yürütemez. Mesleki Yeterlilik Kurumu (MYK) ve İSG mevzuatı gereği her vardiyada aşağıdaki 4 kritik yetkinlikten en az 1 uzman personelin bulunması <b>yasal zorunluluktur</b>:
            </p>
            <ul style="font-size: 0.88rem; color: #334155; padding-left: 1.1rem; margin-bottom: 0.4rem;">
                <li><b>🏗️ Tavan Vinci Operatörü:</b> 100+ tonluk sıvı çelik potalarını havadan taşıyan ve milimetrik manevra yapan yüksek riskli operatör.</li>
                <li><b>🪣 Potacı:</b> Sıvı metalin döküm potalarına alınması, cüruf temizliği ve pota nozul güvenliğinden sorumlu personel.</li>
                <li><b>🔥 Sıcak Metal Döküm Uzmanı:</b> Kalıplama ve sürekli döküm makinelerinde sıvı çeliği kütüğe dönüştüren proses uzmanı.</li>
                <li><b>☣️ Gaz İzleme Sorumlusu:</b> Yüksek fırın gazı (CO - Karbonmonoksit) sızıntılarını 7/24 dedektörlerle izleyerek zehirlenme ve patlamaları önleyen emniyet görevlisi.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card-box" style="border-left: 5px solid #059669; background-color: #f0fdf4;">
            <div class="card-title" style="color: #065f46;">⚖️ 1.4. Yasal Mevzuat & İSG Standartları</div>
            <p style="font-size: 0.92rem; color: #1e293b; line-height: 1.6;">
                Modelimiz, Türkiye Cumhuriyeti Çalışma Mevzuatı ve Toplu İş Sözleşmesi (TİS) hükümlerini tam olarak yansıtır:
            </p>
            <ul style="font-size: 0.88rem; color: #334155; padding-left: 1.1rem; margin-bottom: 0.4rem;">
                <li><b>4857 Sayılı Kanun Madde 69:</b> Gece döneminde (24:00 - 08:00) işçilerin fiili çalışma süresi <b>7.5 saati aşamaz</b>.</li>
                <li><b>4857 Sayılı Kanun Madde 46:</b> İşçilere 7 günlük bir zaman dilimi içinde kesintisiz <b>en az 24 saat hafta tatili</b> verilmesi zorunludur.</li>
                <li><b>Vardiyalar Arası 11 Saat Dinlenme:</b> Gece nöbetinden çıkan işçinin fizyolojik toparlanması için en az 11 saat dinlenmeden sabah vardiyasına yazılması yasaktır.</li>
                <li><b>Sirkadiyen Ritim (Biyolojik Saat):</b> Vardiyaların saat yönünde (Gündüz → Akşam → Gece → İzin) dönmesi yorgunluk kaynaklı refleks kaybını önler.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ==============================================================================
    # 2. 5 SERT (HARD) VE 5 YUMUŞAK (SOFT) KISIT DETAY TABLOSU
    # ==============================================================================
    st.markdown("### ⚖️ 2. Çelik Tesis Modeli: 5 Sert (Hard) ve 5 Yumuşak (Soft) Kısıt Yapısı")
    st.markdown("""
    Optimizasyon algoritmalarımızda uygulanan 10 temel kısıtın detaylı operasyonel gerekçeleri ve yasal dayanakları:
    """)

    col_steel_hard, col_steel_soft = st.columns(2)

    # --- Sol Sütun: Sert Kısıtlar (Hard Constraints) ---
    with col_steel_hard:
        st.markdown("""
        <div class="card-box" style="border-top: 5px solid #dc2626; background-color: #fff8f8;">
            <div class="card-title" style="color: #991b1b;"><span class="badge-hard">ZORUNLU</span> 5 Sert Kısıt (Hard Constraints)</div>
            <p style="margin-bottom: 0.8rem; font-weight: 600; font-size: 0.92rem; color: #7f1d1d;">
                Üretim emniyeti ve yasal zorunluluk olup %100 sağlanmalıdır (İhlal = Feasible Değil):
            </p>
            <div class="constraint-list-item" style="border-left: 4px solid #dc2626;">
                <b>1) Kesintisiz Üretim (Fırın Duruş Yasağı):</b><br>
                <span style="font-size: 0.88rem; color: #334155;">
                    Her 3 vardiyada (Gündüz 08-16, Akşam 16-24, Gece 24-08) sahada olması gereken minimum kadro sayısı tam karşılanmalıdır.
                </span>
            </div>
            <div class="constraint-list-item" style="border-left: 4px solid #dc2626;">
                <b>2) Kritik 4 MYK Sertifika & Ehliyet Zorunluluğu:</b><br>
                <span style="font-size: 0.88rem; color: #334155;">
                    Her vardiyada en az 1 Tavan Vinci Operatörü, 1 Potacı, 1 Döküm Uzmanı ve 1 Gaz İzleme Sorumlusu hazır bulunmalıdır.
                </span>
            </div>
            <div class="constraint-list-item" style="border-left: 4px solid #dc2626;">
                <b>3) 4857 Sayılı Kanun Md. 69 (Gece Çalışma Sınırı):</b><br>
                <span style="font-size: 0.88rem; color: #334155;">
                    Gece vardiyasında fiili çalışma 7.5 saati aşamaz (0.5 saat zorunlu yemek/ara dinlenme verilir).
                </span>
            </div>
            <div class="constraint-list-item" style="border-left: 4px solid #dc2626;">
                <b>4) Vardiyalar Arası Minimum 11 Saat Dinlenme:</b><br>
                <span style="font-size: 0.88rem; color: #334155;">
                    Gece vardiyasından çıkan bir işçi, fizyolojik toparlanma için en az 11 saat dinlenmeden ertesi sabah vardiyasına yazılamaz.
                </span>
            </div>
            <div class="constraint-list-item" style="border-left: 4px solid #dc2626;">
                <b>5) Zorunlu Hafta Tatili (Kayan 7 Günlük İSG Standardı):</b><br>
                <span style="font-size: 0.88rem; color: #334155;">
                    12 gün kesintisiz çalışma ve iş kazası riskini engellemek amacıyla herhangi bir kayan 7 günlük periyotta en fazla 6 gün kesintisiz çalışılabilir; en az 1 gün zorunlu OFF dinlenmesi garanti edilir.
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- Sağ Sütun: Yumuşak Kısıtlar (Soft Constraints) ---
    with col_steel_soft:
        st.markdown("""
        <div class="card-box" style="border-top: 5px solid #2563eb; background-color: #f8faff;">
            <div class="card-title" style="color: #1e40af;"><span class="badge-soft">ESNEK</span> 5 Yumuşak Kısıt (Soft Constraints)</div>
            <p style="margin-bottom: 0.8rem; font-weight: 600; font-size: 0.92rem; color: #1e3a8a;">
                Çözüm kalitesini, çalışan memnuniyetini ve operasyonel verimliliği artırır (İhlal = Ceza Puanı):
            </p>
            <div class="constraint-list-item" style="border-left: 4px solid #2563eb;">
                <b>6) Posta Takım Bütünlüğü (4 Posta Sistemi):</b><br>
                <span style="font-size: 0.88rem; color: #334155;">
                    Ekip uyumu ve refleks birliği için A, B, C ve D postalarındaki çalışanların vardiyalarda takım halinde birlikte görev yapması.
                </span>
            </div>
            <div class="constraint-list-item" style="border-left: 4px solid #2563eb;">
                <b>7) Usta - Çırak (Kıdem & Tecrübe) Dengesi:</b><br>
                <span style="font-size: 0.88rem; color: #334155;">
                    Vardiyalarda tecrübe ortalamasını korumak amacıyla her vardiyada kıdemli ustalar ile yeni personelin dengeli eşleştirilmesi.
                </span>
            </div>
            <div class="constraint-list-item" style="border-left: 4px solid #2563eb;">
                <b>8) Adil Gece Nöbeti Dağılımı:</b><br>
                <span style="font-size: 0.88rem; color: #334155;">
                    Gece vardiyalarının tüm personel arasında homojen ve adil paylaştırılması (ortalama hedeften sapmaların minimizasyonu).
                </span>
            </div>
            <div class="constraint-list-item" style="border-left: 4px solid #2563eb;">
                <b>9) Sirkadiyen Ritim Uyumlu Vardiya Dönüşü:</b><br>
                <span style="font-size: 0.88rem; color: #334155;">
                    Vardiya geçişlerinin biyolojik ritme uygun olarak saat yönünde (Gündüz → Akşam → Gece → İzin) ilerlemesi; akşamdan hemen sabah vardiyasına ters dönüşün engellenmesi.
                </span>
            </div>
            <div class="constraint-list-item" style="border-left: 4px solid #2563eb;">
                <b>10) Kişisel İzin ve Mazeret Taleplerine Uyum:</b><br>
                <span style="font-size: 0.88rem; color: #334155;">
                    Çalışanların önceden sunduğu özel mazeret izinlerine ve vardiya değişim taleplerine azami düzeyde uyulması.
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ==============================================================================
    # 3. MATEMATİKSEL OPTİMİZASYON MODELİ VE FORMÜLASYONU (MILP)
    # ==============================================================================
    st.markdown("## 📐 3. Çelik Sanayisi Vardiya Optimizasyonu Matematiksel Modeli (MILP)")
    st.markdown("""
    Yukarıda tanımlanan endüstriyel kısıtların Yöneylem Araştırması (Karışık Tamsayılı Doğrusal Programlama - MILP) 
    standartlarına dönüştürülmüş tam matematiksel formülasyonu aşağıda açıklanmıştır:
    """)

    # 3.1 Kümeler ve Değişken Sözlüğü
    st.markdown("### 3.1. Kümeler, İndeksler ve Karar Değişkenleri")
    
    df_notation = pd.DataFrame({
        "Sembol": [
            "I = {1, 2, ..., N}",
            "T = {1, 2, ..., D}",
            "K = {0, 1, 2, 3}",
            "S (MYK Kümesi)",
            "Iₛ ⊆ I",
            "Rₜ,ₖ",
            "xᵢ,ₜ,ₖ ∈ {0, 1}",
            "dᵢ⁺, dᵢ⁻ ≥ 0",
            "posta_devₚ,ₜ ≥ 0",
            "sᵢ,ₜˢⁱʳᵏ ∈ {0, 1}",
            "no_ustaₜ,ₖ ∈ {0, 1}"
        ],
        "Tür": [
            "Küme / İndeks",
            "Küme / İndeks",
            "Küme / İndeks",
            "Küme",
            "Alt Küme",
            "Parametre",
            "İkili Karar Değişkeni",
            "Sürekli Sapma Değişkeni",
            "Sürekli Sapma Değişkeni",
            "İkili Ceza Değişkeni",
            "İkili Ceza Değişkeni"
        ],
        "Tanım & Açıklama": [
            "Fabrikadaki toplam personel kümesi (i ∈ I)",
            "Planlama periyodundaki günler kümesi (t ∈ T)",
            "Vardiya türleri: 1: Gündüz (08-16), 2: Akşam (16-24), 3: Gece (24-08), 0: İzin (OFF)",
            "Kritik MYK yetkinlikleri kümesi: {Vinç Operatörü, Potacı, Döküm Uzmanı, Gaz Sorumlusu}",
            "s ehliyetine/sertifikasına sahip çalışanların alt kümesi",
            "t günü k vardiyasında sahada bulunması zorunlu minimum personel sayısı",
            "i çalışanı t gününde k vardiyasına atanırsa 1, aksi halde 0",
            "i çalışanının dönem boyunca tuttuğu gece nöbeti sayısının ortalamadan pozitif/negatif sapması",
            "t gününde p postasındaki çalışanların ana çoğunluktan ayrılma (bölünme) sayısı",
            "i çalışanının t gününde akşam (2), t+1 gününde gündüz (1) çalışması halinde 1 olan sirkadiyen ihlal değişkeni",
            "t günü k vardiyasında en az bir kıdemli usta bulunmuyorsa 1 değerini alan ceza değişkeni"
        ]
    })
    st.dataframe(df_notation, width="stretch", hide_index=True)

    # 3.2 Karar Değişkeni Formülü
    st.markdown("#### İkili Karar Değişkeni (Binary Decision Variable):")
    st.latex(r"""
    x_{i,t,k} = \begin{cases} 
    1, & \text{eğer } i \text{ çalışanı } t \text{ gününde } k \text{ vardiyasına atanırsa} \\ 
    0, & \text{aksi halde (OFF veya başka vardiya)} 
    \end{cases}
    """)

    st.divider()

    # 3.3 Amaç Fonksiyonu
    st.markdown("### 3.2. Amaç Fonksiyonu (Objective Function - Ağırlıklı Ceza Minimizasyonu)")
    st.markdown("""
    Modelimiz, geçerli (feasible) tüm çözümler arasından **toplam ceza puanını (Z) minimize eden** en kaliteli çizelgeyi arar:
    """)
    st.latex(r"""
    \min Z = w_{\text{posta}} \sum_{p \in P} \sum_{t \in T} \text{posta\_dev}_{p,t} 
    + w_{\text{sirk}} \sum_{i \in I} \sum_{t=1}^{D-1} s_{i,t}^{\text{sirk}} 
    + w_{\text{izin}} \sum_{i \in I} (1 - x_{i, p_i, 0}) 
    + w_{\text{gece}} \sum_{i \in I} (d_i^+ + d_i^-) 
    + w_{\text{usta}} \sum_{t \in T} \sum_{k=1}^3 \text{no\_usta}_{t,k}
    """)
    st.caption("Burada w_posta, w_sirk, w_izin, w_gece, w_usta kullanıcı tarafından kontrol panelinde belirlenen ceza katsayılarıdır.")

    st.divider()

    # 3.4 Sert Kısıt Denklem Kümeleri
    st.markdown("### 3.3. Matematiksel Sert Kısıt Denklem Kümeleri (Hard Constraints)")
    st.markdown("Aşağıdaki denklemler çözümün geçerli sayılması için %100 sağlanmak zorundadır:")

    eq1_col, eq2_col = st.columns(2)

    with eq1_col:
        st.markdown("**1. Tek Vardiya / Çakışma Yasağı:**")
        st.caption("Her çalışan bir günde yalnızca 1 göreve (Gündüz, Akşam, Gece veya İzin) atanabilir.")
        st.latex(r"\sum_{k \in \{0,1,2,3\}} x_{i,t,k} = 1, \quad \forall i \in I, \forall t \in T")

        st.markdown("**2. Kesintisiz Üretim Minimum Kadro İhtiyacı (Sert Kısıt 1):**")
        st.caption("Her vardiyada sahada gereken asgari personel sayısı tam olarak karşılanmalıdır.")
        st.latex(r"\sum_{i \in I} x_{i,t,k} \ge R_{t,k}, \quad \forall t \in T, \forall k \in \{1, 2, 3\}")

        st.markdown("**3. Kritik 4 MYK Sertifikası Zorunluluğu (Sert Kısıt 2):**")
        st.caption("Her vardiyada 4 kritik ehliyetin (Vinç, Potacı, Döküm, Gaz) her birinden en az 1 uzman bulunmalıdır.")
        st.latex(r"\sum_{i \in I_s} x_{i,t,k} \ge 1, \quad \forall t \in T, \forall k \in \{1, 2, 3\}, \forall s \in S")

    with eq2_col:
        st.markdown("**4. Gece Vardiyası Sonrası 11 Saat Dinlenme (Sert Kısıt 3 & 4):**")
        st.caption("Gece vardiyasından (k=3) çıkan işçi ertesi gün sabah (k=1) vardiyasına yazılamaz.")
        st.latex(r"x_{i,t,3} + x_{i,t+1,1} \le 1, \quad \forall i \in I, \forall t \in \{1, \dots, D-1\}")

        st.markdown("**5. Zorunlu Hafta Tatili (Kayan 7 Günlük İSG Standardı - Sert Kısıt 5):**")
        st.caption("Herhangi bir ardışık 7 günlük periyotta işçinin en az 1 gün OFF (k=0) izni olması garantilenir.")
        st.latex(r"\sum_{\tau=t}^{t+6} x_{i,\tau,0} \ge 1, \quad \forall i \in I, \forall t \in \{1, \dots, D-6\}")

    st.divider()

    # 3.4 Yumuşak Kısıt Doğrusallaştırma Denklemleri
    st.markdown("### 3.4. Matematiksel Yumuşak Kısıt Doğrusallaştırma Denklemleri (Soft Constraints)")
    st.markdown("Doğrusal programlama (MILP) yapısını korumak amacıyla doğrusal olmayan hedefler yapay değişkenlerle doğrusallaştırılmıştır:")

    seq1_col, seq2_col = st.columns(2)

    with seq1_col:
        st.markdown("**6. Posta Takım Bütünlüğü Doğrusallaştırması (Yumuşak Kısıt 6):**")
        st.caption("Aynı postadaki çalışanların ana çoğunluktan ayrılması (sapması) değişkenlerle yakalanır:")
        st.latex(r"\text{posta\_dev}_{p,t} \ge \sum_{i \in I_p, k \ne k_{\text{maj}}} x_{i,t,k}, \quad \text{posta\_dev}_{p,t} \ge 0")

        st.markdown("**7. Usta - Çırak / Kıdemli Usta Varlığı (Yumuşak Kısıt 7):**")
        st.caption("Vardiyada görevli personeller arasında kıdemli usta yoksa no_usta değişkeni 1 olur:")
        st.latex(r"\text{no\_usta}_{t,k} \ge 1 - \sum_{i \in I_{\text{usta}}} x_{i,t,k}, \quad \text{no\_usta}_{t,k} \in \{0, 1\}")

        st.markdown("**8. Gece Nöbeti Adil Dağılım Doğrusallaştırması (Yumuşak Kısıt 8):**")
        st.caption("Hedef ortalama gece nöbeti sayısından (y_hedef) sapmalar pozitif/negatif değişkenlerle yakalanır:")
        st.latex(r"\sum_{t \in T} x_{i,t,3} - \bar{y}^N = d_i^+ - d_i^-, \quad d_i^+, d_i^- \ge 0, \quad \forall i \in I")

    with seq2_col:
        st.markdown("**9. Sirkadiyen Ritim İleri Yönlü Geçiş (Yumuşak Kısıt 9):**")
        st.caption("Akşam vardiyasından (k=2) ertesi gün sabah vardiyasına (k=1) ters geçiş cezalandırılır:")
        st.latex(r"s_{i,t}^{\text{sirk}} \ge x_{i,t,2} + x_{i,t+1,1} - 1, \quad s_{i,t}^{\text{sirk}} \ge 0")

        st.markdown("**10. Kişisel İzin Taleplerinin Karşılanması (Yumuşak Kısıt 10):**")
        st.caption("İşçinin talep ettiği izin gününde (p_i) çalışması (x ≠ 0) halinde ceza tetiklenir:")
        st.latex(r"\text{pref\_viol}_i \ge 1 - x_{i, p_i, 0}, \quad \text{pref\_viol}_i \in \{0, 1\}")

    st.divider()

    # 3.6 Teorik Alt Sınır & MIP Gap Açıklaması
    st.markdown("### 3.5. Teorik Alt Sınır (Continuous LP Relaxation Best Bound) & MIP Gap Nedir?")
    
    bb_col1, bb_col2 = st.columns([3, 2])

    with bb_col1:
        st.markdown(r"""
        MILP çözücülerinde bulunan bir çözümün **"Gerçekten En İyi (Optimal)"** olduğunu kanıtlamak için **Teorik Alt Sınır (Best Bound)** referans alınır:
        
        * **Sürekli LP Gevşetmesi (Continuous LP Relaxation):** Tamsayılık zorunluluğu (<i>x</i> &isin; {0, 1}) kaldırılarak değişkenlerin sürekli gerçel sayılar (0 &le; <i>x</i> &le; 1) olmasına izin verilir.
        * **Neden Binlerce Kat Daha Hızlıdır?** Sürekli LP problemleri **P (Polinomial Time)** sınıfındadır ve Simplex / İç Nokta algoritmaları ile **1-5 milisaniyede** çözülür. MILP ise **NP-Hard** olup arka planda binlerce kez bu LP gevşetmesini çözer.
        * **Neden Doğrudan Çizelge Olarak Kullanılamaz?** LP gevşetmesi *"Ahmet Salı günü %40 Gündüz, %60 Gece çalışsın"* gibi kesirli sonuçlar üretir (insan bölünemez). Ancak bu kesirli çözümün ceza puanı, hiçbir tamsayılı çözümün altına inemeyeceği **matematiksel mutlak taban puanını (<i>Z</i><sub>bound</sub>)** verir.
        * **MIP Gap (Optimizasyon Açıklığı):** Bulunan geçerli çözüm (<i>Z</i><sub>best</sub>) ile teorik alt sınır (<i>Z</i><sub>bound</sub>) arasındaki yüzdesel farktır:
        """, unsafe_allow_html=True)
        st.latex(r"\text{MIP Gap (\%)} = \frac{|Z_{\text{best}} - Z_{\text{bound}}|}{\max(1, Z_{\text{best}})} \times 100")

    with bb_col2:
        st.markdown("""
        <div style="background-color: #ffffff; border: 2px solid #e2e8f0; border-radius: 10px; padding: 1.2rem; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
            <div style="font-weight: 700; color: #0f172a; margin-bottom: 0.6rem; font-size: 1.05rem;">🏆 Matematiksel Garanti Kriterleri</div>
            <div style="font-size: 0.88rem; color: #334155; line-height: 1.6;">
                • <b>MIP Gap = %0.0:</b> Çözümün küresel olarak en iyi (Global Optimal) olduğu matematiksel olarak kanıtlanmıştır.<br>
                • <b>MIP Gap > %0:</b> Zaman sınırı nedeniyle solver aramayı durdurmuştur; çözüm geçerlidir fakat teorik alt sınırdan sapma mevcuttur.<br>
                • <b>Infeasible:</b> Verilen personel sayısı veya sertifikalarla 5 sert kısıtın aynı anda sağlanması matematiksel olarak imkansızdır.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 3.7 Alternatif Optimal Çözümler & Simetri Kutusu
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f0fdf4 0%, #eff6ff 100%); border: 1px solid #bfdbfe; border-radius: 10px; padding: 1.3rem; margin-top: 1.2rem;">
        <h4 style="color: #1e40af; margin-top: 0; margin-bottom: 0.6rem;">💡 Kritik Soru: MIP Gap = %0.0 Olan Birden Fazla Farklı Çizelge Olabilir mi?</h4>
        <p style="font-size: 0.95rem; color: #1e293b; line-height: 1.65; margin-bottom: 0.6rem;">
            <b>Evet, kesinlikle olabilir!</b> Literatürde buna <b>"Alternatif / Çoklu Global Optima (Multiple Global Optima)"</b> veya <b>"İşçi Simetrisi (Symmetry)"</b> denir:
        </p>
        <ul style="font-size: 0.9rem; color: #334155; line-height: 1.6; margin-bottom: 0.6rem; padding-left: 1.2rem;">
            <li><b>Aynı Minimum Ceza, Farklı Çizelge:</b> MIP Gap'in %0.0 olması, amaç fonksiyonu değerinin (toplam cezanın <i>Z*</i>) ulaşılabilecek en dip seviyede olduğunu gösterir. Ancak aynı minimum cezayı veren birden fazla farklı vardiya atama matrisi (<i>X</i><sup>(1)</sup> &ne; <i>X</i><sup>(2)</sup>) bulunabilir.</li>
            <li><b>Somut Simetri Örneği:</b> Aynı yetkinliğe sahip iki vinç operatörünün (Ahmet ve Mehmet) Pazartesi ve Salı günkü gece/gündüz nöbetlerini aralarında değiş tokuş yapması toplam ceza puanını değiştirmez; her iki çizelge de <b>%100 Global Optimaldir</b>.</li>
            <li><b>Yönetimsel Avantaj:</b> Bu durum fabrika yönetimine esneklik sağlar; matematiksel olarak eşit kalitedeki alternatif optimal çizelgeler arasından en uygun insani tercih seçilebilir.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
