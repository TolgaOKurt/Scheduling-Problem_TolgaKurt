"""
================================================================================
  VIEWS/COMMON_COMPONENTS.PY - MERKEZİ ARAYÜZ VE GÖRSELLEŞTİRME BİLEŞENLERİ
================================================================================
  Bu modül, tüm optimizasyon sekmelerinde kullanılan ortak grafik, tablo,
  ısı haritası, matris çizelgesi ve metrik kartı bileşenlerini içerir.
  
  Tüm fonksiyonlar saf (stateless) olup verileri bağımsız parametre olarak alır;
  dataların birbirine karışması %0 oranında engellenmiştir.
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


# 1. ÖN-ANALİZ MALİYET & TAHMİNİ SÜRE ROZETİ (BADGE)
def render_evaluator_cost_badge(calls, call_cost_ms, label="Tahmini Ceza Değerlendirme"):
    """
    Şık ve modern bir rozet (Badge Pill) ile tahmini ceza çağrısını,
    birim maliyet formülünü ve tahmini CPU süresini gösterir.
    """
    total_ms = calls * call_cost_ms
    time_str = f"{total_ms/1000:.2f} sn" if total_ms >= 1000 else (f"{total_ms:.1f} ms" if total_ms >= 1 else f"{total_ms:.2f} ms")
    
    st.markdown(f"""
    <div class="metric-card" style="padding: 12px 14px; margin-bottom: 8px;">
        <div class="metric-label">{label}</div>
        <div style="display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; margin-top: 4px; gap: 6px;">
            <span class="metric-value" style="color: #1e293b; font-size: 1.35rem;">~{calls:,} <span style="font-size: 0.85rem; font-weight: 500; color: #64748b;">Çağrı</span></span>
            <span style="background: linear-gradient(135deg, #e0f2fe 0%, #dbeafe 100%); color: #0369a1; border: 1px solid #bae6fd; font-weight: 700; font-size: 0.82rem; padding: 3px 8px; border-radius: 16px; display: inline-flex; align-items: center; gap: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                ⏱️ ~{time_str}
            </span>
        </div>
        <div style="font-size: 0.76rem; color: #64748b; margin-top: 5px;">
            🧮 <code>{call_cost_ms:.3f} ms</code> &times; <code>{calls:,}</code> = <b>{time_str}</b> (Saf CPU)
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_preanalysis_prediction_card(title, calls, call_cost_ms, desc_items=None):
    """
    Algoritmaların çalıştırma butonunun hemen üstünde yer alan
    tahmini ceza değerlendirme çağrısı ve birim maliyet çarpım panosu.
    """
    st.markdown(f"#### 🔍 {title}")
    
    m_col1, m_col2 = st.columns([1, 2])
    with m_col1:
        render_evaluator_cost_badge(calls, call_cost_ms, label="Tahmini Ceza Değerlendirme")
    with m_col2:
        if desc_items:
            bullets = "".join([f"<li>{item}</li>" for item in desc_items])
            st.markdown(f"""
            <div style="background-color: #f8fafc; padding: 10px 14px; border-radius: 8px; border-left: 4px solid #3b82f6; font-size: 0.88rem;">
                <b>Hesaplama Dinamikleri:</b>
                <ul style="margin-top: 4px; margin-bottom: 0; padding-left: 20px;">
                    {bullets}
                </ul>
            </div>
            """, unsafe_allow_html=True)


# 2. ÇALIŞAN İŞ YÜKÜ VE GECE NÖBETİ DAĞILIMI BARI
def render_workload_chart(schedule, workers, key_prefix, title="Çalışan İş Yükü ve Gece Nöbeti Dağılımı", color_seq=None):
    """Her çalışanın toplam çalıştığı gün ve tuttuğu gece nöbeti sayısını karşılaştırmalı bar grafik olarak çizer."""
    if color_seq is None:
        color_seq = ["#0284c7", "#dc2626"]
        
    num_workers = len(workers)
    work_days = [int(np.sum(schedule[i, :] > 0)) for i in range(num_workers)]
    night_days = [int(np.sum(schedule[i, :] == 3)) for i in range(num_workers)]

    df_workload = pd.DataFrame({
        "İşçi": [w['name'] for w in workers],
        "Toplam Çalışma": work_days,
        "Gece Nöbeti": night_days
    })

    fig = px.bar(
        df_workload,
        x="İşçi",
        y=["Toplam Çalışma", "Gece Nöbeti"],
        barmode="group",
        color_discrete_sequence=color_seq
    )
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        height=380,
        legend=dict(orientation="h", y=1.15),
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.markdown(f"##### {title}")
    st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_workload")


# 3. YUMUŞAK KISIT CEZA PUANI DAĞILIMI (YATAY BAR)
def render_penalties_chart(penalties, key_prefix, title="Minimize Edilmiş Yumuşak Kısıt Cezaları (min Z)"):
    """5 yumuşak kısıt türünden gelen ceza puanlarını yatay renkli bar grafiğinde görselleştirir."""
    df_penalties = pd.DataFrame({
        "Kısıt Tipi": list(penalties.keys()),
        "Ceza Puanı": list(penalties.values())
    })
    fig = px.bar(
        df_penalties,
        x="Ceza Puanı",
        y="Kısıt Tipi",
        orientation="h",
        text="Ceza Puanı",
        color="Ceza Puanı",
        color_continuous_scale="Reds"
    )
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        height=340,
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.markdown(f"##### {title}")
    st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_penalties")


# 4. POSTA BAZINDA (A, B, C, D) YÜK VE GECE NÖBETİ DAĞILIMI
def render_posta_load_chart(schedule, workers, key_prefix, title="Posta Bazında (A, B, C, D) Gece Nöbeti ve Yük Dağılımı"):
    """4 postanın (Posta A, B, C, D) toplam vardiya ve gece vardiyası dağılımını çizer."""
    posta_data = []
    for p in ['Posta A', 'Posta B', 'Posta C', 'Posta D']:
        p_wids = [w['id'] for w in workers if w['posta'] == p]
        if len(p_wids) > 0:
            tot_w = int(np.sum(schedule[p_wids, :] > 0))
            tot_n = int(np.sum(schedule[p_wids, :] == 3))
            posta_data.append({"Posta": p, "Toplam Vardiya": tot_w, "Gece Vardiyası": tot_n})

    df_posta = pd.DataFrame(posta_data)
    fig = px.bar(
        df_posta,
        x="Posta",
        y=["Toplam Vardiya", "Gece Vardiyası"],
        barmode="group",
        color_discrete_sequence=["#2563eb", "#dc2626"]
    )
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        height=340,
        legend=dict(orientation="h", y=1.15),
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.markdown(f"##### {title}")
    st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_posta_load")


# 5. GÜN VE VARDİYA BAZINDA POSTA DAĞILIMI (YIĞILMIŞ STACKED BAR)
def render_shift_posta_stacked_chart(schedule, workers, num_days, key_prefix, title="Gün ve Vardiya Bazında Posta Dağılımı"):
    """Her gün ve vardiyada (Gündüz, Akşam, Gece) hangi postadan kaç kişi çalıştığını yığılmış sütun olarak gösterir."""
    shift_labels = {1: "Gündüz (08-16)", 2: "Akşam (16-24)", 3: "Gece (24-08)"}
    shift_posta_rows = []

    for d in range(num_days):
        for k in [1, 2, 3]:
            for p in ['Posta A', 'Posta B', 'Posta C', 'Posta D']:
                p_wids = [w['id'] for w in workers if w['posta'] == p]
                count_in_shift = sum(1 for wid in p_wids if schedule[wid, d] == k)
                shift_posta_rows.append({
                    "Gün_Vardiya": f"G{d+1} - {shift_labels[k]}",
                    "Gün": f"Gün {d+1:02d}",
                    "Vardiya": shift_labels[k],
                    "Posta": p,
                    "Çalışan Sayısı": int(count_in_shift)
                })

    df_shift_posta = pd.DataFrame(shift_posta_rows)
    fig = px.bar(
        df_shift_posta,
        x="Gün_Vardiya",
        y="Çalışan Sayısı",
        color="Posta",
        barmode="stack",
        text="Çalışan Sayısı",
        color_discrete_map={
            "Posta A": "#2563eb",
            "Posta B": "#059669",
            "Posta C": "#d97706",
            "Posta D": "#7c3aed"
        }
    )
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        height=430,
        legend=dict(orientation="h", y=1.15),
        xaxis_tickangle=-45,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.markdown(f"### 🏢 {title}")
    st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_shift_posta_stack")


# 6. VARDİYA ATAMA RENKLİ ISI HARİTASI (HEATMAP)
def render_schedule_heatmap(schedule, workers, num_days, key_prefix, title="Vardiya Çizelgesi Isı Haritası (Heatmap)"):
    """Vardiya atamalarını (0:OFF, 1:Gündüz, 2:Akşam, 3:Gece) renk skalalı ısı haritası olarak çizer."""
    fig = px.imshow(
        schedule,
        labels=dict(x="Gün", y="İşçi", color="Vardiya Kodu"),
        x=[f"G{d+1}" for d in range(num_days)],
        y=[w['name'] for w in workers],
        color_continuous_scale=[[0, '#cbd5e1'], [0.33, '#fde047'], [0.66, '#f97316'], [1.0, '#1e3a8a']]
    )
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
    st.markdown(f"##### {title}")
    st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_heatmap")


# 7. TAM VARDİYA ÇİZELGESİ MATRİS TABLOSU (AÇILIR/KAPANIR EXPANDER)
def render_schedule_matrix_table(schedule, workers, num_days, title="🗓️ Tam Vardiya Çizelge Tablosu", expanded=False):
    """Tüm çalışanların gün bazlı vardiyalarını metinsel (Gündüz, Akşam, Gece, OFF) açılır/kapanır expander tablo olarak listeler."""
    with st.expander(title, expanded=expanded):
        shift_names = {0: "OFF", 1: "Gündüz", 2: "Akşam", 3: "Gece"}
        matrix_data = []

        for i, w in enumerate(workers):
            row = {
                "İşçi": w['name'],
                "Posta": w['posta'],
                "Unvan": "Kıdemli Usta" if w.get('is_usta', False) else "İşçi"
            }
            for d in range(num_days):
                row[f"Gün {d+1}"] = shift_names.get(schedule[i, d], "OFF")
            matrix_data.append(row)

        df_view = pd.DataFrame(matrix_data)
        st.dataframe(df_view, width="stretch", hide_index=True)


# 8. KİŞİSEL İZİN TALEPLERİ DETAY RAPORU TABLOSU
def render_request_details_expander(request_details, title="📋 Kişisel İzin Talepleri ve Karşılanma Durumu Detay Raporu", expanded=False):
    """Çözücünün ürettiği request_details listesini genişletilebilir panel içinde biçimlendirerek gösterir."""
    with st.expander(title, expanded=expanded):
        df_reqs = pd.DataFrame(request_details)
        df_reqs = df_reqs.rename(columns={
            'id': 'İşçi ID',
            'name': 'İşçi Adı',
            'posta': 'Posta',
            'unvan': 'Unvan',
            'talep_gun': 'Talep Edilen İzin (Gün)',
            'atandi_vardiya': 'Atanan Vardiya',
            'durum': 'Talep Durumu',
            'ceza_puani': 'Ceza Puanı'
        })
        st.dataframe(df_reqs, width="stretch", hide_index=True)


# 9. ÇALIŞAN KADRO PROFİLLERİ VE NİTELİKLERİ TABLOSU (AÇILIR/KAPANIR EXPANDER)
def render_worker_profiles_table(workers, title="🪪 Aktif Personel Yetkinlik ve MYK Sertifika Kadro Listesi", expanded=False):
    """Çalışanların postalarını, kıdem unvanlarını, MYK sertifikalarını ve talep ettikleri izin günlerini açılır/kapanır expander panelde listeler."""
    with st.expander(title, expanded=expanded):
        profile_rows = []
        for w in workers:
            skills_formatted = ", ".join(sorted(list(w.get('skills', []))))
            profile_rows.append({
                "İşçi Adı": w['name'],
                "Posta": w.get('posta', '-'),
                "Unvan": "Kıdemli Usta" if w.get('is_usta', False) else "Operatör/İşçi",
                "Sahip Olduğu MYK Sertifika & Ehliyetler": skills_formatted,
                "Talep Ettiği İzin (Gün)": int(w.get('pref_off', 1))
            })
        df_profiles = pd.DataFrame(profile_rows)
        st.dataframe(df_profiles, width="stretch", hide_index=True)


# 10. SERT KISIT UYGUNLUK VE İHLAL BİLDİRİM KARTI
def render_hard_constraints_status_card(is_feasible, hard_violations_count=0, hard_violation_logs=None, solver_name="Çözücü"):
    """
    Sert Kısıt uygunluk durumunu (Feasible / Infeasible) ekranda görsel uyarı kartı ve
    genişletilebilir ayrıntılı hata listesi olarak render eder.
    """
    if hard_violation_logs is None:
        hard_violation_logs = []

    if is_feasible:
        st.success("✅ **Sert Kısıt Uygunluğu:** %100 GEÇERLİ ÇÖZÜM (Tüm vardiya kotaları, 4 MYK zorunlu ehliyeti ve yasal dinlenme süreleri eksiksiz sağlandı.)")
    else:
        st.error(f"""
        🚨 **DİKKAT: SERT KISIT İHLALİ TESPİT EDİLDİ ({hard_violations_count:,} Adet İhlal - Çözüm Matematiksel Olarak Geçersizdir!)**
        
        📌 **Yöneylem Araştırması ve Matematiksel Neden:**  
        Sezgisel / Metasezgisel algoritmalar ({solver_name}), başlangıç çözümündeki yumuşak ceza puanlarını düşürmeyi hedefler. 
        Mevcut personel sayısı ve vardiya talepleri **matematiksel olarak imkansız** olduğunda (örneğin 24 işçi varken günde 21 kişi istenmesi durumunda 7 günde 21 × 7 = 147 vardiya gerekirken, 6 gün çalışma kuralıyla 24 işçi en fazla 24 × 6 = 144 vardiya sağlayabilir - **Güvercin Yuvası İlkesi**), **ILP** matematiksel olarak *'Infeasible / Çözülemez'* derken; {solver_name} başlangıçtaki eksik kadrolu çizelge üzerinde arama yapıp yerel minimuma ulaşır ve sert kısıtları gideremez.
        """)
        if hard_violation_logs:
            with st.expander(f"📋 Tespit Edilen {hard_violations_count} Sert Kısıt İhlalinin Ayrıntılı Listesi", expanded=False):
                for log in hard_violation_logs[:50]:
                    st.markdown(f"- `{log}`")
                if len(hard_violation_logs) > 50:
                    st.caption(f"... ve {len(hard_violation_logs) - 50} adet daha ihlal.")


# ================================================================================
# 11. MERKEZİ CANLI OPTİMİZASYON İZLEME YÖNETİCİSİ (LIVE STREAM TRACKER)
# ================================================================================
import time
import plotly.graph_objects as go

class LiveStreamTracker:
    """
    Optimizasyon algoritmaları çalışırken ilerlemeyi, anlık adayı (Z_curr), en iyi skoru (Z_best),
    arama hızını ve yakınsama grafiğini canlı olarak Streamlit arayüzünde render eden merkezi yönetici.
    """
    def __init__(self, placeholder, title, max_steps, unit_name="İterasyon", initial_score=None, enabled=True, stream_interval=50, key_prefix=None):
        self.placeholder = placeholder
        self.title = title
        self.max_steps = max(1, max_steps)
        self.unit_name = unit_name
        self.initial_score = initial_score
        self.enabled = enabled
        self.stream_interval = max(1, stream_interval)
        self.start_time = time.time()
        self.last_update_time = 0
        self.history_x = []
        self.history_y = []
        self.history_curr_y = []
        self.best_score = initial_score if initial_score is not None else float('inf')
        self.curr_score = initial_score if initial_score is not None else float('inf')
        self.key_prefix = key_prefix or f"stream_{abs(hash(title)) % 1000000}"
        
    def should_update(self, step, current_best):
        if not self.enabled:
            return False
        # 1. Yeni rekor bulunduysa anında güncelle
        if current_best is not None and current_best < self.best_score:
            return True
        # 2. Periyodik adım aralığına gelindiyse
        if step % self.stream_interval == 0:
            return True
        # 3. Son adım
        if step >= self.max_steps:
            return True
        return False
        
    def update(self, step, best_score, cur_score=None, extra_info=""):
        if not self.enabled:
            return
            
        now = time.time()
        # Throttling emniyeti: iki render arasında en az 40 ms olmalı (ancak son adım hariç)
        if (now - self.last_update_time < 0.04) and (step < self.max_steps) and (best_score >= self.best_score):
            return
            
        self.last_update_time = now
        self.best_score = min(self.best_score, best_score)
        self.curr_score = cur_score if cur_score is not None else best_score
        if self.initial_score is None:
            self.initial_score = self.curr_score
            
        self.history_x.append(step)
        self.history_y.append(self.best_score)
        self.history_curr_y.append(self.curr_score)
        
        elapsed = max(0.001, now - self.start_time)
        speed = round(step / elapsed, 1) if step > 0 else 0
        pct = min(1.0, max(0.0, step / self.max_steps))
        score_drop = max(0, self.initial_score - self.best_score)
        pct_improve = round((score_drop / max(1, self.initial_score)) * 100, 1) if self.initial_score > 0 else 0
        
        with self.placeholder.container():
            st.markdown(f"#### ⚡ {self.title} (Canlı Optimizasyon Akışı)")
            
            prog_text = f"🔄 %{int(pct*100)} Tamamlandı | {self.unit_name}: {step:,} / {self.max_steps:,} | Arama Hızı: ~{speed:,.0f} {self.unit_name}/sn {extra_info}"
            st.progress(pct, text=prog_text)
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Başlangıç Skoru</div>
                <div class="metric-value" style="color: #64748b;">{self.initial_score:,.0f}</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Anlık Aday Skor</div>
                <div class="metric-value" style="color: #2563eb;">{self.curr_score:,.0f} ⚡</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Anlık En İyi Skor</div>
                <div class="metric-value" style="color: #059669;">{self.best_score:,.0f} ⬇️</div>
                </div>""", unsafe_allow_html=True)
            with c4:
                st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Toplam İyileşme</div>
                <div class="metric-value" style="color: #7c3aed;">-{score_drop:,.0f} (%{pct_improve})</div>
                </div>""", unsafe_allow_html=True)
                
            if len(self.history_x) > 1:
                df_live = pd.DataFrame({
                    "En İyi Skor (Z_best)": self.history_y,
                    "Anlık Aday Skor (Z_curr)": self.history_curr_y
                }, index=self.history_x)
                st.line_chart(df_live, height=220)

    def finish(self, final_step=None, final_score=None):
        """Arama tamamlandığında canlı akış panelini %100 tamamlandı olarak ekranda kalıcı tutar."""
        if not self.enabled:
            return
            
        step = final_step if final_step is not None else self.max_steps
        best_score = final_score if final_score is not None else self.best_score
        
        now = time.time()
        elapsed = max(0.001, now - self.start_time)
        speed = round(step / elapsed, 1) if step > 0 else 0
        score_drop = max(0, self.initial_score - best_score)
        pct_improve = round((score_drop / max(1, self.initial_score)) * 100, 1) if (self.initial_score and self.initial_score > 0) else 0
        
        if not self.history_x or step != self.history_x[-1]:
            self.history_x.append(step)
            self.history_y.append(best_score)
            self.history_curr_y.append(best_score)
            
        with self.placeholder.container():
            st.markdown(f"#### ⚡ {self.title} (Canlı Optimizasyon Akışı & Özet)")
            
            prog_text = f"✅ %100 Tamamlandı | {self.unit_name}: {step:,} / {self.max_steps:,} | Toplam Süre: {elapsed:.2f} sn (Ort. {speed:,.0f} {self.unit_name}/sn) | ✅ Arama Bitti"
            st.progress(1.0, text=prog_text)
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Başlangıç Skoru</div>
                <div class="metric-value" style="color: #64748b;">{self.initial_score:,.0f}</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Son Aday Skor</div>
                <div class="metric-value" style="color: #2563eb;">{self.curr_score:,.0f}</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Nihai En İyi Skor</div>
                <div class="metric-value" style="color: #059669;">{best_score:,.0f} 🎯</div>
                </div>""", unsafe_allow_html=True)
            with c4:
                st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Toplam İyileşme</div>
                <div class="metric-value" style="color: #7c3aed;">-{score_drop:,.0f} (%{pct_improve})</div>
                </div>""", unsafe_allow_html=True)
                
            if len(self.history_x) > 1:
                fig = go.Figure()
                if len(self.history_curr_y) == len(self.history_x):
                    fig.add_trace(go.Scatter(
                        x=self.history_x,
                        y=self.history_curr_y,
                        mode="lines",
                        name="Anlık Aday Skor (Z_curr)",
                        line=dict(color="#94a3b8", width=1.2, dash="dot")
                    ))
                fig.add_trace(go.Scatter(
                    x=self.history_x,
                    y=self.history_y,
                    mode="lines",
                    name="En İyi Ceza Skoru (Z_best)",
                    line=dict(color="#059669", width=2.5)
                ))
                fig.update_layout(
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#f8fafc",
                    height=240,
                    xaxis_title=f"Arama Adımı ({self.unit_name})",
                    yaxis_title="Ceza Skoru (Z)",
                    margin=dict(l=30, r=30, t=20, b=30),
                    legend=dict(orientation="h", y=1.15)
                )
                st.plotly_chart(fig, width="stretch", key=f"{self.key_prefix}_chart_finish")

    def get_stream_data(self):
        """Kalıcı saklama için canlı akış veri sözlüğünü döndürür."""
        elapsed = max(0.001, time.time() - self.start_time)
        step = self.history_x[-1] if self.history_x else self.max_steps
        return {
            'history_x': list(self.history_x),
            'history_y': list(self.history_y),
            'history_curr_y': list(self.history_curr_y),
            'initial_score': self.initial_score,
            'curr_score': self.curr_score,
            'best_score': self.best_score,
            'max_steps': self.max_steps,
            'unit_name': self.unit_name,
            'title': self.title,
            'elapsed_sec': round(elapsed, 2),
            'speed': round(step / elapsed, 1) if step > 0 else 0,
            'key_prefix': self.key_prefix
        }


def render_live_stream_summary(stream_data, key_prefix=None):
    """Kayıtlı canlı akış verilerini ekranda kalıcı olarak render eder."""
    if not stream_data or not stream_data.get('history_x'):
        return
    
    title = stream_data.get('title', 'Canlı Optimizasyon Akışı')
    unit_name = stream_data.get('unit_name', 'İterasyon')
    max_steps = stream_data.get('max_steps', 1)
    step = stream_data['history_x'][-1] if stream_data['history_x'] else max_steps
    init_score = stream_data.get('initial_score', 0)
    curr_score = stream_data.get('curr_score', stream_data.get('best_score', 0))
    best_score = stream_data.get('best_score', 0)
    elapsed = stream_data.get('elapsed_sec', 0)
    speed = stream_data.get('speed', 0)
    score_drop = max(0, init_score - best_score)
    pct_improve = round((score_drop / max(1, init_score)) * 100, 1) if (init_score and init_score > 0) else 0
    k_prefix = key_prefix or stream_data.get('key_prefix', f"live_summary_{abs(hash(title)) % 1000000}")

    st.markdown(f"#### ⚡ {title} (Canlı Optimizasyon Akışı & Özet)")
    
    prog_text = f"✅ %100 Tamamlandı | {unit_name}: {step:,} / {max_steps:,} | Toplam Süre: {elapsed:.2f} sn (Ort. {speed:,.0f} {unit_name}/sn) | ✅ Arama Bitti"
    st.progress(1.0, text=prog_text)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Başlangıç Skoru</div>
        <div class="metric-value" style="color: #64748b;">{init_score:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Son Aday Skor</div>
        <div class="metric-value" style="color: #2563eb;">{curr_score:,.0f}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Nihai En İyi Skor</div>
        <div class="metric-value" style="color: #059669;">{best_score:,.0f} 🎯</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Toplam İyileşme</div>
        <div class="metric-value" style="color: #7c3aed;">-{score_drop:,.0f} (%{pct_improve})</div>
        </div>""", unsafe_allow_html=True)
        
    if len(stream_data['history_x']) > 1:
        fig = go.Figure()
        if stream_data.get('history_curr_y') and len(stream_data['history_curr_y']) == len(stream_data['history_x']):
            fig.add_trace(go.Scatter(
                x=stream_data['history_x'],
                y=stream_data['history_curr_y'],
                mode="lines",
                name="Anlık Aday Skor (Z_curr)",
                line=dict(color="#94a3b8", width=1.2, dash="dot")
            ))
        fig.add_trace(go.Scatter(
            x=stream_data['history_x'],
            y=stream_data['history_y'],
            mode="lines",
            name="En İyi Ceza Skoru (Z_best)",
            line=dict(color="#059669", width=2.5)
        ))
        fig.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafc",
            height=240,
            xaxis_title=f"Arama Adımı ({unit_name})",
            yaxis_title="Ceza Skoru (Z)",
            margin=dict(l=30, r=30, t=20, b=30),
            legend=dict(orientation="h", y=1.15)
        )
        st.plotly_chart(fig, width="stretch", key=f"{k_prefix}_chart_summary")
