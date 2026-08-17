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


# 1. ÖN-ANALİZ MALİYET & TAHMİNİ SÜRE KARTI
def render_preanalysis_prediction_card(title, calls, call_cost_ms, desc_items=None):
    """
    Algoritmaların çalıştırma butonunun hemen üstünde yer alan
    tahmini ceza değerlendirme çağrısı ve birim maliyet çarpım panosu.
    """
    st.markdown(f"#### 🔍 {title}")
    
    total_est_sec = (calls * call_cost_ms) / 1000.0
    time_str = f"{total_est_sec:.2f} sn" if total_est_sec >= 1.0 else f"{calls * call_cost_ms:.1f} ms"
    
    m_col1, m_col2 = st.columns([1, 2])
    with m_col1:
        st.metric(
            label="Tahmini Ceza Değerlendirme",
            value=f"~{calls:,} çağrı",
            delta=f"{call_cost_ms:.2f} ms * {calls:,} = {time_str}",
            delta_color="off"
        )
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


# 7. TAM VARDİYA ÇİZELGESİ MATRİS TABLOSU
def render_schedule_matrix_table(schedule, workers, num_days, title="🗓️ Optimize Edilmiş Vardiya Çizelgesi"):
    """Tüm çalışanların gün bazlı vardiyalarını metinsel (Gündüz, Akşam, Gece, OFF) tablo olarak listeler."""
    st.markdown(f"### {title}")
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
def render_request_details_expander(request_details, title="📋 Kişisel İzin Talepleri ve Karşılanma Durumu Detay Raporu"):
    """Çözücünün ürettiği request_details listesini genişletilebilir panel içinde biçimlendirerek gösterir."""
    with st.expander(title, expanded=False):
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


# 9. ÇALIŞAN KADRO PROFİLLERİ VE NİTELİKLERİ TABLOSU
def render_worker_profiles_table(workers, title="👥 Çalışan Profilleri ve Nitelikleri"):
    """Çalışanların postalarını, kıdem unvanlarını, MYK sertifikalarını ve talep ettikleri izin günlerini listeler."""
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
