"""
================================================================================
  CONFIG.PY - UYGULAMA YAPILANDIRMA VE ÖZEL CSS STİLLERİ
================================================================================
"""
import streamlit as st

def apply_custom_styles():
    """Uygulamada kullanılan özel Light Theme CSS stillerini enjekte eder."""
    st.markdown("""
        <style>
        /* Tüm uygulamanın arka plan rengini ve varsayılan metin rengini ayarlar */
        .stApp {
            background-color: #f8fafc; /* Açık gri/beyaz arka plan */
            color: #0f172a;            /* Koyu lacivert/siyah metin rengi */
        }
        
        /* Tüm başlık tiplerinin (h1-h6) renk ve font kalınlığını düzenler */
        h1, h2, h3, h4, h5, h6 {
            color: #0f172a !important;
            font-weight: 700 !important;
        }
        
        /* Paragraf, liste ve metin alanlarının okunabilirlik ayarları */
        p, li, span, div {
            color: #1e293b;
            font-size: 1.05rem;
            line-height: 1.65;
        }
        
        /* Sayfa tepesindeki Hero Karşılama Kartının Stili */
        .hero-container {
            background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
            border: 2px solid #2563eb; /* Mavi vurgu kenarlığı */
            border-radius: 16px;
            padding: 2.2rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px rgba(37, 99, 235, 0.1);
        }
        
        /* Hero Kartı İçindeki Ana Başlık Stili */
        .hero-title {
            font-size: 2.3rem;
            font-weight: 800;
            color: #1d4ed8;
            margin-bottom: 0.8rem;
        }
        
        /* Hero Kartı İçindeki Alt Açıklama Metni Stili */
        .hero-subtitle {
            color: #334155;
            font-size: 1.15rem;
            line-height: 1.6;
        }
        
        /* İçerik Kartlarının Genel Kutusu (Beyaz zemin, ince gri sınır) */
        .card-box {
            background-color: #ffffff;
            border-radius: 12px;
            border: 1px solid #cbd5e1;
            padding: 1.6rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }
        
        /* Kart İçi Başlık Stili */
        .card-title {
            color: #1e40af;
            font-size: 1.3rem;
            font-weight: 700;
            margin-bottom: 1rem;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 0.5rem;
        }

        /* Kart İçi Metin Stili */
        .card-text {
            color: #0f172a;
        }
        
        /* Sert Kısıt (Hard Constraint) Rozet Stili (Kırmızı) */
        .badge-hard {
            background-color: #dc2626;
            color: #ffffff;
            padding: 0.3rem 0.8rem;
            border-radius: 6px;
            font-size: 0.9rem;
            font-weight: 700;
            display: inline-block;
        }
        
        /* Yumuşak Kısıt (Soft Constraint) Rozet Stili (Mavi) */
        .badge-soft {
            background-color: #2563eb;
            color: #ffffff;
            padding: 0.3rem 0.8rem;
            border-radius: 6px;
            font-size: 0.9rem;
            font-weight: 700;
            display: inline-block;
        }

        /* Kısıt Maddeleri İçi Özel Liste Stili */
        .constraint-list-item {
            background-color: #ffffff;
            border-left: 4px solid #cbd5e1;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
            border-radius: 0 8px 8px 0;
            box-shadow: 0 2px 6px rgba(0,0,0,0.03);
        }

        /* Metrik Kartı Stili */
        .metric-card {
            background-color: #ffffff;
            border-radius: 10px;
            border: 1px solid #cbd5e1;
            padding: 1.2rem;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            color: #1d4ed8;
        }
        .metric-label {
            font-size: 0.95rem;
            color: #64748b;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)
