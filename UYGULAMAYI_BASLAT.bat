@echo off
chcp 65001 > nul
title Hemsire ve Celik Tesisleri Vardiya Cizelgeleme Uygulamasi
cd /d "%~dp0"
start http://localhost:8501

python -m streamlit run app.py --server.headless true
pause
