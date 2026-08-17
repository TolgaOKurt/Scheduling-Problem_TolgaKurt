@echo off
chcp 65001 > nul
title Hemsire ve Celik Tesisleri Vardiya Cizelgeleme Uygulamasi
cd /d "%~dp0"
python -m streamlit run app.py
pause
