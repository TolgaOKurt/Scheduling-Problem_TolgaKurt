@echo off
setlocal enabledelayedexpansion
chcp 65001 > nul
title Streamlit Sunucu Durdurucu (8501-8504)

echo ====================================================
echo 8501 - 8504 portlarindaki sunucular sonlandiriliyor...
echo ====================================================
echo.

for /L %%P in (8501,1,8504) do (
    set "found="
    for /f "tokens=5" %%A in ('netstat -ano ^| findstr /r /c:":%%P " ^| findstr LISTENING 2^>nul') do (
        echo [Port %%P] calisan PID: %%A bulundu. Kapatiliyor...
        taskkill /F /PID %%A >nul 2>&1
        set "found=1"
    )
    if "!found!"=="" (
        echo [Port %%P] bosta ^(calisan sunucu yok^).
    )
)

echo.
echo ====================================================
echo Islem tamamlandi.
echo ====================================================
pause