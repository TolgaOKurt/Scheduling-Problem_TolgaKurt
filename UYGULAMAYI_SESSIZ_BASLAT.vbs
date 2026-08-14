Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
strPath = fso.GetAbsolutePathName(".")

' Streamlit uygulamasını siyah konsol penceresi olmadan arka planda çalıştırır
WshShell.Run "python -m streamlit run """ & strPath & "\app.py"" --server.headless true", 0, False

' 2 saniye bekleyip tarayıcıyı açar
WScript.Sleep 2000
WshShell.Run "http://localhost:8501"
