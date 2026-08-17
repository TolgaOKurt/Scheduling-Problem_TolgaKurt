Set WshShell = CreateObject("WScript.Shell")
Set env = WshShell.Environment("PROCESS")
env("PYTHONUTF8") = "1"
env("PYTHONIOENCODING") = "utf-8"

Set fso = CreateObject("Scripting.FileSystemObject")
strPath = fso.GetAbsolutePathName(".")

' 8501 portunun bos olup olmadigini kontrol et (0: bos, 1: dolu)
checkPort = WshShell.Run("python -c ""import socket, sys; s = socket.socket(); is_free = (s.connect_ex(('127.0.0.1', 8501)) != 0); s.close(); sys.exit(0 if is_free else 1)""", 0, True)

If checkPort = 0 Then
    ' Port 8501 bossa Streamlit sunucusunu baslat
    WshShell.Run "python -m streamlit run """ & strPath & "\app.py"" --server.port 8501", 0, False
Else
    ' Port 8501 doluysa baska porta sunucu acma, mevcut sayfayi tarayicida ac
    WshShell.Run "http://localhost:8501"
End If