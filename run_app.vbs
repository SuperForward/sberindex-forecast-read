Set objShell = CreateObject("WScript.Shell")
objShell.CurrentDirectory = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
objShell.Environment("Process")("PYTHONUTF8") = "1"
objShell.Run """.venv\Scripts\pythonw.exe"" -m app.main", 0, False
