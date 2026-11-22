' AI 去背景工具 启动器：使用虚拟环境 pythonw 启动，无 cmd 窗口
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
base = fso.GetParentFolderName(WScript.ScriptFullName)
sh.CurrentDirectory = base
sh.Run """" & base & "\venv\Scripts\pythonw.exe"" """ & base & "\main.py""", 0, False
