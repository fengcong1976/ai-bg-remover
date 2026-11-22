; AI 去背景工具 安装包脚本 (Inno Setup 6)
[Setup]
LicenseFile=LICENSE
RedirectionGuard=no
AppId={{7E1F3C2A-9B4D-4E8A-A6C3-5F2D8B901234}
AppName=AI去背景工具
AppVersion=2.5
AppVerName=AI去背景工具 2.5
AppPublisher=深圳市创炫互动网络科技有限公司
AppPublisherURL=https://www.mrheedu.com
AppSupportURL=https://www.mrheedu.com
DefaultDirName={localappdata}\Programs\AI去背景工具
DefaultGroupName=AI去背景工具
DisableProgramGroupPage=yes
OutputDir=installer_output
OutputBaseFilename=AI去背景工具_v2.5_安装包
SetupIconFile=图标.ico
UninstallDisplayIcon={app}\AI去背景工具.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "dist\AI去背景工具\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion createallsubdirs

[Icons]
Name: "{group}\AI去背景工具"; Filename: "{app}\AI去背景工具.exe"
Name: "{group}\卸载 AI去背景工具"; Filename: "{uninstallexe}"
Name: "{autodesktop}\AI去背景工具"; Filename: "{app}\AI去背景工具.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\AI去背景工具.exe"; Description: "{cm:LaunchProgram,AI去背景工具}"; Flags: nowait postinstall skipifsilent
