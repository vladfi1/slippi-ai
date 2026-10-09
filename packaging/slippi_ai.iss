; Inno Setup script for the Windows installer, around the PyInstaller bundle.
;
;   pyinstaller --noconfirm packaging/slippi_ai.spec
;   iscc /DAppVersion=0.3.0 packaging/slippi_ai.iss
;
; The version is the app's, from slippi_ai/gui/version.py, not slippi-ai's.
;
; Produces dist/phillip-setup-<version>.exe. It installs per user by
; default (no admin), adds a Start menu shortcut and, unless it's already
; installed, downloads and installs the Windows App SDK Runtime that Windows
; ML needs to run models on the GPU (see slippi_ai/winml.py). Without the
; runtime the app runs models on the CPU.
;
; The app updates itself by running a newer installer with /SILENT
; /relaunch=1 and exiting (slippi_ai/gui/updates.py): setup waits for the
; app's mutex to be released, upgrades in place and starts the app again.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

; Overridden only to test installs without touching the real one.
#ifndef AppId
  #define AppId "{{725F5556-F82C-491E-918C-7A7B10A0F195}"
#endif

; Held by the running app; must match APP_MUTEX in slippi_ai/gui/updates.py.
#define AppMutexName "phillip-launcher"

; Must match the wasdk packages in setup.cfg's winml extra, and the version
; checked for in CI (.github/workflows/bundle.yml).
#ifndef RuntimeVersion
  #define RuntimeVersion "2.3.1.0"
#endif
#define RuntimeUrl "https://aka.ms/windowsappsdk/2.3/2.3.1/windowsappruntimeinstall-x64.exe"
#define RuntimeSha256 "4011748ddf472b7e856d909fdfb4e9b19c3d23fcd8121039ac91f99d5ffa65db"
#define RuntimeInstaller "windowsappruntimeinstall-x64.exe"

[Setup]
AppId={#AppId}
AppName=phillip
AppVersion={#AppVersion}
AppPublisher=Vlad Firoiu
AppPublisherURL=https://github.com/vladfi1/slippi-ai
AppSupportURL=https://github.com/vladfi1/slippi-ai/issues
DefaultDirName={autopf}\phillip
DefaultGroupName=phillip
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Windows 10 1809, the oldest the Windows App SDK supports.
MinVersion=10.0.17763
OutputDir=..\dist
OutputBaseFilename=phillip-setup-{#AppVersion}
UninstallDisplayIcon={app}\phillip.exe
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern

[Tasks]
Name: desktopicon; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: gpu; Description: "Run phillip on the GPU with Windows ML (downloads Microsoft's Windows App SDK Runtime, 113 MB)"; Check: RuntimeNeeded

[InstallDelete]
; Files of the previous version that this one no longer has.
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "..\dist\phillip\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\phillip"; Filename: "{app}\phillip.exe"
Name: "{autodesktop}\phillip"; Filename: "{app}\phillip.exe"; Tasks: desktopicon

[Run]
Filename: "{tmp}\{#RuntimeInstaller}"; Parameters: "--quiet"; StatusMsg: "Installing the Windows App SDK Runtime..."; Flags: runhidden waituntilterminated; Check: RuntimeDownloaded
Filename: "{app}\phillip.exe"; Description: "{cm:LaunchProgram,phillip}"; Flags: nowait postinstall skipifsilent
; Restarts the app after an update from inside it.
Filename: "{app}\phillip.exe"; Flags: nowait; Check: Relaunch

[UninstallDelete]
; Downloaded models, the saved model index, downloaded updates and
; TensorRT-RTX's compiled kernels. Settings in {userappdata} are kept.
Type: filesandordirs; Name: "{localappdata}\phillip"

[Code]
var
  DownloadPage: TDownloadWizardPage;
  RuntimeChecked, RuntimeMissing, Downloaded: Boolean;

// Whether the Windows App SDK Runtime's Main package and framework, at
// version {#RuntimeVersion} or later, are installed for this user.
function RuntimeInstalled: Boolean;
var
  ResultCode: Integer;
  Command: String;
begin
  Command :=
    '$v = [version]''{#RuntimeVersion}''; ' +
    'function Has($name) { @(Get-AppxPackage $name | Where-Object ' +
    '{ $_.Architecture -eq ''X64'' -and [version]$_.Version -ge $v }).Count -gt 0 }; ' +
    'if ((Has ''MicrosoftCorporationII.WinAppRuntime.Main.2'') -and ' +
    '(Has ''Microsoft.WindowsAppRuntime.2'')) { exit 0 } else { exit 1 }';
  Result := Exec('powershell.exe', '-NoProfile -NonInteractive -Command "' + Command + '"',
    '', SW_HIDE, ewWaitUntilTerminated, ResultCode) and (ResultCode = 0);
  Log(Format('Windows App SDK Runtime {#RuntimeVersion} installed: %d', [Ord(Result)]));
end;

function RuntimeNeeded: Boolean;
begin
  if not RuntimeChecked then
  begin
    RuntimeMissing := not RuntimeInstalled;
    RuntimeChecked := True;
  end;
  Result := RuntimeMissing;
end;

function RuntimeDownloaded: Boolean;
begin
  Result := Downloaded;
end;

function Relaunch: Boolean;
begin
  Result := WizardSilent and (ExpandConstant('{param:relaunch|0}') = '1');
end;

// Waits for the app to exit: when it updates itself it starts setup, then
// exits. If it's still running, asks the user to close it.
function InitializeSetup: Boolean;
var
  Waited: Integer;
begin
  Result := True;
  Waited := 0;
  while CheckForMutexes('{#AppMutexName}') and (Waited < 30000) do
  begin
    Sleep(250);
    Waited := Waited + 250;
  end;
  while CheckForMutexes('{#AppMutexName}') do
  begin
    if SuppressibleMsgBox('phillip is running. Close it, then click OK to continue.',
        mbError, MB_OKCANCEL, IDCANCEL) = IDCANCEL then
    begin
      Result := False;
      Exit;
    end;
  end;
end;

procedure InitializeWizard;
begin
  DownloadPage := CreateDownloadPage(SetupMessage(msgWizardPreparing),
    SetupMessage(msgPreparingDesc), nil);
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if (CurPageID <> wpReady) or not RuntimeNeeded or not WizardIsTaskSelected('gpu') then
    Exit;
  DownloadPage.Clear;
  DownloadPage.Add('{#RuntimeUrl}', '{#RuntimeInstaller}', '{#RuntimeSha256}');
  DownloadPage.Show;
  try
    try
      DownloadPage.Download;
      Downloaded := True;
    except
      // Not fatal: the app runs models on the CPU without the runtime.
      if not DownloadPage.AbortedByUser then
        SuppressibleMsgBox('Could not download the Windows App SDK Runtime, so ' +
          'phillip will run on the CPU: ' + GetExceptionMessage, mbError, MB_OK, IDOK);
    end;
  finally
    DownloadPage.Hide;
  end;
end;
