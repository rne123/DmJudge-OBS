# 디맥 판정 분석기(DmJudge) 설치 스크립트 — INSTALL.bat 이 실행합니다.
# 하는 일: 파이썬 찾기(없으면 winget으로 설치) → 이 폴더 안에 전용 공간(.venv) → 부품 설치 → 바탕화면 바로가기 → 실행
$ErrorActionPreference = 'Continue'
try { $Host.UI.RawUI.WindowTitle = 'DmJudge 설치' } catch {}
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $root

function Say([string]$m, [string]$c = 'White') { Write-Host $m -ForegroundColor $c }
function Fail([string]$m) {
    Say ''
    Say "설치 실패: $m" 'Red'
    Say ''
    Say '이 창을 사진으로 찍어서 GitHub의 Issues에 올려 주시면 도와드릴게요.' 'Yellow'
    Say '   https://github.com/rne123/DmJudge-OBS/issues' 'Yellow'
    Read-Host '엔터를 누르면 창이 닫혀요'
    exit 1
}

# 파이썬 3.10 이상인지 확인. 맞으면 "3.12" 같은 버전 글자, 아니면 $null
function Test-Py([string]$exe, [string[]]$pre) {
    try {
        # 따옴표 없이 작성 (Windows PowerShell 5.1이 프로그램에 넘길 때 큰따옴표를 깨뜨림)
        $code = 'import sys;v=sys.version_info;print(str(v[0])+chr(46)+str(v[1]) if v>=(3,10) else 0)'
        $v = & $exe @pre -c $code 2>$null
        if ($LASTEXITCODE -eq 0 -and "$v" -match '^\d+\.\d+$') { return "$v" }
    } catch {}
    return $null
}

function Find-Py {
    $c = @()
    $c += , @('py', @('-3'))
    $c += , @('python', @())
    $dirs = @("$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe", "$env:ProgramFiles\Python3*\python.exe")
    Get-ChildItem $dirs -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | ForEach-Object { $c += , @($_.FullName, @()) }
    foreach ($x in $c) {
        $v = Test-Py $x[0] $x[1]
        if ($v) { return @{ exe = $x[0]; pre = $x[1]; ver = $v } }
    }
    return $null
}

Say '=============================================' 'Cyan'
Say '   디맥 판정 분석기 (DmJudge) OBS판 설치' 'Cyan'
Say '=============================================' 'Cyan'
Say '처음 한 번은 5~10분쯤 걸려요. 창을 끄지 말고 기다려 주세요.'
Say ''

# ---------------------------------------------------------------- 1. 파이썬
$venvPy = Join-Path $root '.venv\Scripts\python.exe'
$venvPyw = Join-Path $root '.venv\Scripts\pythonw.exe'
if (-not (Test-Path -LiteralPath $venvPy)) {
    Say '[1/4] 파이썬 찾는 중...' 'Cyan'
    $py = Find-Py
    if (-not $py) {
        Say '      파이썬이 없어서 자동으로 설치할게요. (1~3분)' 'Yellow'
        if (Get-Command winget -ErrorAction SilentlyContinue) {
            winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
            $py = Find-Py
        }
        if (-not $py) {
            Start-Process 'https://www.python.org/downloads/'
            Fail ('파이썬을 자동으로 설치하지 못했어요.' + "`n" +
                  '   방금 열린 사이트에서 [Download Python] 을 눌러 받고 설치하세요.' + "`n" +
                  '   설치 첫 화면 맨 아래 "Add python.exe to PATH" 를 꼭 체크!' + "`n" +
                  '   설치가 끝나면 INSTALL.bat 을 다시 더블클릭하세요.')
        }
    }
    Say ("      찾았어요: Python " + $py.ver) 'Green'

    # ------------------------------------------------------------ 2. 전용 공간
    Say '[2/4] 전용 공간(.venv) 만드는 중...' 'Cyan'
    $pyExe = $py.exe; $pyPre = $py.pre
    & $pyExe @pyPre -m venv .venv
    if (-not (Test-Path -LiteralPath $venvPy)) { Fail '전용 공간(.venv)을 만들지 못했어요.' }
} else {
    Say '[1/4] 파이썬: 이미 준비돼 있어요' 'Green'
    Say '[2/4] 전용 공간: 이미 있어요' 'Green'
}

# ---------------------------------------------------------------- 3. 부품
Say '[3/4] 필요한 부품 받는 중... (처음엔 몇 분 걸려요)' 'Cyan'
& $venvPy -m pip install --disable-pip-version-check --upgrade pip
& $venvPy -m pip install --disable-pip-version-check -r requirements.txt
if ($LASTEXITCODE -ne 0) { Fail '부품을 받지 못했어요. 인터넷 연결을 확인하고 INSTALL.bat 을 다시 실행해 주세요.' }
& $venvPy -c 'import numpy, cv2, dxcam, PIL' 2>$null
if ($LASTEXITCODE -ne 0) { Fail '부품은 받았는데 불러오지 못했어요.' }
Say '      부품 준비 완료' 'Green'

# ---------------------------------------------------------------- 4. 바로가기
Say '[4/4] 바탕화면에 바로가기 만드는 중...' 'Cyan'
try {
    $desk = [Environment]::GetFolderPath('Desktop')
    $ws = New-Object -ComObject WScript.Shell
    $lnk = $ws.CreateShortcut((Join-Path $desk 'DmJudge OBS.lnk'))
    $lnk.TargetPath = $venvPyw
    $lnk.Arguments = '"' + (Join-Path $root 'main.py') + '"'
    $lnk.WorkingDirectory = $root
    $lnk.Description = '디맥 판정 분석기 (OBS판)'
    $lnk.Save()
    Say '      바탕화면에 "DmJudge OBS" 바로가기를 만들었어요' 'Green'
} catch {
    Say '      바로가기는 못 만들었어요. 이 폴더의 START.bat 으로 실행하면 돼요.' 'Yellow'
}

Say ''
Say '=============================================' 'Green'
Say '   설치 끝!' 'Green'
Say '   다음부터는 바탕화면의 "DmJudge OBS" 를 더블클릭하세요.' 'Green'
Say '   (설정과 기록은 이 폴더에 저장돼요. 폴더를 지우지 마세요)' 'Green'
Say '=============================================' 'Green'
Say ''
Read-Host '엔터를 누르면 분석기를 실행해요'
Start-Process -FilePath $venvPyw -ArgumentList ('"' + (Join-Path $root 'main.py') + '"') -WorkingDirectory $root
