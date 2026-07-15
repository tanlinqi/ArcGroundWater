@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "ARC_ROOT=%~dp0"
set "ENV_DIR=%ARC_ROOT%envs\ssin"
set "YML_FILE=%ARC_ROOT%environment-gpu.yml"
set "TOOLS_DIR=%ARC_ROOT%tools"
set "MINICONDA_DIR=%TOOLS_DIR%\miniconda"
set "INSTALLER_DIR=%TOOLS_DIR%\installers"
set "INSTALLER=%INSTALLER_DIR%\Miniconda3-latest-Windows-x86_64.exe"
set "MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"

if not exist "%YML_FILE%" (
    echo ERROR: environment file was not found: %YML_FILE%
    exit /b 1
)

set "CONDA_EXE="
if exist "%MINICONDA_DIR%\Scripts\conda.exe" (
    set "CONDA_EXE=%MINICONDA_DIR%\Scripts\conda.exe"
) else if exist "%USERPROFILE%\Miniconda3\Scripts\conda.exe" (
    set "CONDA_EXE=%USERPROFILE%\Miniconda3\Scripts\conda.exe"
) else if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" (
    set "CONDA_EXE=%USERPROFILE%\anaconda3\Scripts\conda.exe"
) else if exist "%LOCALAPPDATA%\miniconda3\Scripts\conda.exe" (
    set "CONDA_EXE=%LOCALAPPDATA%\miniconda3\Scripts\conda.exe"
) else (
    for /f "usebackq delims=" %%C in (`where conda 2^>nul`) do (
        if not defined CONDA_EXE set "CONDA_EXE=%%C"
    )
)

if not defined CONDA_EXE (
    echo Conda was not found. Downloading Miniconda...
    if not exist "%INSTALLER_DIR%" mkdir "%INSTALLER_DIR%"
    powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri '%MINICONDA_URL%' -OutFile '%INSTALLER%'"
    if errorlevel 1 (
        echo ERROR: failed to download Miniconda.
        exit /b 1
    )

    echo Installing Miniconda to %MINICONDA_DIR% ...
    if not exist "%TOOLS_DIR%" mkdir "%TOOLS_DIR%"
    start /wait "" "%INSTALLER%" /InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /S /D=%MINICONDA_DIR%
    if errorlevel 1 (
        echo ERROR: failed to install Miniconda.
        exit /b 1
    )
    set "CONDA_EXE=%MINICONDA_DIR%\Scripts\conda.exe"
)

if not exist "%CONDA_EXE%" (
    echo ERROR: conda executable was not found: %CONDA_EXE%
    exit /b 1
)

echo Using conda: %CONDA_EXE%
call "%CONDA_EXE%" --version
if errorlevel 1 exit /b 1

if exist "%ENV_DIR%\python.exe" (
    echo Updating ssin environment: %ENV_DIR%
    call "%CONDA_EXE%" env update -p "%ENV_DIR%" -f "%YML_FILE%" --prune -y
) else (
    echo Creating ssin environment: %ENV_DIR%
    call "%CONDA_EXE%" env create -p "%ENV_DIR%" -f "%YML_FILE%" -y
)
if errorlevel 1 (
    echo ERROR: failed to create or update ssin environment.
    exit /b 1
)

echo Verifying ssin environment...
"%ENV_DIR%\python.exe" -c "import sys, numpy, pandas, sklearn, statsmodels, torch, torch_geometric, pyproj, geographiclib; print('Python:', sys.executable); print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
if errorlevel 1 (
    echo ERROR: ssin environment verification failed.
    exit /b 1
)

echo Done. Python executable: %ENV_DIR%\python.exe
exit /b 0
