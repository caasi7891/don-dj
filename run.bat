@echo off
cd /d "%~dp0"
rem Anaconda's Qt 5 plugin path breaks PySide6. The app resets this itself;
rem clear it here too so a parent Anaconda Prompt cannot put it back.
set QT_PLUGIN_PATH=
set QT_QPA_PLATFORM_PLUGIN_PATH=
py -3.9 -c "import PySide6,sounddevice,miniaudio,mutagen,numpy,scipy" 1>nul 2>nul
if errorlevel 1 (
  echo Installing components...
  py -3.9 -m pip install -r "%~dp0requirements.txt"
  if errorlevel 1 (
    echo Install failed.
    pause
    exit /b 1
  )
)
py -3.9 -m app
if errorlevel 1 (
  echo.
  echo don-dj stopped because of an error. See the message above.
  pause
)
