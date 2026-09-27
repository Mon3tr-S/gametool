@echo off
chcp 65001 > nul
cd /d "%~dp0"

if "%ANTHROPIC_API_KEY%"=="" (
  echo.
  echo   APIキーが設定されていません。
  echo   下に貼り付けて Enter を押してください。
  echo   ^(console.anthropic.com で取得できます^)
  echo.
  set /p ANTHROPIC_API_KEY="API KEY: "
)

echo.
echo   起動しています...
echo.
python -m uvicorn app:app --host 0.0.0.0 --port 8000
pause
