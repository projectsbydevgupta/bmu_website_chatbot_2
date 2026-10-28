@echo off
title BMU Chat Assistant - Launcher
color 0A



echo ============================================
echo    BMU Chat Assistant - Starting Up...
echo ============================================
echo.

:: Navigate to the project directory
cd /d "%~dp0"

:: Auto-detect proper Python interpreter (prioritizing Python 3.10 with installed packages)
set "PYTHON_CMD="
py -c "import chromadb" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PYTHON_CMD=py"
    set "PIP_CMD=py -m pip"
) else (
    python -c "import chromadb" >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        set "PYTHON_CMD=python"
        set "PIP_CMD=python -m pip"
    ) else (
        if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
            set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
            set "PIP_CMD=%LOCALAPPDATA%\Programs\Python\Python310\python.exe -m pip"
        ) else (
            set "PYTHON_CMD=python"
            set "PIP_CMD=pip"
        )
    )
)

:: Install dependencies
echo [1/3] Checking and installing dependencies...
%PIP_CMD% install -r requirements.txt --quiet 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install dependencies. Please check your Python installation.
    pause
    exit /b 1
)
echo       Done!
echo.

:: Kill any existing processes on ports 8000, 8001, 8002, and 6006
echo [2/4] Starting services and observability...
echo       Clearing ports 8000, 8001, 8002, and 6006...
powershell -Command "Get-NetTCPConnection -LocalPort 8000,8001,8002,6006 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }" 2>nul
timeout /t 2 /nobreak >nul

:: Start Arize Phoenix Tracing Server on port 6006
echo       Starting Phoenix Observability Server on port 6006...
start "BMU-Phoenix-Tracing" /min cmd /c "%PYTHON_CMD% phoenix_service.py"

:: Start the Retrieval Microservice on port 8001
echo       Starting Retrieval Microservice on port 8001...
start "BMU-Retrieval-Microservice" /min cmd /c "%PYTHON_CMD% retrieval_service.py"

:: Start the Ingestion Microservice on port 8002
echo       Starting Ingestion Microservice on port 8002...
start "BMU-Ingestion-Microservice" /min cmd /c "%PYTHON_CMD% ingestion_service.py"

:: Start the FastAPI backend server on port 8000
echo       Starting Backend Orchestrator on port 8000...
start "BMU-Backend" /min cmd /c "%PYTHON_CMD% semantic.py"

:: Wait for servers to boot up
echo       Waiting for microservices to initialize...
set RETRIES=0
:healthcheck
timeout /t 3 /nobreak >nul
set /a RETRIES+=1
curl -s -o nul -w "" http://127.0.0.1:8001/health 2>nul
set RETRIEVAL_OK=%ERRORLEVEL%
curl -s -o nul -w "" http://127.0.0.1:8002/health 2>nul
set INGESTION_OK=%ERRORLEVEL%
curl -s -o nul -w "" http://127.0.0.1:8000/health 2>nul
set BACKEND_OK=%ERRORLEVEL%

if %RETRIEVAL_OK% EQU 0 if %INGESTION_OK% EQU 0 if %BACKEND_OK% EQU 0 goto servers_ready
if %RETRIES% GEQ 12 (
    echo [ERROR] One or more microservices failed to start after 36 seconds.
    echo         Check your .env file and dependencies.
    pause
    exit /b 1
)
echo       Still loading... (attempt %RETRIES%/12)
goto healthcheck

:servers_ready
echo       Phoenix Observability running at http://127.0.0.1:6006
echo       Retrieval Microservice running at http://127.0.0.1:8001
echo       Ingestion Microservice running at http://127.0.0.1:8002
echo       Backend Orchestrator running at http://127.0.0.1:8000
echo.

:: Open the frontend in the default browser
echo [3/4] Opening frontend in browser...
start "" "frontend\index.html"
echo       Frontend opened!
echo.

echo ============================================
echo    BMU Chat Assistant is ready!
echo ============================================
echo.
echo    Backend API:        http://127.0.0.1:8000
echo    Backend Docs:       http://127.0.0.1:8000/docs
echo    Retrieval Service:  http://127.0.0.1:8001
echo    Retrieval Docs:     http://127.0.0.1:8001/docs
echo    Ingestion Service:  http://127.0.0.1:8002
echo    Ingestion Docs:     http://127.0.0.1:8002/docs
echo    Phoenix Tracing:    http://127.0.0.1:6006
echo    Frontend:           frontend\index.html
echo.
echo    Press any key to STOP all services...
echo ============================================
pause >nul

:: Shutdown: kill all services
echo.
echo Shutting down services...
powershell -Command "Get-NetTCPConnection -LocalPort 8000,8001,8002,6006 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }" 2>nul
echo Done. Goodbye!
timeout /t 2 /nobreak >nul
