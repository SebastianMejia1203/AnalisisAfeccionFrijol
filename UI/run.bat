@echo off
echo 🌱 Iniciando Sistema de Análisis de Plantas de Frijol
echo ================================================

cd /d "%~dp0"

REM Verificar si Python está instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python no está instalado o no está en el PATH
    echo.
    echo Por favor instala Python 3.8 o superior desde:
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Verificar si es la primera ejecución
if not exist "logs" (
    echo 🔧 Primera ejecución detectada. Ejecutando configuración...
    python setup.py
    if errorlevel 1 (
        echo ❌ Error durante la configuración
        pause
        exit /b 1
    )
)

REM Ejecutar la aplicación
echo 🚀 Iniciando aplicación...
python main.py

if errorlevel 1 (
    echo.
    echo ❌ La aplicación terminó con errores
    echo Revisa los logs en la carpeta 'logs' para más detalles
    pause
)

echo.
echo ✅ Aplicación cerrada correctamente
pause
