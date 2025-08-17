"""
Script de instalación y configuración
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Instalar requisitos desde requirements.txt"""
    print("🔧 Instalando dependencias...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ Archivo requirements.txt no encontrado")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ Dependencias instaladas correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False

def setup_directories():
    """Configurar directorios necesarios"""
    print("📁 Configurando directorios...")
    
    base_dir = Path(__file__).parent.parent
    
    directories = [
        base_dir / "content" / "runs" / "detect",
        base_dir / "content" / "temp_uploads",
        base_dir / "UI" / "logs",
        base_dir / "UI" / "exports",
        base_dir / "UI" / "resources" / "icons",
        base_dir / "UI" / "resources" / "styles"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")
    
    print("✅ Directorios configurados")

def check_system():
    """Verificar sistema"""
    print("🔍 Verificando sistema...")
    
    # Verificar Python
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 o superior requerido")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Verificar CUDA (opcional)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA disponible - {torch.cuda.device_count()} GPU(s)")
        else:
            print("ℹ️ CUDA no disponible - se usará CPU")
    except ImportError:
        print("ℹ️ PyTorch no instalado aún")
    
    return True

def create_desktop_shortcut():
    """Crear acceso directo en el escritorio (Windows)"""
    if os.name != 'nt':
        return
    
    try:
        import winshell
        from win32com.client import Dispatch
        
        desktop = winshell.desktop()
        path = os.path.join(desktop, "Análisis de Plantas de Frijol.lnk")
        target = str(Path(__file__).parent / "main.py")
        wDir = str(Path(__file__).parent)
        
        shell = Dispatch('WScript.Shell')
        shortcut = shell.CreateShortCut(path)
        shortcut.Targetpath = target
        shortcut.WorkingDirectory = wDir
        shortcut.IconLocation = target
        shortcut.save()
        
        print("✅ Acceso directo creado en el escritorio")
    except:
        print("ℹ️ No se pudo crear acceso directo en el escritorio")

def main():
    """Función principal de instalación"""
    print("🌱 Configurando Análisis de Plantas de Frijol")
    print("=" * 50)
    
    # Verificar sistema
    if not check_system():
        return False
    
    # Configurar directorios
    setup_directories()
    
    # Instalar dependencias
    if not install_requirements():
        return False
    
    # Crear acceso directo (opcional)
    create_desktop_shortcut()
    
    print("\n✅ ¡Instalación completada!")
    print("\nPara ejecutar la aplicación:")
    print("1. Abrir terminal en la carpeta UI")
    print("2. Ejecutar: python main.py")
    print("\nO usar el acceso directo del escritorio (si se creó)")
    
    return True

if __name__ == "__main__":
    success = main()
    
    if not success:
        print("\n❌ Instalación fallida")
        input("Presiona Enter para salir...")
        sys.exit(1)
    else:
        print("\n¿Deseas ejecutar la aplicación ahora? (y/n): ", end="")
        response = input().lower()
        if response in ['y', 'yes', 's', 'si']:
            print("\n🚀 Iniciando aplicación...")
            try:
                from main import main as run_app
                run_app()
            except Exception as e:
                print(f"❌ Error ejecutando aplicación: {e}")
                input("Presiona Enter para salir...")
