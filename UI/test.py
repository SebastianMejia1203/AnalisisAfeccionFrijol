#!/usr/bin/env python3
"""
Script de prueba para validar la instalación
"""

import sys
import os
from pathlib import Path

# Agregar el directorio actual al path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Probar importaciones críticas"""
    print("🔍 Probando importaciones...")
    
    try:
        from PyQt6.QtWidgets import QApplication
        print("✅ PyQt6 importado correctamente")
    except ImportError as e:
        print(f"❌ Error importando PyQt6: {e}")
        return False
    
    try:
        import cv2
        print("✅ OpenCV importado correctamente")
    except ImportError as e:
        print(f"❌ Error importando OpenCV: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy importado correctamente")
    except ImportError as e:
        print(f"❌ Error importando NumPy: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics importado correctamente")
    except ImportError as e:
        print(f"❌ Error importando Ultralytics: {e}")
        return False
    
    return True

def test_config():
    """Probar configuración"""
    print("⚙️ Probando configuración...")
    
    try:
        from utils.config import config
        print(f"✅ Configuración cargada")
        print(f"  - Dataset dir: {config.DATASET_DIR}")
        print(f"  - Content dir: {config.CONTENT_DIR}")
        print(f"  - Test images dir: {config.TEST_IMAGES_DIR}")
        return True
    except Exception as e:
        print(f"❌ Error en configuración: {e}")
        return False

def test_validators():
    """Probar validadores"""
    print("🔧 Probando validadores...")
    
    try:
        from utils.validators import SystemValidator
        validation_results = SystemValidator.validate_environment()
        
        print("Resultados de validación:")
        for key, value in validation_results.items():
            status = "✅" if value else "❌"
            print(f"  {status} {key}: {value}")
        
        return True
    except Exception as e:
        print(f"❌ Error en validadores: {e}")
        return False

def test_gui_creation():
    """Probar creación básica de GUI"""
    print("🖼️ Probando creación de GUI...")
    
    try:
        from PyQt6.QtWidgets import QApplication, QLabel
        
        app = QApplication([])
        label = QLabel("Test GUI")
        print("✅ GUI básica creada correctamente")
        
        # No mostramos la ventana, solo probamos la creación
        app.quit()
        return True
    except Exception as e:
        print(f"❌ Error creando GUI: {e}")
        return False

def main():
    """Función principal de prueba"""
    print("🌱 Prueba del Sistema de Análisis de Plantas de Frijol")
    print("=" * 60)
    
    tests = [
        ("Importaciones", test_imports),
        ("Configuración", test_config), 
        ("Validadores", test_validators),
        ("GUI", test_gui_creation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Ejecutando prueba: {test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                print(f"✅ Prueba {test_name} PASADA")
                passed += 1
            else:
                print(f"❌ Prueba {test_name} FALLIDA")
        except Exception as e:
            print(f"❌ Prueba {test_name} FALLIDA con excepción: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Resumen: {passed}/{total} pruebas pasadas")
    
    if passed == total:
        print("🎉 ¡Todas las pruebas pasaron! El sistema está listo.")
        print("\nPuedes ejecutar la aplicación con:")
        print("  python main.py")
        print("  o")
        print("  run.bat")
        return True
    else:
        print("⚠️ Algunas pruebas fallaron. Revisa la instalación.")
        return False

if __name__ == "__main__":
    success = main()
    
    print(f"\n¿Deseas ejecutar la aplicación completa ahora? (y/n): ", end="")
    response = input().lower()
    
    if response in ['y', 'yes', 's', 'si'] and success:
        print("\n🚀 Iniciando aplicación completa...")
        try:
            os.system('"E:/Python/Vision Computacional/.venv/Scripts/python.exe" main.py')
        except Exception as e:
            print(f"❌ Error ejecutando aplicación: {e}")
    
    input("\nPresiona Enter para salir...")
