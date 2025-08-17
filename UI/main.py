#!/usr/bin/env python3
"""
Aplicación principal para el análisis de plantas de frijol con YOLO
Autor: Sistema de Visión Computacional
Fecha: Agosto 2025
"""

import sys
import os
from pathlib import Path

# Agregar el directorio padre al path para importar módulos
sys.path.insert(0, str(Path(__file__).parent))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QDir
from gui.main_window import MainWindow
from utils.config import Config

def setup_directories():
    """Configurar directorios necesarios para la aplicación"""
    base_dir = Path(__file__).parent.parent
    
    # Crear directorios necesarios si no existen
    directories = [
        base_dir / "content" / "runs" / "detect",
        base_dir / "content" / "temp_uploads",
        base_dir / "UI" / "resources" / "icons",
        base_dir / "UI" / "resources" / "styles"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def main():
    """Función principal de la aplicación"""
    # Configurar directorios
    setup_directories()
    
    # Crear aplicación PyQt6
    app = QApplication(sys.argv)
    app.setApplicationName("Análisis de Plantas de Frijol")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Visión Computacional")
    
    # Configurar estilo global con colores fijos para evitar problemas de modo oscuro
    app.setStyleSheet("""
        /* ESTILO GLOBAL - COLORES FIJOS PARA EVITAR MODO OSCURO */
        QApplication, QMainWindow, QWidget {
            background-color: #ffffff;
            color: #000000;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 10pt;
        }
        
        /* ETIQUETAS Y TEXTO */
        QLabel {
            color: #000000;
            background-color: transparent;
        }
        
        /* BOTONES */
        QPushButton {
            color: #000000;
            background-color: #f0f0f0;
            border: 1px solid #cccccc;
            border-radius: 4px;
            padding: 5px 10px;
        }
        QPushButton:hover {
            background-color: #e0e0e0;
        }
        QPushButton:pressed {
            background-color: #d0d0d0;
        }
        
        /* CAMPOS DE ENTRADA */
        QLineEdit, QTextEdit, QPlainTextEdit {
            color: #000000;
            background-color: #ffffff;
            border: 1px solid #cccccc;
            border-radius: 3px;
            padding: 3px;
        }
        
        /* COMBOS Y SPINS */
        QComboBox, QSpinBox, QDoubleSpinBox {
            color: #000000;
            background-color: #ffffff;
            border: 1px solid #cccccc;
            border-radius: 3px;
            padding: 2px 5px;
        }
        
        /* CHECKBOXES */
        QCheckBox {
            color: #000000;
        }
        
        /* GRUPOS */
        QGroupBox {
            color: #000000;
            border: 1px solid #cccccc;
            border-radius: 5px;
            margin-top: 10px;
            font-weight: bold;
        }
        QGroupBox::title {
            color: #000000;
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 5px;
        }
        
        /* TABS */
        QTabWidget::pane {
            border: 1px solid #cccccc;
            background-color: #ffffff;
        }
        QTabBar::tab {
            color: #000000;
            background-color: #f0f0f0;
            border: 1px solid #cccccc;
            padding: 5px 15px;
        }
        QTabBar::tab:selected {
            background-color: #ffffff;
            border-bottom: none;
        }
        
        /* BARRAS DE PROGRESO */
        QProgressBar {
            color: #000000;
            background-color: #f0f0f0;
            border: 1px solid #cccccc;
            border-radius: 3px;
            text-align: center;
        }
        
        /* MENÚS */
        QMenuBar {
            color: #000000;
            background-color: #f8f8f8;
            border-bottom: 1px solid #cccccc;
        }
        QMenuBar::item {
            color: #000000;
            background-color: transparent;
        }
        QMenuBar::item:selected {
            background-color: #e0e0e0;
        }
        QMenu {
            color: #000000;
            background-color: #ffffff;
            border: 1px solid #cccccc;
        }
        QMenu::item:selected {
            background-color: #e0e0e0;
        }
        
        /* BARRA DE STATUS */
        QStatusBar {
            color: #000000;
            background-color: #f8f8f8;
            border-top: 1px solid #cccccc;
        }
        
        /* FRAMES Y PANELES */
        QFrame {
            color: #000000;
            background-color: #ffffff;
        }
        
        /* SCROLL AREAS */
        QScrollArea {
            background-color: #ffffff;
        }
        QScrollBar {
            background-color: #f0f0f0;
        }
        
        /* TABLAS */
        QTableWidget {
            color: #000000;
            background-color: #ffffff;
            gridline-color: #cccccc;
        }
        QHeaderView::section {
            color: #000000;
            background-color: #f0f0f0;
            border: 1px solid #cccccc;
        }
    """)
    
    # Crear y mostrar ventana principal
    main_window = MainWindow()
    main_window.show()
    
    # Ejecutar aplicación
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
