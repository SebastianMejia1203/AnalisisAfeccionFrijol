"""
Ventana principal de la aplicación
"""

from PyQt6.QtWidgets import (QMainWindow, QTabWidget, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QFrame)
from PyQt6.QtGui import QFont

from gui.train_tab import TrainTab
from gui.predict_tab import PredictTab
from gui.analysis_tab import AnalysisTab
from gui.results_tab import ResultsTab

class MainWindow(QMainWindow):
    """Ventana principal de la aplicación"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Configurar la interfaz de usuario"""
        self.setWindowTitle("Análisis de Plantas de Frijol - YOLO v9")
        
        # CONFIGURACIÓN DE VENTANA - Ajustada para pantallas pequeñas
        # Obtener dimensiones de la pantalla
        screen_geometry = self.screen().availableGeometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()
        
        # Calcular tamaño de ventana (90% de la pantalla disponible)
        window_width = int(screen_width * 0.9)
        window_height = int(screen_height * 0.9)
        
        # Establecer tamaño mínimo más pequeño para pantallas de portátil
        self.setMinimumSize(1200, 700)
        
        # Centrar ventana en pantalla
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.setGeometry(x, y, window_width, window_height)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal con márgenes reducidos para pantallas pequeñas
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)  # Márgenes reducidos
        main_layout.setSpacing(5)  # Espaciado reducido
        
        # HEADER - Sección superior con información del proyecto
        self.create_header(main_layout)
        
        # TABS PRINCIPALES - Área principal de trabajo
        self.create_tabs(main_layout)
        
        # APLICAR ESTILOS - Configuración visual
        self.apply_styles()
    
    def create_header(self, parent_layout):
        """
        HEADER - Crear encabezado de la aplicación
        EDITAR AQUÍ: Título, subtítulo, colores del header
        """
        header_frame = QFrame()
        header_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        header_frame.setMaximumHeight(60)  # Altura reducida para pantallas pequeñas
        
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(10, 5, 10, 5)  # Márgenes reducidos
        
        # TÍTULO PRINCIPAL - Texto principal del header
        title_label = QLabel("🌱 Sistema de detección y análisis de enfermedades usando YOLO v9")
        title_font = QFont()
        title_font.setPointSize(14)  # Tamaño reducido para pantallas pequeñas
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #2E7D32; padding: 2px;")  # EDITAR COLOR AQUÍ
        
        # INFORMACIÓN DEL PROYECTO - Subtítulo descriptivo
        
        # LAYOUT DEL HEADER - Organización vertical del texto
        header_text_layout = QVBoxLayout()
        header_text_layout.setSpacing(2)  # Espaciado reducido
        header_text_layout.addWidget(title_label)
        
        header_layout.addLayout(header_text_layout)
        header_layout.addStretch()  # Empuja contenido a la izquierda
        
        parent_layout.addWidget(header_frame)
    
    def create_tabs(self, parent_layout):
        """
        TABS PRINCIPALES - Crear pestañas del sistema
        EDITAR AQUÍ: Nombres de pestañas, iconos, orden de pestañas
        """
        self.tab_widget = QTabWidget()
        self.tab_widget.setTabPosition(QTabWidget.TabPosition.North)
        
        # CREACIÓN DE PESTAÑAS - Instanciar cada módulo
        self.train_tab = TrainTab()       # Pestaña de entrenamiento
        self.predict_tab = PredictTab()   # Pestaña de predicción
        self.analysis_tab = AnalysisTab() # Pestaña de análisis
        self.results_tab = ResultsTab()   # Pestaña de resultados
        
        # AGREGAR PESTAÑAS AL WIDGET - Configurar títulos y iconos
        # EDITAR AQUÍ: Cambiar nombres o iconos de pestañas
        self.tab_widget.addTab(self.train_tab, "🏋️ Entrenamiento")
        self.tab_widget.addTab(self.predict_tab, "🔍 Predicción")
        self.tab_widget.addTab(self.analysis_tab, "📊 Análisis")
        self.tab_widget.addTab(self.results_tab, "📈 Resultados")
        
        parent_layout.addWidget(self.tab_widget)
    
    
    def apply_styles(self):
        """
        APLICAR ESTILOS - Configuración visual de toda la aplicación
        EDITAR AQUÍ: Colores, tamaños, fuentes, bordes, etc.
        """
        self.setStyleSheet("""
            /* VENTANA PRINCIPAL - Fondo y estilo general */
            QMainWindow {
                background-color: #f5f5f5;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 9pt;  /* Tamaño reducido para pantallas pequeñas */
            }
            
            /* PESTAÑAS - Contenedor y paneles */
            QTabWidget::pane {
                border: 1px solid #c0c0c0;
                background-color: white;
                border-radius: 5px;
            }
            
            QTabWidget::tab-bar {
                alignment: center;
            }
            
            /* PESTAÑAS INDIVIDUALES - Apariencia de cada tab */
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 16px;  /* Padding reducido */
                margin-right: 2px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                font-weight: bold;
                font-size: 9pt;  /* EDITAR TAMAÑO FUENTE PESTAÑAS */
                min-width: 100px;  /* Ancho mínimo para evitar cortes */
            }
            
            QTabBar::tab:selected {
                background-color: white;
                color: #2E7D32;  /* EDITAR COLOR PESTAÑA ACTIVA */
            }
            
            QTabBar::tab:hover {
                background-color: #f0f0f0;
            }
            
            /* FRAMES Y CONTENEDORES - Paneles generales */
            QFrame {
                background-color: white;
                border-radius: 5px;
            }
            
            /* BOTONES PRINCIPALES - Estilo de botones */
            QPushButton {
                background-color: #4CAF50;  /* EDITAR COLOR BOTONES */
                color: white;
                border: none;
                padding: 8px 16px;  /* Padding reducido */
                border-radius: 5px;
                font-weight: bold;
                font-size: 9pt;  /* EDITAR TAMAÑO FUENTE BOTONES */
                min-height: 20px;  /* Altura mínima */
            }
            
            QPushButton:hover {
                background-color: #45a049;  /* EDITAR COLOR HOVER */
            }
            
            QPushButton:pressed {
                background-color: #3d8b40;  /* EDITAR COLOR PRESSED */
            }
            
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            
            /* LABELS Y TEXTO - Configuración de texto */
            QLabel {
                font-size: 9pt;  /* EDITAR TAMAÑO FUENTE LABELS */
            }
            
            /* COMBOS Y INPUTS - Controles de entrada */
            QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: white;
                color: black;
                font-size: 9pt;  /* EDITAR TAMAÑO FUENTE INPUTS */
                padding: 4px;
                min-height: 20px;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
            
            QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
                border: 1px solid #4CAF50;
            }
            
            QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 2px solid #4CAF50;
            }
            
            /* SPINBOX BUTTONS - Botones de subir/bajar en SpinBox */
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 16px;
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-bottom: none;
                border-top-right-radius: 3px;
            }
            
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
                background-color: #4CAF50;
            }
            
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                image: none;
                border-left: 4px solid none;
                border-right: 4px solid none;
                border-bottom: 5px solid #666;
                width: 0px;
                height: 0px;
            }
            
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 16px;
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-top: none;
                border-bottom-right-radius: 3px;
            }
            
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #4CAF50;
            }
            
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                image: none;
                border-left: 4px solid none;
                border-right: 4px solid none;
                border-top: 5px solid #666;
                width: 0px;
                height: 0px;
            }
            
            /* COMBOBOX DROPDOWN - Botón desplegable */
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                background-color: #f0f0f0;
                border-left: 1px solid #ccc;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }
            
            QComboBox::drop-down:hover {
                background-color: #4CAF50;
            }
            
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid none;
                border-right: 4px solid none;
                border-top: 5px solid #666;
                width: 0px;
                height: 0px;
            }
            
            /* CHECKBOXES - Casillas de verificación */
            QCheckBox {
                background-color: transparent;
                color: black;
                font-size: 9pt;
            }
            
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                background-color: white;
                border: 1px solid #ccc;
                border-radius: 3px;
            }
            
            QCheckBox::indicator:checked {
                background-color: #4CAF50;
                border: 1px solid #4CAF50;
            }
            
            QCheckBox::indicator:checked::after {
                content: "✓";
                color: white;
                font-weight: bold;
            }
            
            /* GRUPOS - GroupBox styling */
            QGroupBox {
                font-size: 9pt;  /* EDITAR TAMAÑO FUENTE GRUPOS */
                font-weight: bold;
                margin-top: 10px;
                padding-top: 10px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            
            /* TABLAS - Styling para tablas */
            QTableWidget {
                font-size: 8pt;  /* EDITAR TAMAÑO FUENTE TABLAS */
                gridline-color: #ddd;
            }
            
            QTableWidget::item {
                padding: 4px;
                border-bottom: 1px solid #eee;
            }
            
            /* SCROLLBARS - Barras de desplazamiento más delgadas */
            QScrollBar:vertical {
                background: #f0f0f0;
                width: 12px;  /* Ancho reducido */
                border-radius: 6px;
            }
            
            QScrollBar::handle:vertical {
                background: #c0c0c0;
                border-radius: 6px;
            }
            
            QScrollBar::handle:vertical:hover {
                background: #a0a0a0;
            }
        """)
    
    def setup_connections(self):
        """Configurar conexiones de señales"""
        # Conectar señales entre pestañas
        self.train_tab.training_completed.connect(self.on_training_completed)
        self.predict_tab.prediction_completed.connect(self.on_prediction_completed)
    
    def on_training_completed(self, success: bool, model_path: str):
        """Manejar completación del entrenamiento"""
        if success:
            self.results_tab.load_training_results(model_path)
    
    def on_prediction_completed(self, success: bool, results_path: str):
        """Manejar completación de la predicción"""
        if success:
            # Actualizar lista de predicciones en análisis y resultados
            self.analysis_tab.load_available_predictions()
            self.results_tab.load_available_results()
    
    def show_about(self):
        """Mostrar ventana Acerca de"""
        from PyQt6.QtWidgets import QMessageBox
        
        about_text = """
        <h3>Análisis de Plantas de Frijol</h3>
        <p><b>Versión:</b> 1.0.0</p>
        <p><b>Autor:</b> Sistema de Visión Computacional</p>
        <p><b>Descripción:</b> Sistema para detección y análisis de enfermedades en plantas de frijol usando YOLO v9</p>
        <p><b>Características:</b></p>
        <ul>
        <li>Entrenamiento de modelos YOLO</li>
        <li>Detección automática de enfermedades</li>
        <li>Análisis de afectación por color</li>
        <li>Visualización de resultados</li>
        </ul>
        """
        
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Acerca de")
        msg_box.setText(about_text)
        msg_box.exec()
    
    def closeEvent(self, event):
        """Manejar evento de cierre"""
        # Aquí puedes agregar confirmación de cierre si hay procesos en ejecución
        event.accept()
