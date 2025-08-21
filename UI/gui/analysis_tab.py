"""
Pestaña de análisis de afectación en plantas
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QPushButton, QLabel, QComboBox, QProgressBar, 
                             QTextEdit, QFormLayout, QSplitter, QFrame,
                             QScrollArea, QGridLayout, QMessageBox, QSpinBox,
                             QTableWidget, QTableWidgetItem, QHeaderView, QDialog,
                             QCheckBox, QRadioButton, QButtonGroup)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QPainter, QPen, QImage

import os
import numpy as np
import cv2  # Para análisis de color
from pathlib import Path
from utils.config import config
from utils.plant_analyzer import PlantAnalyzer

# Importaciones para histograma
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class AnalysisWorker(QThread):
    """Worker thread para análisis de plantas"""
    
    progress_update = pyqtSignal(str)
    analysis_completed = pyqtSignal(bool, dict)
    
    def __init__(self, crops_dir):
        super().__init__()
        self.crops_dir = crops_dir
        self.analyzer = PlantAnalyzer()
    
    def run(self):
        """Ejecutar análisis"""
        try:
            self.progress_update.emit("Iniciando análisis de afectación...")
            
            # Analizar directorio de crops
            results = self.analyzer.analyze_crops_directory(self.crops_dir)
            
            if results:
                self.progress_update.emit("Calculando estadísticas...")
                summary = self.analyzer.calculate_summary_statistics(results)
                
                self.progress_update.emit("¡Análisis completado exitosamente!")
                self.analysis_completed.emit(True, {"results": results, "summary": summary})
            else:
                self.progress_update.emit("No se encontraron resultados válidos")
                self.analysis_completed.emit(False, {})
                
        except Exception as e:
            self.progress_update.emit(f"Error: {str(e)}")
            self.analysis_completed.emit(False, {})

class ImageDisplayWidget(QLabel):
    """Widget para mostrar imágenes con información y zoom"""
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(300, 200)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #cccccc;
                border-radius: 5px;
                background-color: #ffffff;
                color: #000000;
                cursor: pointer;
            }
            QLabel:hover {
                border: 2px solid #4CAF50;
            }
        """)  # COLORES FIJOS PARA MODO OSCURO
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("Seleccione una imagen para visualizar")
        
        # Variables para manejar imágenes
        self.current_image_path = None
        self.current_title = ""
        self.comparison_image_path = None  # Para alternar entre original y análisis
        self.comparison_title = ""
        self.original_image_path = None  # NUEVA: Para guardar siempre la ruta de la imagen original
    
    def display_image(self, image_path, title=""):
        """Mostrar imagen con título"""
        self.current_image_path = image_path
        self.current_title = title
        
        if Path(image_path).exists():
            pixmap = QPixmap(str(image_path))
            if not pixmap.isNull():
                # Escalar imagen manteniendo aspecto
                scaled_pixmap = pixmap.scaled(
                    self.size(), Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
                self.setPixmap(scaled_pixmap)
                self.setToolTip(f"{title}\n{image_path}\nClic para ampliar")
            else:
                self.setText(f"Error cargando imagen:\n{title}")
        else:
            self.setText(f"Imagen no encontrada:\n{title}")
    
    def set_comparison_image(self, comparison_path, comparison_title):
        """Establecer imagen de comparación para alternar"""
        self.comparison_image_path = comparison_path
        self.comparison_title = comparison_title
    
    def mousePressEvent(self, event):
        """Manejar clic en la imagen"""
        if event.button() == Qt.MouseButton.LeftButton and self.current_image_path:
            self.open_zoom_window()
        super().mousePressEvent(event)
    
    def open_zoom_window(self):
        """Abrir ventana avanzada con filtros y análisis completo"""
        try:
            from .advanced_image_viewer import AdvancedImageViewer
            
            # Crear y mostrar ventana avanzada
            viewer = AdvancedImageViewer(
                parent=self,
                image_path=self.current_image_path,
                title=self.current_title,
                comparison_path=self.comparison_image_path,
                comparison_title=self.comparison_title
            )
            viewer.exec()
            
        except Exception as e:
            print(f"Error abriendo visor avanzado: {e}")
            # Fallback a la ventana simple anterior
            self.open_simple_zoom_window()
    
    def open_simple_zoom_window(self):
        """Ventana simple de zoom (fallback)"""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QScrollArea, QPushButton, QSlider
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Vista detallada - {self.current_title}")
        dialog.resize(900, 700)
        
        layout = QVBoxLayout(dialog)
        
        # Área de scroll para la imagen
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        # Label para mostrar la imagen
        self.zoom_image_label = QLabel()
        self.zoom_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Cargar imagen inicial
        self.zoom_current_path = self.current_image_path
        self.zoom_current_title = self.current_title
        self.load_zoom_image()
        
        scroll_area.setWidget(self.zoom_image_label)
        layout.addWidget(scroll_area)
        
        # Controles inferiores
        controls_layout = QHBoxLayout()
        
        # Controles de zoom
        zoom_slider = QSlider(Qt.Orientation.Horizontal)
        zoom_slider.setMinimum(25)
        zoom_slider.setMaximum(300)
        zoom_slider.setValue(100)
        zoom_label = QLabel("100%")
        
        def on_zoom_changed(value):
            zoom_label.setText(f"{value}%")
            if hasattr(self, 'original_pixmap'):
                new_width = int(self.original_pixmap.width() * value / 100)
                new_height = int(self.original_pixmap.height() * value / 100)
                scaled_pixmap = self.original_pixmap.scaled(
                    new_width, new_height, 
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
                self.zoom_image_label.setPixmap(scaled_pixmap)
        
        zoom_slider.valueChanged.connect(on_zoom_changed)
        
        # Botones de zoom
        zoom_fit_btn = QPushButton("Ajustar")
        zoom_fit_btn.clicked.connect(lambda: zoom_slider.setValue(100))
        
        zoom_in_btn = QPushButton("Zoom +")
        zoom_in_btn.clicked.connect(lambda: zoom_slider.setValue(min(300, zoom_slider.value() + 25)))
        
        zoom_out_btn = QPushButton("Zoom -")
        zoom_out_btn.clicked.connect(lambda: zoom_slider.setValue(max(25, zoom_slider.value() - 25)))
        
        # Botón de comparar (solo si hay imagen de comparación)
        self.compare_btn = QPushButton("🔄 Comparar")
        self.compare_btn.clicked.connect(lambda: self.toggle_comparison_image(zoom_slider))
        self.compare_btn.setEnabled(self.comparison_image_path is not None)
        
        if self.comparison_image_path:
            self.compare_btn.setToolTip(f"Alternar con: {self.comparison_title}")
        else:
            self.compare_btn.setToolTip("No hay imagen de comparación disponible")
        
        # Botón de análisis con máscaras
        self.mask_btn = QPushButton("🎭 Ver Máscaras")
        self.mask_btn.clicked.connect(lambda: self.show_mask_options(dialog, zoom_slider))
        self.mask_btn.setToolTip("Ver imagen con máscaras de afectación aplicadas")
        
        controls_layout.addWidget(QLabel("Zoom:"))
        controls_layout.addWidget(zoom_out_btn)
        controls_layout.addWidget(zoom_slider)
        controls_layout.addWidget(zoom_in_btn)
        controls_layout.addWidget(zoom_fit_btn)
        controls_layout.addWidget(zoom_label)
        controls_layout.addStretch()
        controls_layout.addWidget(self.compare_btn)
        controls_layout.addWidget(self.mask_btn)
        
        layout.addLayout(controls_layout)
        
        # Botón cerrar
        close_btn = QPushButton("Cerrar")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.exec()
    
    def load_zoom_image(self):
        """Cargar imagen en el zoom"""
        if Path(self.zoom_current_path).exists():
            self.original_pixmap = QPixmap(str(self.zoom_current_path))
            if not self.original_pixmap.isNull():
                # Mostrar imagen en tamaño original
                self.zoom_image_label.setPixmap(self.original_pixmap)
            else:
                self.zoom_image_label.setText("Error cargando imagen")
        else:
            self.zoom_image_label.setText("Imagen no encontrada")
    
    def toggle_comparison_image(self, zoom_slider):
        """Alternar entre imagen actual y de comparación"""
        if self.comparison_image_path:
            # Intercambiar imágenes
            temp_path = self.zoom_current_path
            temp_title = self.zoom_current_title
            
            self.zoom_current_path = self.comparison_image_path
            self.zoom_current_title = self.comparison_title
            
            self.comparison_image_path = temp_path
            self.comparison_title = temp_title
            
            # Actualizar imagen
            self.load_zoom_image()
            
            # Actualizar título de la ventana
            dialog = self.compare_btn.window()
            dialog.setWindowTitle(f"Vista detallada - {self.zoom_current_title}")
            
            # Actualizar tooltip del botón
            self.compare_btn.setToolTip(f"Alternar con: {self.comparison_title}")
            
            # Resetear zoom
            zoom_slider.setValue(100)
    
    def show_mask_options(self, parent_dialog, zoom_slider):
        """Mostrar opciones para seleccionar máscaras"""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, QRadioButton, QButtonGroup, QPushButton, QGroupBox
        
        options_dialog = QDialog(parent_dialog)
        options_dialog.setWindowTitle("Opciones de Máscaras de Afectación")
        options_dialog.resize(350, 300)
        
        layout = QVBoxLayout(options_dialog)
        
        # Título
        title = QLabel("🎭 Configurar Visualización de Máscaras")
        title.setFont(QFont("", 12, QFont.Weight.Bold))
        title.setStyleSheet("color: #000000; padding: 10px; background-color: #e8f5e8; border-radius: 5px;")
        layout.addWidget(title)
        
        # Grupo de máscaras
        masks_group = QGroupBox("Seleccionar Máscaras")
        masks_layout = QVBoxLayout(masks_group)
        
        self.mask_healthy = QCheckBox("🟢 Zona Saludable (Verde)")
        self.mask_healthy.setChecked(True)
        self.mask_healthy.setStyleSheet("color: #000000; font-weight: bold;")
        masks_layout.addWidget(self.mask_healthy)
        
        self.mask_disease1 = QCheckBox("🟡 Afectación Leve (Amarillo)")
        self.mask_disease1.setChecked(True) 
        self.mask_disease1.setStyleSheet("color: #000000; font-weight: bold;")
        masks_layout.addWidget(self.mask_disease1)
        
        self.mask_disease2 = QCheckBox("🔴 Afectación Severa (Rojo)")
        self.mask_disease2.setChecked(True)
        self.mask_disease2.setStyleSheet("color: #000000; font-weight: bold;")
        masks_layout.addWidget(self.mask_disease2)
        
        layout.addWidget(masks_group)
        
        # Grupo de color de fondo
        bg_group = QGroupBox("Color de Fondo")
        bg_layout = QVBoxLayout(bg_group)
        
        self.bg_button_group = QButtonGroup()
        
        self.bg_white = QRadioButton("⚪ Fondo Blanco")
        self.bg_white.setChecked(True)
        self.bg_white.setStyleSheet("color: #000000; font-weight: bold;")
        self.bg_button_group.addButton(self.bg_white)
        bg_layout.addWidget(self.bg_white)
        
        self.bg_black = QRadioButton("⚫ Fondo Negro")
        self.bg_black.setStyleSheet("color: #000000; font-weight: bold;")
        self.bg_button_group.addButton(self.bg_black)
        bg_layout.addWidget(self.bg_black)
        
        layout.addWidget(bg_group)
        
        # Botones
        buttons_layout = QHBoxLayout()
        
        apply_btn = QPushButton("✅ Aplicar Máscaras")
        apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        apply_btn.clicked.connect(lambda: self.apply_masks(options_dialog, zoom_slider))
        buttons_layout.addWidget(apply_btn)
        
        cancel_btn = QPushButton("❌ Cancelar")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                padding: 10px;
                border: none;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        cancel_btn.clicked.connect(options_dialog.close)
        buttons_layout.addWidget(cancel_btn)
        
        layout.addLayout(buttons_layout)
        
        options_dialog.exec()
    
    def apply_masks(self, options_dialog, zoom_slider):
        """Aplicar máscaras seleccionadas para hacer recorte de zonas específicas"""
        try:
            import cv2
            import numpy as np
            from pathlib import Path
            
            # Cerrar diálogo de opciones
            options_dialog.close()
            
            # USAR IMAGEN ORIGINAL, no la de análisis que puede estar cargada
            original_path = self.original_image_path if self.original_image_path else self.zoom_current_path
            
            print(f"DEBUG: Usando imagen: {original_path}")
            print(f"DEBUG: Imagen actual en zoom: {self.zoom_current_path}")
            
            # Leer imagen original
            image = cv2.imread(original_path)
            if image is None:
                QMessageBox.warning(self.parent(), "Error", "No se pudo cargar la imagen original")
                return
            
            # Convertir a RGB para procesamiento
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convertir a HSV para análisis de color
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Definir rangos de color para plantas
            # Verde saludable
            lower_healthy = np.array([40, 40, 40])
            upper_healthy = np.array([80, 255, 255])
            
            # Amarillo/marrón (enfermedad leve)
            lower_disease1 = np.array([15, 40, 40])
            upper_disease1 = np.array([35, 255, 255])
            
            # Marrón oscuro/negro (necrosis severa)
            lower_disease2 = np.array([0, 0, 0])
            upper_disease2 = np.array([15, 255, 100])
            
            # Crear máscaras
            mask_healthy_cv = cv2.inRange(hsv, lower_healthy, upper_healthy)
            mask_disease1_cv = cv2.inRange(hsv, lower_disease1, upper_disease1)
            mask_disease2_cv = cv2.inRange(hsv, lower_disease2, upper_disease2)
            
            # Combinar máscaras seleccionadas
            combined_mask = np.zeros_like(mask_healthy_cv)
            
            if self.mask_healthy.isChecked():
                combined_mask = cv2.bitwise_or(combined_mask, mask_healthy_cv)
            
            if self.mask_disease1.isChecked():
                combined_mask = cv2.bitwise_or(combined_mask, mask_disease1_cv)
                
            if self.mask_disease2.isChecked():
                combined_mask = cv2.bitwise_or(combined_mask, mask_disease2_cv)
            
            # Encontrar el contorno del área combinada para hacer el recorte
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Encontrar el bounding box que contiene todas las áreas seleccionadas
                all_contours = np.vstack(contours)
                x, y, w, h = cv2.boundingRect(all_contours)
                
                # Agregar un pequeño margen al recorte
                margin = 20
                x = max(0, x - margin)
                y = max(0, y - margin) 
                w = min(image_rgb.shape[1] - x, w + 2 * margin)
                h = min(image_rgb.shape[0] - y, h + 2 * margin)
                
                # Hacer el recorte de la imagen original
                cropped_image = image_rgb[y:y+h, x:x+w]
                cropped_mask = combined_mask[y:y+h, x:x+w]
                
                # Crear imagen resultado con el fondo seleccionado
                result_image = np.zeros_like(cropped_image)
                
                # Determinar color de fondo
                if self.bg_white.isChecked():
                    result_image.fill(255)  # Fondo blanco
                else:
                    result_image.fill(0)    # Fondo negro
                
                # Aplicar solo las zonas de la máscara al recorte
                # AQUÍ ES DONDE MOSTRAMOS LA IMAGEN ORIGINAL, NO LOS COLORES DE ANÁLISIS
                result_image[cropped_mask > 0] = cropped_image[cropped_mask > 0]
                
                print(f"DEBUG: Recorte aplicado - Dimensiones: {cropped_image.shape}")
                print(f"DEBUG: Píxeles de máscara encontrados: {np.sum(cropped_mask > 0)}")
                print(f"DEBUG: Color de fondo: {'Blanco' if self.bg_white.isChecked() else 'Negro'}")
                
                # Convertir numpy array a QPixmap
                height, width, channel = result_image.shape
                bytes_per_line = 3 * width
                q_image = QImage(result_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                self.original_pixmap = QPixmap.fromImage(q_image)
                
                # Actualizar imagen en la ventana de zoom
                self.zoom_image_label.setPixmap(self.original_pixmap)
                
                # Actualizar título
                mask_names = []
                if self.mask_healthy.isChecked():
                    mask_names.append("Saludable")
                if self.mask_disease1.isChecked():
                    mask_names.append("Leve")
                if self.mask_disease2.isChecked():
                    mask_names.append("Severa")
                
                bg_name = "Blanco" if self.bg_white.isChecked() else "Negro"
                
                parent_dialog = self.mask_btn.window()
                parent_dialog.setWindowTitle(f"Recorte de máscaras: {', '.join(mask_names)} - Fondo {bg_name}")
                
                # Resetear zoom
                zoom_slider.setValue(100)
                
            else:
                QMessageBox.information(self.parent(), "Sin resultados", 
                    "No se encontraron zonas que coincidan con las máscaras seleccionadas")
            
        except Exception as e:
            QMessageBox.critical(self.parent(), "Error", f"Error aplicando máscaras:\n{str(e)}")

class AnalysisTab(QWidget):
    """Pestaña para análisis de afectación en plantas"""
    
    def __init__(self):
        super().__init__()
        self.analyzer = PlantAnalyzer()
        self.analysis_worker = None
        self.current_results = {}
        self.current_summary = {}
        self.setup_ui()
        self.load_available_predictions()
    
    def setup_ui(self):
        """
        CONFIGURAR INTERFAZ DE ANÁLISIS
        EDITAR AQUÍ: Layout principal, distribución de paneles
        """
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)  # Márgenes reducidos
        layout.setSpacing(5)
        
        # SPLITTER PRINCIPAL - Divide en panel de controles y resultados
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # PANEL IZQUIERDO - Configuración y control (más estrecho)
        left_panel = self.create_control_panel()
        splitter.addWidget(left_panel)
        
        # PANEL DERECHO - Resultados y visualización (más ancho)
        right_panel = self.create_results_panel()
        splitter.addWidget(right_panel)
        
        # PROPORCIONES - Adaptadas para pantallas pequeñas
        splitter.setSizes([320, 780])
    
    def create_control_panel(self):
        """
        PANEL DE CONTROL - Configuración del análisis
        EDITAR AQUÍ: Parámetros de análisis, selección de datos
        """
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        panel.setMaximumWidth(340)  # Ancho controlado para pantallas pequeñas
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)  # Espaciado reducido
        
        # TÍTULO DE LA SECCIÓN
        title = QLabel("🔬 Análisis de Afectación")
        title_font = QFont()
        title_font.setPointSize(12)  # Tamaño reducido para pantallas pequeñas
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #000000; background-color: #e8f5e8; padding: 8px; border-radius: 4px; font-weight: bold;")  # EDITAR COLOR TÍTULO - FIJO
        layout.addWidget(title)
        
        # GRUPO: SELECCIÓN DE PREDICCIÓN
        pred_group = QGroupBox("Seleccionar Predicción")
        pred_group.setMaximumHeight(90)  # Altura compacta
        pred_layout = QFormLayout(pred_group)
        pred_layout.setSpacing(5)
        
        self.prediction_combo = QComboBox()
        self.prediction_combo.setMinimumHeight(25)  # Altura mínima para visibilidad
        pred_layout.addRow("Predicción:", self.prediction_combo)
        
        refresh_btn = QPushButton("🔄 Actualizar")
        refresh_btn.setMaximumHeight(25)
        refresh_btn.clicked.connect(self.load_available_predictions)
        pred_layout.addRow("", refresh_btn)
        
        layout.addWidget(pred_group)
        
        # INFORMACIÓN DE LA PREDICCIÓN SELECCIONADA
        info_group = QGroupBox("Información")
        info_group.setMaximumHeight(100)  # Altura compacta
        info_layout = QVBoxLayout(info_group)
        info_layout.setSpacing(5)
        
        self.info_label = QLabel("Seleccione una predicción para ver información")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("""
            padding: 8px; 
            color: #000000;
            background-color: #ffffff;
            border: 1px solid #cccccc;
            border-radius: 3px;
            font-size: 10px;  /* Tamaño compacto */
        """)  # EDITAR ESTILO INFORMACIÓN - COLORES FIJOS
        info_layout.addWidget(self.info_label)
        
        layout.addWidget(info_group)
        
        # GRUPO: PARÁMETROS DE ANÁLISIS
        params_group = QGroupBox("Parámetros de Análisis")
        params_group.setMaximumHeight(120)  # Altura controlada
        params_layout = QFormLayout(params_group)
        params_layout.setSpacing(5)
        
        # SENSIBILIDAD - Control de detección de afectaciones
        self.sensitivity_spin = QSpinBox()
        self.sensitivity_spin.setRange(1, 100)
        self.sensitivity_spin.setValue(50)
        self.sensitivity_spin.setMinimumHeight(25)
        params_layout.addRow("Sensibilidad:", self.sensitivity_spin)
        
        # ÁREA MÍNIMA - Tamaño mínimo de afectación a detectar
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(10, 1000)
        self.min_area_spin.setValue(100)
        self.min_area_spin.setMinimumHeight(25)
        params_layout.addRow("Área mínima:", self.min_area_spin)
        
        layout.addWidget(params_group)
        
        # BOTONES PRINCIPALES
        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(8)
        
        # BOTÓN INICIAR ANÁLISIS
        self.analyze_btn = QPushButton("🚀 Iniciar Análisis")
        self.analyze_btn.setMinimumHeight(35)  # Altura mínima para visibilidad
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                font-size: 11px;  /* Tamaño reducido */
                border: none;
                border-radius: 6px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)  # EDITAR ESTILO BOTÓN ANÁLISIS
        self.analyze_btn.clicked.connect(self.start_analysis)
        buttons_layout.addWidget(self.analyze_btn)
        
        # BOTÓN EXPORTAR RESULTADOS
        self.export_btn = QPushButton("💾 Exportar Resultados")
        self.export_btn.setMinimumHeight(35)
        self.export_btn.setEnabled(False)
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-weight: bold;
                font-size: 11px;
                border: none;
                border-radius: 6px;
                padding: 8px;
            }
            QPushButton:hover:enabled {
                background-color: #F57C00;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)  # EDITAR ESTILO BOTÓN EXPORTAR
        self.export_btn.clicked.connect(self.export_results)
        buttons_layout.addWidget(self.export_btn)
        
        # BOTÓN ESTADÍSTICAS DETALLADAS
        self.stats_btn = QPushButton("📊 Estadísticas Avanzadas")
        self.stats_btn.setMinimumHeight(35)
        self.stats_btn.setEnabled(False)
        self.stats_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                font-weight: bold;
                font-size: 11px;
                border: none;
                border-radius: 6px;
                padding: 8px;
            }
            QPushButton:hover:enabled {
                background-color: #7B1FA2;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)  # EDITAR ESTILO BOTÓN ESTADÍSTICAS
        self.stats_btn.clicked.connect(self.show_detailed_statistics)
        buttons_layout.addWidget(self.stats_btn)
        
        layout.addLayout(buttons_layout)
        
        # ESTADO DEL ANÁLISIS
        self.status_label = QLabel("Listo para analizar")
        self.status_label.setStyleSheet("""
            font-weight: bold; 
            color: #000000; 
            background-color: #ffffff;
            padding: 3px;
            font-size: 10pt;
        """)  # EDITAR ESTILO STATUS - COLORES FIJOS
        layout.addWidget(self.status_label)
        
        # BARRA DE PROGRESO
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(20)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 4px;
                text-align: center;
                font-size: 9px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 3px;
            }
        """)  # EDITAR ESTILO BARRA PROGRESO
        layout.addWidget(self.progress_bar)
        
        layout.addStretch()  # Espacio flexible al final
        return panel
    
    def create_results_panel(self):
        """
        PANEL DE RESULTADOS - Visualización de análisis de afectación
        EDITAR AQUÍ: Disposición de resultados, gráficos, estadísticas
        """
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # TÍTULO DEL PANEL
        title = QLabel("📊 Resultados del Análisis")
        title_font = QFont()
        title_font.setPointSize(12)  # Tamaño reducido
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #000000; background-color: #e8f5e8; padding: 4px; border-radius: 4px; font-weight: bold;")  # EDITAR COLOR - FIJO
        layout.addWidget(title)
        
        # Splitter vertical para dividir resultados
        results_splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(results_splitter)
        
        # Panel superior - Tabla de resumen
        summary_panel = self.create_summary_panel()
        results_splitter.addWidget(summary_panel)
        
        # Panel inferior - Visualización detallada
        detail_panel = self.create_detail_panel()
        results_splitter.addWidget(detail_panel)
        
        # Configurar proporciones
        results_splitter.setSizes([300, 400])
        
        return panel
    
    def create_summary_panel(self):
        """Crear panel de resumen"""
        panel = QFrame()
        layout = QVBoxLayout(panel)
        
        # Título
        summary_title = QLabel("📈 Resumen por Imagen")
        summary_title.setFont(QFont("", 12, QFont.Weight.Bold))
        summary_title.setStyleSheet("color: black; padding: 2px;")
        
        layout.addWidget(summary_title)
        
        # Tabla de resumen
        self.summary_table = QTableWidget()
        self.summary_table.setColumnCount(7)  # Agregar una columna para el botón
        self.summary_table.setHorizontalHeaderLabels([
            "Imagen", "Categoría", "Sano (%)", "Afectado (%)", "Severo (%)", "Total Afectado (%)", "🔍"
        ])
        self.summary_table.setMinimumHeight(100)  # Establecer altura mínima

        # Configurar tabla
        header = self.summary_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Fixed)  # Columna de botón fija
        self.summary_table.setColumnWidth(6, 50)  # Ancho fijo para botón
        
        layout.addWidget(self.summary_table)
        
        # Área de recortes disponibles
        crops_group = QGroupBox("🖼️ Recortes Disponibles")
        crops_group.setStyleSheet("color: black; font-weight: bold; padding-top: 3px;")
        crops_group.setMinimumHeight(150)
        crops_layout = QVBoxLayout(crops_group)
        
        # Label informativo
        self.crops_info_label = QLabel("Seleccione una imagen de la tabla para ver sus recortes")
        self.crops_info_label.setStyleSheet("color: #666; font-style: italic; padding: 3px;")
        crops_layout.addWidget(self.crops_info_label)
        
        # Área de scroll para recortes
        crops_scroll = QScrollArea()
        crops_scroll.setWidgetResizable(True)
        # crops_scroll.setMaximumHeight(120)  # Altura limitada
        
        self.crops_widget = QWidget()
        self.crops_layout = QHBoxLayout(self.crops_widget)
        self.crops_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        crops_scroll.setWidget(self.crops_widget)
        crops_layout.addWidget(crops_scroll)
        
        layout.addWidget(crops_group)
        
        return panel
    
    def create_detail_panel(self):
        """Crear panel de detalles"""
        panel = QFrame()
        layout = QHBoxLayout(panel)
        
        # Splitter horizontal para imágenes
        images_splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(images_splitter)
        
        # Imagen original
        original_group = QGroupBox("Imagen Original")
        original_layout = QVBoxLayout(original_group)
        self.original_image = ImageDisplayWidget()
        original_layout.addWidget(self.original_image)
        images_splitter.addWidget(original_group)
        
        # Imagen con análisis
        analysis_group = QGroupBox("Análisis de Color")
        analysis_layout = QVBoxLayout(analysis_group)
        self.analysis_image = ImageDisplayWidget()
        analysis_layout.addWidget(self.analysis_image)
        images_splitter.addWidget(analysis_group)
        
        # Panel de histograma RGB
        histogram_group = QGroupBox("Histograma RGB")
        histogram_layout = QVBoxLayout(histogram_group)
        
        self.histogram_widget = HistogramWidget()
        histogram_layout.addWidget(self.histogram_widget)
        
        images_splitter.addWidget(histogram_group)
        
        # Configurar proporciones
        images_splitter.setSizes([250, 250, 300])
        
        return panel
    
    def load_available_predictions(self):
        """Cargar predicciones disponibles"""
        self.prediction_combo.clear()
        
        predict_runs = config.get_all_predict_runs()
        if predict_runs:
            for run in predict_runs:
                # Verificar si tiene crops
                crops_dir = run / "crops"
                if crops_dir.exists():
                    self.prediction_combo.addItem(f"{run.name}")
        
        # Conectar cambio de selección
        self.prediction_combo.currentTextChanged.connect(self.on_prediction_changed)
        
        if self.prediction_combo.count() > 0:
            self.on_prediction_changed()
    
    def on_prediction_changed(self):
        """Manejar cambio de predicción seleccionada"""
        selection = self.prediction_combo.currentText()
        if selection:
            pred_path = config.RUNS_DIR / selection
            crops_path = pred_path / "crops"
            
            info_text = f"📁 Predicción: {selection}\n"
            info_text += f"📍 Ubicación: {pred_path}\n"
            
            if crops_path.exists():
                # Contar crops por categoría
                categories = {}
                for category_dir in crops_path.iterdir():
                    if category_dir.is_dir():
                        count = len(list(category_dir.glob("*.jpg"))) + len(list(category_dir.glob("*.png")))
                        categories[category_dir.name] = count
                
                info_text += "🔍 Crops encontrados:\n"
                for category, count in categories.items():
                    info_text += f"  • {category}: {count} imágenes\n"
            else:
                info_text += "❌ No se encontraron crops"
            
            self.info_label.setText(info_text)
    
    def start_analysis(self):
        """Iniciar análisis"""
        selection = self.prediction_combo.currentText()
        if not selection:
            QMessageBox.warning(self, "Error", "Seleccione una predicción para analizar")
            return
        
        crops_dir = config.RUNS_DIR / selection / "crops"
        if not crops_dir.exists():
            QMessageBox.warning(self, "Error", "No se encontraron crops para la predicción seleccionada")
            return
        
        # Configurar UI
        self.analyze_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        # Crear y iniciar worker
        self.analysis_worker = AnalysisWorker(str(crops_dir))
        self.analysis_worker.progress_update.connect(self.update_progress)
        self.analysis_worker.analysis_completed.connect(self.on_analysis_completed)
        
        self.analysis_worker.start()
    
    def update_progress(self, message):
        """Actualizar progreso"""
        self.status_label.setText(message)
    
    def on_analysis_completed(self, success, data):
        """Manejar completación del análisis"""
        self.analyze_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if success and data:
            self.current_results = data["results"]
            self.current_summary = data["summary"]
            
            self.update_progress("✅ Análisis completado exitosamente")
            self.display_results()
            self.export_btn.setEnabled(True)
            self.stats_btn.setEnabled(True)
        else:
            self.update_progress("❌ Error durante el análisis")
            self.export_btn.setEnabled(False)
            self.stats_btn.setEnabled(False)
    
    def display_results(self):
        """Mostrar resultados en la interfaz"""
        if not self.current_summary:
            return
        
        # Limpiar tabla
        self.summary_table.setRowCount(0)
        
        # Llenar tabla con resultados
        row = 0
        for image_num, categories in self.current_summary.items():
            for category, stats in categories.items():
                self.summary_table.insertRow(row)
                
                # Llenar columnas
                self.summary_table.setItem(row, 0, QTableWidgetItem(f"img{image_num:03d}"))
                self.summary_table.setItem(row, 1, QTableWidgetItem(category))
                self.summary_table.setItem(row, 2, QTableWidgetItem(f"{stats['sano']:.1f}"))
                self.summary_table.setItem(row, 3, QTableWidgetItem(f"{stats['afectado']:.1f}"))
                self.summary_table.setItem(row, 4, QTableWidgetItem(f"{stats['severo']:.1f}"))
                self.summary_table.setItem(row, 5, QTableWidgetItem(f"{stats['afectacion_total']:.1f}"))
                
                # Agregar botón de selección
                select_btn = QPushButton("� Seleccionar")
                select_btn.setMaximumHeight(25)
                select_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #4CAF50;
                        color: white;
                        border: none;
                        border-radius: 3px;
                        font-size: 8pt;
                        padding: 2px;
                    }
                    QPushButton:hover {
                        background-color: #45a049;
                    }
                """)
                select_btn.clicked.connect(lambda checked, img=image_num, cat=category: self.select_image_for_analysis(img, cat))
                self.summary_table.setCellWidget(row, 6, select_btn)
                
                # Colorear filas según afectación
                afectacion = stats['afectacion_total']
                if afectacion > 70:
                    color = "#ffcdd2"  # Rojo claro
                elif afectacion > 30:
                    color = "#fff3e0"  # Naranja claro
                else:
                    color = "#e8f5e8"  # Verde claro
                
                for col in range(6):
                    item = self.summary_table.item(row, col)
                    item.setBackground(self.palette().color(self.palette().ColorRole.Base))
                    item.setData(Qt.ItemDataRole.BackgroundRole, color)
                
                row += 1
        
        # Generar estadísticas generales
        self.generate_general_statistics()
    
    def generate_general_statistics(self):
        """Generar estadísticas generales"""
        if not self.current_summary:
            return
        
        stats_text = "ESTADÍSTICAS GENERALES\n"
        stats_text += "=" * 30 + "\n\n"
        
        # Calcular promedios por categoría
        category_stats = {}
        for image_num, categories in self.current_summary.items():
            for category, stats in categories.items():
                if category not in category_stats:
                    category_stats[category] = []
                category_stats[category].append(stats)
        
        # Mostrar promedios por categoría
        stats_text += "PROMEDIOS POR CATEGORÍA:\n"
        stats_text += "-" * 25 + "\n"
        
        for category, stats_list in category_stats.items():
            avg_sano = np.mean([s['sano'] for s in stats_list])
            avg_afectado = np.mean([s['afectado'] for s in stats_list])
            avg_severo = np.mean([s['severo'] for s in stats_list])
            avg_total = np.mean([s['afectacion_total'] for s in stats_list])
            
            stats_text += f"\n{category}:\n"
            stats_text += f"  Sano: {avg_sano:.1f}%\n"
            stats_text += f"  Afectado: {avg_afectado:.1f}%\n"
            stats_text += f"  Severo: {avg_severo:.1f}%\n"
            stats_text += f"  Total Afectado: {avg_total:.1f}%\n"
        
        # Mostrar distribución de afectación
        stats_text += "\nDISTRIBUCIÓN DE AFECTACIÓN:\n"
        stats_text += "-" * 28 + "\n"
        
        total_samples = sum(len(stats_list) for stats_list in category_stats.values())
        
        sano_count = sum(1 for stats_list in category_stats.values() 
                        for stats in stats_list if stats['afectacion_total'] < 10)
        leve_count = sum(1 for stats_list in category_stats.values() 
                        for stats in stats_list if 10 <= stats['afectacion_total'] < 30)
        moderado_count = sum(1 for stats_list in category_stats.values() 
                            for stats in stats_list if 30 <= stats['afectacion_total'] < 70)
        severo_count = sum(1 for stats_list in category_stats.values() 
                          for stats in stats_list if stats['afectacion_total'] >= 70)
        
        stats_text += f"Prácticamente sano (< 10%): {sano_count} ({sano_count/total_samples*100:.1f}%)\n"
        stats_text += f"Afectación leve (10-30%): {leve_count} ({leve_count/total_samples*100:.1f}%)\n"
        stats_text += f"Afectación moderada (30-70%): {moderado_count} ({moderado_count/total_samples*100:.1f}%)\n"
        stats_text += f"Afectación severa (≥ 70%): {severo_count} ({severo_count/total_samples*100:.1f}%)\n"
        
        # Las estadísticas generales ahora se muestran en el histograma RGB
        # El histograma se actualiza cuando se selecciona una imagen
        pass
    
    def show_image_detail(self, image_num, category):
        """Mostrar imagen en ventana de detalle con zoom"""
        try:
            # Buscar la imagen correspondiente
            prediction_name = self.prediction_combo.currentText()
            if not prediction_name:
                return
            
            # Buscar en los resultados de predicción - RUTA CORREGIDA
            predict_dir = config.RUNS_DIR / prediction_name
            
            # Primero intentar mostrar la imagen original
            image_file = None
            search_patterns = [
                f"img{image_num:03d}",      # img028
                f"img{image_num:04d}",      # img0028
                f"img{image_num}",          # img28
                f"image{image_num:03d}",    # image028
                f"image{image_num:04d}",    # image0028
                f"image{image_num}",        # image28
            ]
            
            # Buscar con diferentes extensiones
            extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
            
            # Buscar imagen original primero
            for pattern in search_patterns:
                for ext in extensions:
                    potential_file = predict_dir / f"{pattern}{ext}"
                    if potential_file.exists():
                        image_file = potential_file
                        break
                if image_file:
                    break
            
            # Si no se encuentra la imagen original, buscar recortes
            crop_images = []
            if not image_file:
                # Buscar en el subdirectorio crops
                crops_dir = predict_dir / "crops" / category.lower()
                if crops_dir.exists():
                    # Buscar todos los recortes de esta imagen
                    for crop_file in crops_dir.iterdir():
                        if crop_file.is_file():
                            # Verificar si el archivo pertenece a esta imagen
                            for pattern in search_patterns:
                                if pattern in crop_file.name:
                                    crop_images.append(crop_file)
                                    break
            
            # Mostrar imagen o recortes
            if image_file and image_file.exists():
                self.create_image_detail_window(str(image_file), f"Imagen Original {image_num} - {category}")
            elif crop_images:
                # Si hay múltiples recortes, mostrar el primero y crear galería
                self.create_crops_gallery_window(crop_images, f"Recortes de Imagen {image_num} - {category}")
            else:
                # Mostrar información de debug
                debug_info = f"Directorios buscados:\n"
                debug_info += f"- Original: {predict_dir}\n"
                debug_info += f"- Crops: {predict_dir / 'crops' / category.lower()}\n\n"
                
                # Mostrar archivos disponibles
                if predict_dir.exists():
                    available_files = list(predict_dir.glob("*"))[:10]
                    debug_info += f"Archivos en predicción (primeros 10):\n"
                    debug_info += "\n".join([f"  - {f.name}" for f in available_files])
                else:
                    debug_info += "El directorio de predicción no existe"
                
                QMessageBox.warning(self, "Error", 
                    f"No se encontró la imagen para el número {image_num}\n"
                    f"Predicción: {prediction_name}\n"
                    f"Categoría: {category}\n\n"
                    f"{debug_info}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error mostrando imagen:\n{str(e)}")
    
    def create_image_detail_window(self, image_path, title):
        """Crear ventana de detalle de imagen con zoom"""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QScrollArea, QPushButton, QSlider
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QPixmap
        
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        # Crear área de scroll para la imagen
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        # Label para mostrar la imagen
        image_label = QLabel()
        pixmap = QPixmap(image_path)
        
        if not pixmap.isNull():
            # Escalar imagen inicialmente
            scaled_pixmap = pixmap.scaled(600, 400, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            image_label.setPixmap(scaled_pixmap)
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
            scroll_area.setWidget(image_label)
            layout.addWidget(scroll_area)
            
            # Controles de zoom
            controls_layout = QHBoxLayout()
            
            # Slider de zoom
            zoom_slider = QSlider(Qt.Orientation.Horizontal)
            zoom_slider.setRange(25, 300)  # 25% a 300%
            zoom_slider.setValue(100)  # 100% inicial
            
            zoom_label = QLabel("Zoom: 100%")
            
            def on_zoom_changed(value):
                zoom_label.setText(f"Zoom: {value}%")
                # Calcular nuevo tamaño
                new_width = int(pixmap.width() * value / 100)
                new_height = int(pixmap.height() * value / 100)
                
                # Escalar imagen
                scaled_pixmap = pixmap.scaled(new_width, new_height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                image_label.setPixmap(scaled_pixmap)
            
            zoom_slider.valueChanged.connect(on_zoom_changed)
            
            # Botones de zoom rápido
            zoom_fit_btn = QPushButton("Ajustar")
            zoom_fit_btn.clicked.connect(lambda: zoom_slider.setValue(100))
            
            zoom_in_btn = QPushButton("Zoom +")
            zoom_in_btn.clicked.connect(lambda: zoom_slider.setValue(min(300, zoom_slider.value() + 25)))
            
            zoom_out_btn = QPushButton("Zoom -")
            zoom_out_btn.clicked.connect(lambda: zoom_slider.setValue(max(25, zoom_slider.value() - 25)))
            
            controls_layout.addWidget(QLabel("Zoom:"))
            controls_layout.addWidget(zoom_out_btn)
            controls_layout.addWidget(zoom_slider)
            controls_layout.addWidget(zoom_in_btn)
            controls_layout.addWidget(zoom_fit_btn)
            controls_layout.addWidget(zoom_label)
            
            layout.addLayout(controls_layout)
            
            # Botón cerrar
            close_btn = QPushButton("Cerrar")
            close_btn.clicked.connect(dialog.close)
            layout.addWidget(close_btn)
            
        else:
            layout.addWidget(QLabel("Error cargando imagen"))
        
        dialog.exec()
    
    def create_crops_gallery_window(self, crop_images, title):
        """Crear ventana de galería para mostrar múltiples recortes"""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QScrollArea, QPushButton, QGridLayout
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QPixmap
        
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        # Información
        info_label = QLabel(f"Se encontraron {len(crop_images)} recortes:")
        info_label.setStyleSheet("font-weight: bold; padding: 10px; color: black;")
        layout.addWidget(info_label)
        
        # Área de scroll para la galería
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        gallery_widget = QWidget()
        gallery_layout = QGridLayout(gallery_widget)
        
        # Mostrar cada recorte
        for i, crop_path in enumerate(crop_images):
            crop_frame = QFrame()
            crop_frame.setFrameStyle(QFrame.Shape.StyledPanel)
            crop_frame.setMaximumSize(200, 200)
            
            crop_layout = QVBoxLayout(crop_frame)
            
            # Imagen
            crop_label = QLabel()
            pixmap = QPixmap(str(crop_path))
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(150, 150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                crop_label.setPixmap(scaled_pixmap)
            else:
                crop_label.setText("Error")
            crop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            crop_layout.addWidget(crop_label)
            
            # Nombre del archivo
            name_label = QLabel(crop_path.name)
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_label.setStyleSheet("font-size: 8pt; color: black;")
            crop_layout.addWidget(name_label)
            
            # Botón para ver en detalle
            view_btn = QPushButton("Ver")
            view_btn.clicked.connect(lambda checked, path=str(crop_path), name=crop_path.name: 
                                   self.create_image_detail_window(path, f"Recorte: {name}"))
            crop_layout.addWidget(view_btn)
            
            # Agregar a la galería (3 columnas)
            row = i // 3
            col = i % 3
            gallery_layout.addWidget(crop_frame, row, col)
        
        scroll_area.setWidget(gallery_widget)
        layout.addWidget(scroll_area)
        
        # Botón cerrar
        close_btn = QPushButton("Cerrar")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.exec()
    
    def show_detailed_statistics(self):
        """Mostrar ventana de estadísticas detalladas con gráficos"""
        if not self.current_summary:
            return
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            import numpy as np
        except ImportError:
            QMessageBox.warning(self, "Error", "Se requiere matplotlib para mostrar gráficos.\nInstale con: pip install matplotlib")
            return
        
        # Crear ventana de estadísticas
        from PyQt6.QtWidgets import QDialog, QTabWidget
        
        stats_dialog = QDialog(self)
        stats_dialog.setWindowTitle("📊 Estadísticas Avanzadas de Análisis")
        stats_dialog.resize(1000, 700)
        
        layout = QVBoxLayout(stats_dialog)
        
        # Crear tabs para diferentes gráficos
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # Tab 1: Distribución de afectación por categoría
        self.create_category_distribution_tab(tabs)
        
        # Tab 2: Evolución de afectación por imagen
        self.create_evolution_tab(tabs)
        
        # Tab 3: Estadísticas generales
        self.create_general_stats_tab(tabs)
        
        # Tab 4: Distribución de severidad
        self.create_severity_distribution_tab(tabs)
        
        # Botón cerrar
        close_btn = QPushButton("Cerrar")
        close_btn.clicked.connect(stats_dialog.close)
        layout.addWidget(close_btn)
        
        stats_dialog.exec()
    
    def create_category_distribution_tab(self, parent_tabs):
        """Crear tab de distribución por categoría"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            import numpy as np
            
            tab = QWidget()
            layout = QVBoxLayout(tab)
            
            # Crear figura
            fig = Figure(figsize=(12, 8))
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)
            
            # Procesar datos
            categories = {}
            for image_data in self.current_summary.values():
                for category, stats in image_data.items():
                    if category not in categories:
                        categories[category] = {'sano': [], 'afectado': [], 'severo': []}
                    
                    categories[category]['sano'].append(stats['sano'])
                    categories[category]['afectado'].append(stats['afectado'])
                    categories[category]['severo'].append(stats['severo'])
            
            # Crear subplots
            fig.suptitle('Distribución de Afectación por Categoría', fontsize=16, fontweight='bold')
            
            n_categories = len(categories)
            cols = min(3, n_categories)
            rows = (n_categories + cols - 1) // cols
            
            for i, (category, data) in enumerate(categories.items()):
                ax = fig.add_subplot(rows, cols, i + 1)
                
                # Calcular promedios
                avg_sano = np.mean(data['sano'])
                avg_afectado = np.mean(data['afectado'])
                avg_severo = np.mean(data['severo'])
                
                # Crear gráfico de barras
                labels = ['Sano', 'Afectado', 'Severo']
                values = [avg_sano, avg_afectado, avg_severo]
                colors = ['#4CAF50', '#FF9800', '#F44336']
                
                bars = ax.bar(labels, values, color=colors)
                ax.set_title(f'{category}', fontweight='bold')
                ax.set_ylabel('Porcentaje (%)')
                ax.set_ylim(0, 100)
                
                # Agregar valores en las barras
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{value:.1f}%', ha='center', va='bottom')
            
            fig.tight_layout()
            parent_tabs.addTab(tab, "📊 Por Categoría")
            
        except Exception as e:
            # Tab de error
            tab = QWidget()
            layout = QVBoxLayout(tab)
            layout.addWidget(QLabel(f"Error generando gráfico: {str(e)}"))
            parent_tabs.addTab(tab, "📊 Por Categoría")
    
    def create_evolution_tab(self, parent_tabs):
        """Crear tab de evolución por imagen"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            import numpy as np
            
            tab = QWidget()
            layout = QVBoxLayout(tab)
            
            # Crear figura
            fig = Figure(figsize=(12, 6))
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)
            
            ax = fig.add_subplot(111)
            
            # Procesar datos por imagen
            images = sorted(self.current_summary.keys())
            
            for category in set(cat for img_data in self.current_summary.values() for cat in img_data.keys()):
                sano_vals = []
                afectado_vals = []
                
                for img_num in images:
                    if category in self.current_summary[img_num]:
                        stats = self.current_summary[img_num][category]
                        sano_vals.append(stats['sano'])
                        afectado_vals.append(stats['afectacion_total'])
                    else:
                        sano_vals.append(0)
                        afectado_vals.append(0)
                
                # Graficar líneas
                ax.plot([f"img{i:03d}" for i in images], afectado_vals, 
                       marker='o', linewidth=2, label=f'{category} - Afectado')
            
            ax.set_title('Evolución de Afectación por Imagen', fontsize=14, fontweight='bold')
            ax.set_xlabel('Imagen')
            ax.set_ylabel('Porcentaje de Afectación (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Rotar etiquetas del eje x si hay muchas imágenes
            if len(images) > 10:
                plt.setp(ax.get_xticklabels(), rotation=45)
            
            fig.tight_layout()
            parent_tabs.addTab(tab, "📈 Evolución")
            
        except Exception as e:
            tab = QWidget()
            layout = QVBoxLayout(tab)
            layout.addWidget(QLabel(f"Error generando gráfico: {str(e)}"))
            parent_tabs.addTab(tab, "📈 Evolución")
    
    def create_general_stats_tab(self, parent_tabs):
        """Crear tab de estadísticas generales"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Título
        title = QLabel("📋 Resumen General del Análisis")
        title.setFont(QFont("", 14, QFont.Weight.Bold))
        layout.addWidget(title)
        
        # Crear área de texto con estadísticas
        stats_text = QTextEdit()
        stats_text.setReadOnly(True)
        stats_text.setFont(QFont("Consolas", 10))
        
        # Generar estadísticas detalladas
        total_images = len(self.current_summary)
        categories = set(cat for img_data in self.current_summary.values() for cat in img_data.keys())
        
        stats_content = f"RESUMEN GENERAL DEL ANÁLISIS\n"
        stats_content += "=" * 40 + "\n\n"
        stats_content += f"📊 Total de imágenes analizadas: {total_images}\n"
        stats_content += f"🏷️ Categorías detectadas: {len(categories)}\n"
        stats_content += f"📋 Categorías: {', '.join(categories)}\n\n"
        
        # Estadísticas por categoría
        for category in categories:
            category_data = []
            for img_data in self.current_summary.values():
                if category in img_data:
                    category_data.append(img_data[category])
            
            if category_data:
                avg_sano = sum(d['sano'] for d in category_data) / len(category_data)
                avg_afectado = sum(d['afectado'] for d in category_data) / len(category_data)
                avg_severo = sum(d['severo'] for d in category_data) / len(category_data)
                avg_total = sum(d['afectacion_total'] for d in category_data) / len(category_data)
                
                stats_content += f"🔸 CATEGORÍA: {category.upper()}\n"
                stats_content += f"   └─ Muestras: {len(category_data)}\n"
                stats_content += f"   └─ Sano promedio: {avg_sano:.2f}%\n"
                stats_content += f"   └─ Afectado promedio: {avg_afectado:.2f}%\n"
                stats_content += f"   └─ Severo promedio: {avg_severo:.2f}%\n"
                stats_content += f"   └─ Afectación total: {avg_total:.2f}%\n\n"
        
        # Clasificación de severidad global
        all_totals = []
        for img_data in self.current_summary.values():
            for stats in img_data.values():
                all_totals.append(stats['afectacion_total'])
        
        if all_totals:
            sano_count = sum(1 for x in all_totals if x < 10)
            leve_count = sum(1 for x in all_totals if 10 <= x < 30)
            moderado_count = sum(1 for x in all_totals if 30 <= x < 70)
            severo_count = sum(1 for x in all_totals if x >= 70)
            total = len(all_totals)
            
            stats_content += "🎯 DISTRIBUCIÓN DE SEVERIDAD\n"
            stats_content += "-" * 30 + "\n"
            stats_content += f"🟢 Sano (< 10%): {sano_count} ({sano_count/total*100:.1f}%)\n"
            stats_content += f"🟡 Leve (10-30%): {leve_count} ({leve_count/total*100:.1f}%)\n"
            stats_content += f"🟠 Moderado (30-70%): {moderado_count} ({moderado_count/total*100:.1f}%)\n"
            stats_content += f"🔴 Severo (> 70%): {severo_count} ({severo_count/total*100:.1f}%)\n"
        
        stats_text.setText(stats_content)
        layout.addWidget(stats_text)
        
        parent_tabs.addTab(tab, "📋 Resumen")
    
    def create_severity_distribution_tab(self, parent_tabs):
        """Crear tab de distribución de severidad"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
            from matplotlib.figure import Figure
            import numpy as np
            
            tab = QWidget()
            layout = QVBoxLayout(tab)
            
            # Crear figura
            fig = Figure(figsize=(12, 8))
            canvas = FigureCanvas(fig)
            layout.addWidget(canvas)
            
            # Procesar datos para distribución
            all_totals = []
            for img_data in self.current_summary.values():
                for stats in img_data.values():
                    all_totals.append(stats['afectacion_total'])
            
            if all_totals:
                # Gráfico de pastel
                ax1 = fig.add_subplot(221)
                
                sano_count = sum(1 for x in all_totals if x < 10)
                leve_count = sum(1 for x in all_totals if 10 <= x < 30)
                moderado_count = sum(1 for x in all_totals if 30 <= x < 70)
                severo_count = sum(1 for x in all_totals if x >= 70)
                
                sizes = [sano_count, leve_count, moderado_count, severo_count]
                labels = ['Sano\n(< 10%)', 'Leve\n(10-30%)', 'Moderado\n(30-70%)', 'Severo\n(> 70%)']
                colors = ['#4CAF50', '#FFC107', '#FF9800', '#F44336']
                
                # Filtrar valores cero
                filtered_data = [(size, label, color) for size, label, color in zip(sizes, labels, colors) if size > 0]
                if filtered_data:
                    sizes, labels, colors = zip(*filtered_data)
                    
                    wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                    ax1.set_title('Distribución de Severidad', fontweight='bold')
                
                # Histograma
                ax2 = fig.add_subplot(222)
                ax2.hist(all_totals, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
                ax2.set_xlabel('Porcentaje de Afectación')
                ax2.set_ylabel('Frecuencia')
                ax2.set_title('Distribución de Afectación', fontweight='bold')
                ax2.grid(True, alpha=0.3)
                
                # Box plot por categoría
                ax3 = fig.add_subplot(223)
                categories = set(cat for img_data in self.current_summary.values() for cat in img_data.keys())
                category_data = []
                category_names = []
                
                for category in categories:
                    cat_totals = []
                    for img_data in self.current_summary.values():
                        if category in img_data:
                            cat_totals.append(img_data[category]['afectacion_total'])
                    if cat_totals:
                        category_data.append(cat_totals)
                        category_names.append(category)
                
                if category_data:
                    bp = ax3.boxplot(category_data, labels=category_names, patch_artist=True)
                    colors_box = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
                    for patch, color in zip(bp['boxes'], colors_box * len(bp['boxes'])):
                        patch.set_facecolor(color)
                    
                    ax3.set_title('Distribución por Categoría', fontweight='bold')
                    ax3.set_ylabel('Porcentaje de Afectación')
                    ax3.grid(True, alpha=0.3)
                
                # Estadísticas descriptivas
                ax4 = fig.add_subplot(224)
                ax4.axis('off')
                
                mean_val = np.mean(all_totals)
                median_val = np.median(all_totals)
                std_val = np.std(all_totals)
                min_val = np.min(all_totals)
                max_val = np.max(all_totals)
                
                stats_text = f"""
                ESTADÍSTICAS DESCRIPTIVAS
                
                Media: {mean_val:.2f}%
                Mediana: {median_val:.2f}%
                Desviación Std: {std_val:.2f}%
                Mínimo: {min_val:.2f}%
                Máximo: {max_val:.2f}%
                
                Total muestras: {len(all_totals)}
                """
                
                ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
                        verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
            fig.suptitle('Análisis de Distribución de Severidad', fontsize=16, fontweight='bold')
            fig.tight_layout()
            parent_tabs.addTab(tab, "🎯 Severidad")
            
        except Exception as e:
            tab = QWidget()
            layout = QVBoxLayout(tab)
            layout.addWidget(QLabel(f"Error generando gráfico: {str(e)}"))
            parent_tabs.addTab(tab, "🎯 Severidad")
    
    def export_results(self):
        """Exportar resultados a archivo CSV"""
        if not self.current_summary:
            QMessageBox.warning(self, "Error", "No hay resultados para exportar")
            return
        
        from PyQt6.QtWidgets import QFileDialog
        import csv
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Exportar resultados", "analisis_resultados.csv",
            "Archivos CSV (*.csv)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    # Escribir encabezados
                    writer.writerow([
                        'Imagen', 'Categoria', 'Sano_Porcentaje', 'Afectado_Porcentaje',
                        'Severo_Porcentaje', 'Total_Afectado_Porcentaje', 'Cantidad_Muestras'
                    ])
                    
                    # Escribir datos
                    for image_num, categories in self.current_summary.items():
                        for category, stats in categories.items():
                            writer.writerow([
                                f'img{image_num:03d}',
                                category,
                                f"{stats['sano']:.2f}",
                                f"{stats['afectado']:.2f}",
                                f"{stats['severo']:.2f}",
                                f"{stats['afectacion_total']:.2f}",
                                stats['count']
                            ])
                
                QMessageBox.information(self, "Éxito", f"Resultados exportados a:\n{file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error exportando archivo:\n{str(e)}")
    
    def load_prediction_results(self, results_path):
        """Cargar resultados de predicción desde otra pestaña"""
        # Esta función será llamada desde la ventana principal
        # cuando se complete una predicción
        self.load_available_predictions()
        
        # Seleccionar la predicción más reciente
        if self.prediction_combo.count() > 0:
            latest = config.get_latest_predict_run()
            if latest:
                latest_name = latest.name
                index = self.prediction_combo.findText(latest_name)
                if index >= 0:
                    self.prediction_combo.setCurrentIndex(index)
    
    def select_image_for_analysis(self, image_num, category):
        """Seleccionar imagen para análisis completo"""
        try:
            prediction_name = self.prediction_combo.currentText()
            if not prediction_name:
                return
            
            # Buscar en los resultados de predicción
            predict_dir = config.RUNS_DIR / prediction_name
            
            # Buscar la imagen original
            original_image = None
            search_patterns = [
                f"img{image_num:03d}",      # img028
                f"img{image_num:04d}",      # img0028  
                f"img{image_num}",          # img28
                f"image{image_num:03d}",    # image028
                f"image{image_num:04d}",    # image0028
                f"image{image_num}",        # image28
            ]
            
            extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
            
            # Buscar imagen original
            for pattern in search_patterns:
                for ext in extensions:
                    potential_file = predict_dir / f"{pattern}{ext}"
                    if potential_file.exists():
                        original_image = potential_file
                        break
                if original_image:
                    break
            
            if original_image and original_image.exists():
                # 1. Mostrar imagen original en el primer cuadro
                self.original_image.display_image(str(original_image), f"Imagen Original {image_num}")
                self.original_image.original_image_path = str(original_image)  # Guardar ruta original
                
                # 2. Generar y mostrar análisis de color de la imagen original
                analysis_image_path = self.perform_color_analysis(str(original_image), f"analysis_img{image_num:03d}")
                if analysis_image_path:
                    self.analysis_image.display_image(str(analysis_image_path), f"Análisis de Color - Imagen {image_num}")
                    self.analysis_image.original_image_path = str(original_image)  # Guardar ruta original también aquí
                    
                    # Configurar imágenes de comparación para alternar
                    self.original_image.set_comparison_image(str(analysis_image_path), f"Análisis de Color - Imagen {image_num}")
                    self.analysis_image.set_comparison_image(str(original_image), f"Imagen Original {image_num}")
                
                # 3. Actualizar histograma RGB con la imagen original
                self.histogram_widget.update_histogram(str(original_image), f"Histograma - Imagen {image_num}")
                
                # 4. Buscar y mostrar recortes disponibles
                self.load_available_crops(predict_dir, image_num, category)
                
                # Actualizar información
                self.crops_info_label.setText(f"Recortes de la imagen {image_num} - Categoría: {category}")
                
            else:
                QMessageBox.warning(self, "Error", 
                    f"No se encontró la imagen original para el número {image_num}\n"
                    f"Predicción: {prediction_name}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error seleccionando imagen:\n{str(e)}")
    
    def perform_color_analysis(self, image_path, output_name):
        """Realizar análisis de color en una imagen"""
        try:
            # Leer imagen
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # Convertir de BGR a RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convertir a HSV para mejor análisis de color
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Definir rangos de color para plantas
            # Verde saludable
            lower_healthy = np.array([35, 40, 40])
            upper_healthy = np.array([80, 255, 255])
            
            # Amarillo/marrón (enfermedad)
            lower_disease1 = np.array([15, 40, 40])
            upper_disease1 = np.array([35, 255, 255])
            
            # Marrón oscuro/negro (necrosis)
            lower_disease2 = np.array([0, 0, 0])
            upper_disease2 = np.array([19, 255, 200])
            
            # Crear máscaras
            mask_healthy = cv2.inRange(hsv, lower_healthy, upper_healthy)
            mask_disease1 = cv2.inRange(hsv, lower_disease1, upper_disease1)
            mask_disease2 = cv2.inRange(hsv, lower_disease2, upper_disease2)
            
            # CREAR MAPA DE CALOR en lugar de colores planos
            analysis = self.create_heatmap_analysis(image_rgb, mask_healthy, mask_disease1, mask_disease2)
            
            # Guardar imagen de análisis
            temp_dir = Path("temp_analysis")
            temp_dir.mkdir(exist_ok=True)
            analysis_path = temp_dir / f"{output_name}_heatmap.png"
            
            # Convertir de RGB a BGR para guardar
            analysis_bgr = cv2.cvtColor(analysis, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(analysis_path), analysis_bgr)
            
            return analysis_path
            
        except Exception as e:
            print(f"Error en análisis de color: {str(e)}")
            return None
    
    def create_heatmap_analysis(self, image_rgb, mask_healthy, mask_disease1, mask_disease2):
        """Crear mapa de calor para análisis de afectación"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            from matplotlib.colors import ListedColormap
            
            # Crear mapa de intensidad de afectación
            # 0 = fondo, 1 = saludable, 2 = enfermedad leve, 3 = enfermedad severa
            intensity_map = np.zeros((image_rgb.shape[0], image_rgb.shape[1]), dtype=np.float32)
            
            # Asignar valores de intensidad
            intensity_map[mask_healthy > 0] = 1.0    # Saludable
            intensity_map[mask_disease1 > 0] = 2.0   # Enfermedad leve
            intensity_map[mask_disease2 > 0] = 3.0   # Enfermedad severa (máxima prioridad)
            
            # Crear suavizado gaussiano para efecto de calor
            from scipy import ndimage
            intensity_map_smooth = ndimage.gaussian_filter(intensity_map, sigma=1.5)
            
            # Crear colormap personalizado para plantas
            colors = [
                [0.0, 0.0, 0.0, 0.0],      # Transparente para fondo (0)
                [0.0, 0.8, 0.0, 0.8],      # Verde vibrante para saludable (1)
                [1.0, 0.8, 0.0, 0.9],      # Amarillo-naranja para leve (2)
                [1.0, 0.2, 0.0, 1.0]       # Rojo intenso para severo (3)
            ]
            
            # Normalizar mapa de intensidad a rango 0-1
            intensity_normalized = intensity_map_smooth / 3.0
            
            # Aplicar colormap
            cmap = ListedColormap(colors)
            colored_intensity = cmap(intensity_normalized)
            
            # Convertir a RGB (0-255)
            heatmap_rgb = (colored_intensity[:, :, :3] * 255).astype(np.uint8)
            alpha_channel = colored_intensity[:, :, 3]
            
            # Mezclar con imagen original usando alpha blending
            result = image_rgb.copy().astype(np.float32)
            
            # Aplicar mapa de calor con transparencia
            for c in range(3):  # RGB channels
                result[:, :, c] = (
                    alpha_channel * heatmap_rgb[:, :, c] + 
                    (1 - alpha_channel) * result[:, :, c]
                )
            
            # Agregar contornos para mejor definición
            contours_healthy, _ = cv2.findContours(mask_healthy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_disease1, _ = cv2.findContours(mask_disease1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_disease2, _ = cv2.findContours(mask_disease2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Dibujar contornos sutiles
            result = result.astype(np.uint8)
            cv2.drawContours(result, contours_healthy, -1, (0, 200, 0), 1)
            cv2.drawContours(result, contours_disease1, -1, (255, 180, 0), 1)
            cv2.drawContours(result, contours_disease2, -1, (255, 50, 0), 2)
            
            return result
            
        except ImportError:
            # Fallback a método simple si no hay scipy/matplotlib
            print("⚠️ scipy no disponible, usando mapa de calor simplificado")
            return self.create_simple_heatmap(image_rgb, mask_healthy, mask_disease1, mask_disease2)
        except Exception as e:
            print(f"Error creando mapa de calor: {e}")
            return self.create_simple_heatmap(image_rgb, mask_healthy, mask_disease1, mask_disease2)
    
    def create_simple_heatmap(self, image_rgb, mask_healthy, mask_disease1, mask_disease2):
        """Crear mapa de calor simple sin dependencias adicionales"""
        try:
            # Crear imagen base con transparencia
            result = image_rgb.copy().astype(np.float32)
            
            # Crear capas de color con gradientes
            overlay = np.zeros_like(result, dtype=np.float32)
            
            # Verde suave para saludable
            overlay[mask_healthy > 0] = [0, 200, 0]
            
            # Amarillo-naranja para enfermedad leve
            overlay[mask_disease1 > 0] = [255, 200, 0]
            
            # Rojo intenso para enfermedad severa
            overlay[mask_disease2 > 0] = [255, 80, 0]
            
            # Suavizar con filtro gaussiano básico de OpenCV
            overlay = cv2.GaussianBlur(overlay, (5, 5), 1.5)
            
            # Crear máscara alpha basada en intensidad
            alpha = np.zeros((result.shape[0], result.shape[1]), dtype=np.float32)
            alpha[mask_healthy > 0] = 0.6
            alpha[mask_disease1 > 0] = 0.7
            alpha[mask_disease2 > 0] = 0.8
            
            # Suavizar alpha
            alpha = cv2.GaussianBlur(alpha, (5, 5), 1.5)
            
            # Aplicar alpha blending
            for c in range(3):
                result[:, :, c] = alpha * overlay[:, :, c] + (1 - alpha) * result[:, :, c]
            
            return result.astype(np.uint8)
            
        except Exception as e:
            print(f"Error en mapa simple: {e}")
            # Último fallback - colores planos
            analysis = image_rgb.copy()
            analysis[mask_healthy > 0] = [0, 255, 0]
            analysis[mask_disease1 > 0] = [255, 255, 0]
            analysis[mask_disease2 > 0] = [255, 0, 0]
            return analysis
    
    def load_available_crops(self, predict_dir, image_num, category):
        """Cargar recortes disponibles para la imagen seleccionada"""
        # Limpiar recortes anteriores
        for i in reversed(range(self.crops_layout.count())):
            child = self.crops_layout.itemAt(i)
            if child:
                widget = child.widget()
                if widget:
                    widget.setParent(None)
        
        # Buscar recortes
        crops_dir = predict_dir / "crops" / category.lower()
        available_crops = []
        
        if crops_dir.exists():
            search_patterns = [
                f"img{image_num:03d}",
                f"img{image_num:04d}",
                f"img{image_num}",
                f"image{image_num:03d}",
                f"image{image_num:04d}",
                f"image{image_num}",
            ]
            
            # Buscar todos los recortes que coincidan
            for crop_file in crops_dir.iterdir():
                if crop_file.is_file():
                    for pattern in search_patterns:
                        if pattern in crop_file.name:
                            available_crops.append(crop_file)
                            break
        
        if available_crops:
            for crop_file in available_crops:
                # Crear botón clickeable para cada recorte
                crop_btn = self.create_crop_button(crop_file)
                self.crops_layout.addWidget(crop_btn)
        else:
            # Mostrar mensaje si no hay recortes
            no_crops_label = QLabel("No se encontraron recortes para esta imagen")
            no_crops_label.setStyleSheet("color: #999; font-style: italic; padding: 10px;")
            self.crops_layout.addWidget(no_crops_label)
    
    def create_crop_button(self, crop_file):
        """Crear botón clickeable para un recorte"""
        from PyQt6.QtWidgets import QFrame, QVBoxLayout, QPushButton
        from PyQt6.QtGui import QPixmap
        
        frame = QFrame()
        frame.setFixedSize(80, 100)
        frame.setFrameStyle(QFrame.Shape.StyledPanel)
        frame.setStyleSheet("""
            QFrame {
                border: 2px solid #ddd;
                border-radius: 5px;
                background-color: white;
            }
            QFrame:hover {
                border: 2px solid #4CAF50;
                background-color: #f0f8f0;
            }
        """)
        
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(2, 2, 2, 2)
        
        # Imagen miniatura
        img_label = QLabel()
        pixmap = QPixmap(str(crop_file))
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(70, 60, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            img_label.setPixmap(scaled_pixmap)
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(img_label)
        
        # Botón seleccionar
        select_btn = QPushButton("Analizar")
        select_btn.setMaximumHeight(20)
        select_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 3px;
                font-size: 7pt;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        select_btn.clicked.connect(lambda: self.analyze_crop(crop_file))
        layout.addWidget(select_btn)
        
        return frame
    
    def display_image_with_analysis(self, image_path, title):
        """Mostrar imagen original y su análisis de color"""
        try:
            # Cargar imagen original
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    self.original_image.size(), 
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
                self.original_image.setPixmap(scaled_pixmap)
                self.original_image.setToolTip(f"Imagen original: {title}")
                
                # Realizar análisis de color
                self.perform_color_analysis(image_path, title)
                
        except Exception as e:
            print(f"Error mostrando imagen: {e}")
    
    def analyze_crop(self, crop_file):
        """Analizar recorte seleccionado"""
        try:
            # 1. Mostrar recorte original en el primer cuadro
            self.original_image.display_image(str(crop_file), f"Recorte: {crop_file.name}")
            self.original_image.original_image_path = str(crop_file)  # Guardar ruta original del recorte
            
            # 2. Actualizar histograma RGB con el recorte
            self.histogram_widget.update_histogram(str(crop_file), f"Histograma - {crop_file.name}")
            
            # 3. Generar y mostrar análisis de color del recorte
            analysis_image_path = self.perform_color_analysis(str(crop_file), f"crop_analysis_{crop_file.stem}")
            if analysis_image_path:
                self.analysis_image.display_image(str(analysis_image_path), f"Análisis de Color - {crop_file.name}")
                self.analysis_image.original_image_path = str(crop_file)  # Guardar ruta original también aquí
                
                # Configurar imágenes de comparación para alternar
                self.original_image.set_comparison_image(str(analysis_image_path), f"Análisis de Color - {crop_file.name}")
                self.analysis_image.set_comparison_image(str(crop_file), f"Recorte: {crop_file.name}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error analizando recorte:\n{str(e)}")


class HistogramWidget(QWidget):
    """Widget para mostrar histograma RGB clickeable"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.current_image_path = None
        # Inicializar con histograma vacío
        self.clear_histogram()
        
    def setup_ui(self):
        """Configurar interfaz del histograma"""
        layout = QVBoxLayout(self)
        
        # Canvas para el histograma
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect('button_press_event', self.on_histogram_click)
        
        layout.addWidget(self.canvas)
        
        # Label informativo
        self.info_label = QLabel("Seleccione una imagen para ver su histograma RGB")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet("color: #666; font-style: italic; padding: 5px;")
        layout.addWidget(self.info_label)
        
    def update_histogram(self, image_path, title="Histograma RGB"):
        """Actualizar histograma con nueva imagen"""
        try:
            self.current_image_path = image_path
            
            if not os.path.exists(image_path):
                self.clear_histogram()
                return
                
            # Cargar imagen
            image = cv2.imread(image_path)
            if image is None:
                self.clear_histogram()
                return
                
            # Convertir de BGR a RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Limpiar figura
            self.figure.clear()
            
            # Crear histogramas para cada canal
            ax = self.figure.add_subplot(111)
            
            colors = ['red', 'green', 'blue']
            channel_names = ['Rojo', 'Verde', 'Azul']
            
            for i, (color, name) in enumerate(zip(colors, channel_names)):
                hist = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
                ax.plot(hist, color=color, alpha=0.7, linewidth=2, label=name)
            
            ax.set_xlim([0, 256])
            ax.set_xlabel('Intensidad de Color')
            ax.set_ylabel('Frecuencia')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Configurar estilo
            ax.set_facecolor('#f8f8f8')
            self.figure.patch.set_facecolor('white')
            
            # Ajustar layout
            self.figure.tight_layout()
            self.canvas.draw()
            
            # Actualizar info
            self.info_label.setText(f"Histograma de: {os.path.basename(image_path)}\n(Clic para ampliar)")
            
        except Exception as e:
            print(f"Error actualizando histograma: {e}")
            self.clear_histogram()
    
    def clear_histogram(self):
        """Limpiar histograma"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, 'No hay imagen seleccionada', 
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='#666')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw()
        self.info_label.setText("Seleccione una imagen para ver su histograma RGB")
    
    def on_histogram_click(self, event):
        """Manejar clic en el histograma para zoom"""
        if event.button == 1 and self.current_image_path:  # Clic izquierdo
            self.show_histogram_zoom()
    
    def show_histogram_zoom(self):
        """Mostrar histograma con zoom en ventana nueva"""
        if not self.current_image_path:
            return
            
        try:
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Histograma RGB - {os.path.basename(self.current_image_path)}")
            dialog.resize(800, 600)
            
            layout = QVBoxLayout(dialog)
            
            # Canvas más grande para zoom
            zoom_figure = Figure(figsize=(10, 7))
            zoom_canvas = FigureCanvas(zoom_figure)
            
            # Cargar y procesar imagen
            image = cv2.imread(self.current_image_path)
            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Crear histogramas detallados
                ax = zoom_figure.add_subplot(111)
                
                colors = ['red', 'green', 'blue']
                channel_names = ['Canal Rojo', 'Canal Verde', 'Canal Azul']
                
                for i, (color, name) in enumerate(zip(colors, channel_names)):
                    hist = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
                    ax.plot(hist, color=color, alpha=0.8, linewidth=2.5, label=name)
                
                ax.set_xlim([0, 256])
                ax.set_xlabel('Intensidad de Color (0-255)', fontsize=12)
                ax.set_ylabel('Número de Píxeles', fontsize=12)
                ax.set_title(f'Histograma RGB - {os.path.basename(self.current_image_path)}', 
                           fontsize=14, fontweight='bold')
                ax.legend(fontsize=12)
                ax.grid(True, alpha=0.3)
                
                # Agregar estadísticas
                mean_values = [np.mean(image_rgb[:,:,i]) for i in range(3)]
                std_values = [np.std(image_rgb[:,:,i]) for i in range(3)]
                
                stats_text = f"Promedios - R: {mean_values[0]:.1f}, G: {mean_values[1]:.1f}, B: {mean_values[2]:.1f}\n"
                stats_text += f"Desv. Est. - R: {std_values[0]:.1f}, G: {std_values[1]:.1f}, B: {std_values[2]:.1f}"
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Configurar estilo
                ax.set_facecolor('#f8f8f8')
                zoom_figure.patch.set_facecolor('white')
                zoom_figure.tight_layout()
            
            layout.addWidget(zoom_canvas)
            
            # Botón cerrar
            close_btn = QPushButton("Cerrar")
            close_btn.clicked.connect(dialog.close)
            layout.addWidget(close_btn)
            
            dialog.exec()
            
        except Exception as e:
            print(f"Error mostrando histograma con zoom: {e}")
            QMessageBox.warning(self, "Error", f"No se pudo mostrar el histograma:\n{str(e)}")
