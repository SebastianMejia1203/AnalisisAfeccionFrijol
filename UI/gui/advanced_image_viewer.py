"""
Ventana emergente avanzada para visualización de imágenes con filtros
Integra funcionalidad de filtros de procesamiento y análisis RGB
"""

import os
import numpy as np
import cv2
from pathlib import Path

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QScrollArea, 
                             QLabel, QPushButton, QSlider, QComboBox, QSpinBox,
                             QDoubleSpinBox, QGroupBox, QTabWidget, QWidget,
                             QSplitter, QFrame, QGridLayout, QCheckBox, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QFont

# Importaciones para histograma
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Importar nuestro módulo de filtros
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from filtros_procesamiento import FiltrosProcessamiento

class FilterWorker(QThread):
    """Worker thread para aplicar filtros sin bloquear la UI"""
    
    filter_applied = pyqtSignal(np.ndarray)  # Señal con la imagen filtrada
    filter_failed = pyqtSignal(str)  # Señal de error
    
    def __init__(self, image, filter_type, params):
        super().__init__()
        self.image = image.copy()
        self.filter_type = filter_type
        self.params = params
        self.processor = FiltrosProcessamiento()
    
    def run(self):
        """Aplicar filtro en thread separado"""
        try:
            filtered_image = self.processor.aplicar_filtro(
                self.image, self.filter_type, self.params
            )
            self.filter_applied.emit(filtered_image)
        except Exception as e:
            self.filter_failed.emit(str(e))

class AdvancedImageViewer(QDialog):
    """Ventana avanzada para visualización de imágenes con filtros y análisis"""
    
    def __init__(self, parent, image_path, title, comparison_path=None, comparison_title=None):
        super().__init__(parent)
        
        self.image_path = image_path
        self.title = title
        self.comparison_path = comparison_path
        self.comparison_title = comparison_title
        
        # Cargar imagen original
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        
        # Convertir a RGB para procesamiento
        self.current_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.filtered_image = None  # Imagen con filtro aplicado
        
        # Procesador de filtros
        self.processor = FiltrosProcessamiento()
        
        # Variables para manejar imágenes
        self.is_comparing = False  # Estado de comparación
        self.showing_color_analysis = False  # Estado de análisis de color
        
        self.setup_ui()
        self.load_initial_image()
    
    def setup_ui(self):
        """Configurar interfaz de usuario"""
        self.setWindowTitle(f"🖼️ Análisis Avanzado - {self.title}")
        self.resize(1200, 800)
        
        # Layout principal
        main_layout = QHBoxLayout(self)
        
        # Splitter principal
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Panel izquierdo - Controles de filtros
        control_panel = self.create_control_panel()
        splitter.addWidget(control_panel)
        
        # Panel derecho - Visualización
        view_panel = self.create_view_panel()
        splitter.addWidget(view_panel)
        
        # Proporciones
        splitter.setSizes([350, 850])
    
    def create_control_panel(self):
        """Crear panel de controles de filtros"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        panel.setMaximumWidth(370)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Título del panel
        title_label = QLabel("🎛️ Filtros de Procesamiento")
        title_label.setStyleSheet("""
            QLabel {
                background-color: #2c3e50;
                color: white;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
        """)
        layout.addWidget(title_label)
        
        # ============ FILTROS ============
        filter_group = QGroupBox("Tipo de Filtro")
        filter_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 5px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        filter_layout = QVBoxLayout(filter_group)
        
        # Selector de filtro
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("-- Sin Filtro --", None)
        
        # Agregar filtros por categoría
        for filter_name in self.processor.get_filtros_disponibles():
            info = self.processor.get_info_filtro(filter_name)
            tipo = info.get('tipo', 'Desconocido')
            emoji = '🔽' if tipo == 'Pasa-bajo' else '🔼'
            self.filter_combo.addItem(f"{emoji} {filter_name} ({tipo})", filter_name)
        
        self.filter_combo.currentTextChanged.connect(self.on_filter_changed)
        filter_layout.addWidget(self.filter_combo)
        
        # Info del filtro
        self.filter_info = QLabel("Seleccione un filtro para ver información")
        self.filter_info.setWordWrap(True)
        self.filter_info.setStyleSheet("""
            QLabel {
                background-color: #ecf0f1;
                padding: 8px;
                border-radius: 4px;
                border-left: 4px solid #3498db;
            }
        """)
        filter_layout.addWidget(self.filter_info)
        
        # Parámetros del filtro
        self.params_group = QGroupBox("Parámetros")
        self.params_layout = QVBoxLayout(self.params_group)
        filter_layout.addWidget(self.params_group)
        
        layout.addWidget(filter_group)
        
        # ============ CONTROLES DE VISTA ============
        view_group = QGroupBox("Controles de Vista")
        view_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 5px;
            }
        """)
        
        view_layout = QVBoxLayout(view_group)
        
        # Botón para mostrar original
        self.reset_btn = QPushButton("🔄 Mostrar Original")
        self.reset_btn.clicked.connect(self.show_original)
        view_layout.addWidget(self.reset_btn)
        
        # Botón de comparar con original
        self.compare_btn = QPushButton("⚖️ Comparar Original")
        self.compare_btn.clicked.connect(self.toggle_comparison)
        self.compare_btn.setEnabled(False)  # Se habilita cuando hay filtro aplicado
        view_layout.addWidget(self.compare_btn)
        
        # Botón para análisis de color (solo si hay imagen de comparación)
        if self.comparison_path:
            self.color_analysis_btn = QPushButton("🎨 Análisis de Color")
            self.color_analysis_btn.clicked.connect(self.show_color_analysis)
            view_layout.addWidget(self.color_analysis_btn)
        
        # Botón para aplicar filtro
        self.apply_btn = QPushButton("✨ Aplicar Filtro")
        self.apply_btn.clicked.connect(self.apply_current_filter)
        self.apply_btn.setEnabled(False)
        view_layout.addWidget(self.apply_btn)
        
        # Botón para máscaras de recorte
        self.mask_crop_btn = QPushButton("🎭 Recortar con Máscaras")
        self.mask_crop_btn.clicked.connect(self.show_mask_crop_options)
        self.mask_crop_btn.setToolTip("Recortar imagen usando máscaras de afectación")
        view_layout.addWidget(self.mask_crop_btn)
        
        layout.addWidget(view_group)
        
        # ============ INFORMACIÓN ============
        info_group = QGroupBox("Información")
        info_layout = QVBoxLayout(info_group)
        
        self.image_info = QLabel()
        self.image_info.setWordWrap(True)
        self.image_info.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa;
                padding: 8px;
                border-radius: 4px;
            }
        """)
        info_layout.addWidget(self.image_info)
        
        layout.addWidget(info_group)
        
        layout.addStretch()
        return panel
    
    def create_view_panel(self):
        """Crear panel de visualización"""
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Tab widget para imagen y histograma
        self.tab_widget = QTabWidget()
        
        # ============ TAB DE IMAGEN ============
        image_tab = QWidget()
        image_layout = QVBoxLayout(image_tab)
        
        # Área de scroll para la imagen
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        
        # Label para mostrar la imagen
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.scroll_area.setWidget(self.image_label)
        
        image_layout.addWidget(self.scroll_area)
        
        # Controles de zoom
        zoom_controls = QHBoxLayout()
        
        zoom_controls.addWidget(QLabel("Zoom:"))
        
        self.zoom_out_btn = QPushButton("−")
        self.zoom_out_btn.setFixedSize(30, 30)
        self.zoom_out_btn.clicked.connect(lambda: self.change_zoom(-25))
        zoom_controls.addWidget(self.zoom_out_btn)
        
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setMinimum(25)
        self.zoom_slider.setMaximum(300)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.on_zoom_changed)
        zoom_controls.addWidget(self.zoom_slider)
        
        self.zoom_in_btn = QPushButton("+")
        self.zoom_in_btn.setFixedSize(30, 30)
        self.zoom_in_btn.clicked.connect(lambda: self.change_zoom(25))
        zoom_controls.addWidget(self.zoom_in_btn)
        
        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(50)
        zoom_controls.addWidget(self.zoom_label)
        
        self.zoom_fit_btn = QPushButton("Ajustar")
        self.zoom_fit_btn.clicked.connect(lambda: self.zoom_slider.setValue(100))
        zoom_controls.addWidget(self.zoom_fit_btn)
        
        zoom_controls.addStretch()
        image_layout.addLayout(zoom_controls)
        
        self.tab_widget.addTab(image_tab, "🖼️ Imagen")
        
        # ============ TAB DE HISTOGRAMA ============
        histogram_tab = QWidget()
        histogram_layout = QVBoxLayout(histogram_tab)
        
        # Canvas para histograma
        self.hist_figure = Figure(figsize=(10, 6))
        self.hist_canvas = FigureCanvas(self.hist_figure)
        histogram_layout.addWidget(self.hist_canvas)
        
        # Controles de histograma
        hist_controls = QHBoxLayout()
        
        self.hist_original_btn = QPushButton("📊 Original")
        self.hist_original_btn.clicked.connect(lambda: self.update_histogram(self.current_image))
        hist_controls.addWidget(self.hist_original_btn)
        
        self.hist_filtered_btn = QPushButton("📊 Filtrado")
        self.hist_filtered_btn.clicked.connect(lambda: self.update_histogram(self.filtered_image))
        self.hist_filtered_btn.setEnabled(False)
        hist_controls.addWidget(self.hist_filtered_btn)
        
        hist_controls.addStretch()
        histogram_layout.addLayout(hist_controls)
        
        self.tab_widget.addTab(histogram_tab, "📊 Histograma RGB")
        
        layout.addWidget(self.tab_widget)
        
        return panel
    
    def create_filter_params(self, filter_name):
        """Crear controles para parámetros del filtro"""
        # Limpiar parámetros anteriores
        for i in reversed(range(self.params_layout.count())):
            child = self.params_layout.itemAt(i).widget()
            if child:
                child.deleteLater()
        
        if not filter_name:
            return
        
        params = self.processor.get_parametros_filtro(filter_name)
        
        if filter_name in ['Gaussiano']:
            # Kernel size
            kernel_layout = QHBoxLayout()
            kernel_layout.addWidget(QLabel("Tamaño Kernel:"))
            self.kernel_spin = QSpinBox()
            self.kernel_spin.setRange(3, 15)
            self.kernel_spin.setSingleStep(2)  # Solo impares
            self.kernel_spin.setValue(params.get('kernel_size', 5))
            self.kernel_spin.valueChanged.connect(self.on_params_changed)
            kernel_layout.addWidget(self.kernel_spin)
            
            widget = QWidget()
            widget.setLayout(kernel_layout)
            self.params_layout.addWidget(widget)
            
            # Sigma
            sigma_layout = QHBoxLayout()
            sigma_layout.addWidget(QLabel("Sigma:"))
            self.sigma_spin = QDoubleSpinBox()
            self.sigma_spin.setRange(0.1, 5.0)
            self.sigma_spin.setSingleStep(0.1)
            self.sigma_spin.setValue(params.get('sigma', 1.0))
            self.sigma_spin.valueChanged.connect(self.on_params_changed)
            sigma_layout.addWidget(self.sigma_spin)
            
            widget2 = QWidget()
            widget2.setLayout(sigma_layout)
            self.params_layout.addWidget(widget2)
            
        elif filter_name in ['Media', 'Mediana']:
            # Solo kernel size
            kernel_layout = QHBoxLayout()
            kernel_layout.addWidget(QLabel("Tamaño Kernel:"))
            self.kernel_spin = QSpinBox()
            self.kernel_spin.setRange(3, 15)
            self.kernel_spin.setSingleStep(2)
            self.kernel_spin.setValue(params.get('kernel_size', 5))
            self.kernel_spin.valueChanged.connect(self.on_params_changed)
            kernel_layout.addWidget(self.kernel_spin)
            
            widget = QWidget()
            widget.setLayout(kernel_layout)
            self.params_layout.addWidget(widget)
            
        elif filter_name in ['Sobel', 'Prewitt']:
            # Dirección
            dir_layout = QHBoxLayout()
            dir_layout.addWidget(QLabel("Dirección:"))
            self.dir_combo = QComboBox()
            self.dir_combo.addItems(['x', 'y', 'ambas'])
            self.dir_combo.setCurrentText(params.get('direccion', 'ambas'))
            self.dir_combo.currentTextChanged.connect(self.on_params_changed)
            dir_layout.addWidget(self.dir_combo)
            
            widget = QWidget()
            widget.setLayout(dir_layout)
            self.params_layout.addWidget(widget)
            
        elif filter_name in ['Laplaciano']:
            # Tipo de kernel
            type_layout = QHBoxLayout()
            type_layout.addWidget(QLabel("Tipo:"))
            self.type_combo = QComboBox()
            self.type_combo.addItems(['4-conectado', '8-conectado'])
            self.type_combo.setCurrentText(params.get('tipo', '4-conectado'))
            self.type_combo.currentTextChanged.connect(self.on_params_changed)
            type_layout.addWidget(self.type_combo)
            
            widget = QWidget()
            widget.setLayout(type_layout)
            self.params_layout.addWidget(widget)
    
    def get_current_params(self):
        """Obtener parámetros actuales del filtro"""
        filter_name = self.filter_combo.currentData()
        if not filter_name:
            return {}
        
        params = {}
        
        if filter_name == 'Gaussiano':
            if hasattr(self, 'kernel_spin'):
                params['kernel_size'] = self.kernel_spin.value()
            if hasattr(self, 'sigma_spin'):
                params['sigma'] = self.sigma_spin.value()
                
        elif filter_name in ['Media', 'Mediana']:
            if hasattr(self, 'kernel_spin'):
                params['kernel_size'] = self.kernel_spin.value()
                
        elif filter_name in ['Sobel', 'Prewitt']:
            if hasattr(self, 'dir_combo'):
                params['direccion'] = self.dir_combo.currentText()
                
        elif filter_name == 'Laplaciano':
            if hasattr(self, 'type_combo'):
                params['tipo'] = self.type_combo.currentText()
        
        return params
    
    def on_filter_changed(self):
        """Manejar cambio de filtro"""
        filter_name = self.filter_combo.currentData()
        
        if filter_name:
            # Mostrar información del filtro
            info = self.processor.get_info_filtro(filter_name)
            info_text = f"<b>{info.get('tipo', 'N/A')}</b><br>"
            info_text += f"{info.get('descripcion', 'Sin descripción')}<br><br>"
            info_text += f"<b>Uso:</b> {info.get('uso', 'N/A')}<br>"
            info_text += f"<b>Parámetros:</b> {info.get('parametros', 'Ninguno')}"
            self.filter_info.setText(info_text)
            
            # Crear controles de parámetros
            self.create_filter_params(filter_name)
            self.apply_btn.setEnabled(True)
        else:
            self.filter_info.setText("Seleccione un filtro para ver información")
            self.apply_btn.setEnabled(False)
            
            # Limpiar parámetros
            for i in reversed(range(self.params_layout.count())):
                child = self.params_layout.itemAt(i).widget()
                if child:
                    child.deleteLater()
    
    def on_params_changed(self):
        """Manejar cambio de parámetros"""
        # Habilitar aplicar automáticamente cuando cambien parámetros
        if self.filter_combo.currentData():
            self.apply_btn.setText("✨ Aplicar Cambios")
    
    def apply_current_filter(self):
        """Aplicar el filtro actual"""
        filter_name = self.filter_combo.currentData()
        if not filter_name:
            return
        
        params = self.get_current_params()
        
        # Mostrar que se está aplicando
        self.apply_btn.setText("⏳ Aplicando...")
        self.apply_btn.setEnabled(False)
        
        # Usar worker thread para no bloquear UI
        self.filter_worker = FilterWorker(self.current_image, filter_name, params)
        self.filter_worker.filter_applied.connect(self.on_filter_applied)
        self.filter_worker.filter_failed.connect(self.on_filter_failed)
        self.filter_worker.start()
    
    def on_filter_applied(self, filtered_image):
        """Manejar filtro aplicado exitosamente"""
        self.filtered_image = filtered_image
        self.display_image(filtered_image)
        
        # Actualizar controles
        self.apply_btn.setText("✨ Aplicar Filtro")
        self.apply_btn.setEnabled(True)
        self.hist_filtered_btn.setEnabled(True)
        self.compare_btn.setEnabled(True)  # Habilitar comparación con original
        
        # Actualizar histograma automáticamente
        if self.tab_widget.currentIndex() == 1:  # Tab de histograma activo
            self.update_histogram(filtered_image)
    
    def on_filter_failed(self, error_msg):
        """Manejar error al aplicar filtro"""
        self.apply_btn.setText("❌ Error")
        self.apply_btn.setEnabled(True)
        print(f"Error aplicando filtro: {error_msg}")
    
    def show_original(self):
        """Mostrar imagen original"""
        self.is_comparing = False
        self.showing_color_analysis = False
        self.display_image(self.current_image)
        self.compare_btn.setText("⚖️ Comparar Original")
        if self.tab_widget.currentIndex() == 1:  # Tab de histograma activo
            self.update_histogram(self.current_image)
    
    def toggle_comparison(self):
        """Alternar entre imagen actual (filtrada/recortada) y original"""
        if not self.is_comparing:
            # Mostrar original
            self.display_image(self.current_image)
            if self.tab_widget.currentIndex() == 1:
                self.update_histogram(self.current_image)
            
            # Determinar qué tipo de comparación hacer
            if hasattr(self, 'cropped_image'):
                self.compare_btn.setText("🎭 Ver Recorte")
            elif self.filtered_image is not None:
                self.compare_btn.setText("🔄 Ver Filtrada")
            else:
                self.compare_btn.setText("🔄 Ver Procesada")
                
            self.is_comparing = True
        else:
            # Mostrar imagen procesada (filtrada o recortada)
            processed_image = None
            
            if hasattr(self, 'cropped_image'):
                processed_image = self.cropped_image
            elif self.filtered_image is not None:
                processed_image = self.filtered_image
            
            if processed_image is not None:
                self.display_image(processed_image)
                if self.tab_widget.currentIndex() == 1:
                    self.update_histogram(processed_image)
            
            self.compare_btn.setText("⚖️ Ver Original")
            self.is_comparing = False
    
    def show_color_analysis(self):
        """Mostrar análisis de color de comparación"""
        if self.comparison_path and os.path.exists(self.comparison_path):
            comparison_image = cv2.imread(self.comparison_path)
            if comparison_image is not None:
                comparison_rgb = cv2.cvtColor(comparison_image, cv2.COLOR_BGR2RGB)
                self.display_image(comparison_rgb)
                if self.tab_widget.currentIndex() == 1:
                    self.update_histogram(comparison_rgb)
                self.showing_color_analysis = True
    
    def load_initial_image(self):
        """Cargar y mostrar imagen inicial"""
        self.display_image(self.current_image)
        self.update_histogram(self.current_image)
        self.update_image_info()
    
    def display_image(self, image):
        """Mostrar imagen en el label"""
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        
        self.original_pixmap = QPixmap.fromImage(q_image)
        self.on_zoom_changed(self.zoom_slider.value())
    
    def on_zoom_changed(self, value):
        """Manejar cambio de zoom"""
        self.zoom_label.setText(f"{value}%")
        if hasattr(self, 'original_pixmap'):
            new_width = int(self.original_pixmap.width() * value / 100)
            new_height = int(self.original_pixmap.height() * value / 100)
            scaled_pixmap = self.original_pixmap.scaled(
                new_width, new_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
    
    def change_zoom(self, delta):
        """Cambiar zoom por incremento"""
        new_value = max(25, min(300, self.zoom_slider.value() + delta))
        self.zoom_slider.setValue(new_value)
    
    def update_histogram(self, image):
        """Actualizar histograma RGB"""
        if image is None:
            return
            
        try:
            self.hist_figure.clear()
            ax = self.hist_figure.add_subplot(111)
            
            # Calcular histogramas
            hist_b, hist_g, hist_r = self.processor.calcular_histograma_rgb(
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            )
            
            # Plotear histogramas con transparencia
            ax.plot(hist_r, color='red', alpha=0.7, linewidth=2, label='Canal Rojo')
            ax.plot(hist_g, color='green', alpha=0.7, linewidth=2, label='Canal Verde')  
            ax.plot(hist_b, color='blue', alpha=0.7, linewidth=2, label='Canal Azul')
            
            # Configurar gráfico
            ax.set_xlim([0, 256])
            ax.set_xlabel('Intensidad de Color (0-255)', fontsize=11)
            ax.set_ylabel('Número de Píxeles', fontsize=11)
            ax.set_title('Histograma RGB', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('#f8f8f8')
            
            # Estadísticas
            mean_r, mean_g, mean_b = np.mean(image[:,:,0]), np.mean(image[:,:,1]), np.mean(image[:,:,2])
            stats_text = f"Promedios - R: {mean_r:.1f}, G: {mean_g:.1f}, B: {mean_b:.1f}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            self.hist_figure.tight_layout()
            self.hist_canvas.draw()
            
        except Exception as e:
            print(f"Error actualizando histograma: {e}")
    
    def update_image_info(self):
        """Actualizar información de la imagen"""
        if self.original_image is not None:
            height, width = self.original_image.shape[:2]
            size_kb = os.path.getsize(self.image_path) / 1024
            
            info_text = f"<b>Archivo:</b> {os.path.basename(self.image_path)}<br>"
            info_text += f"<b>Dimensiones:</b> {width} × {height} px<br>"
            info_text += f"<b>Tamaño:</b> {size_kb:.1f} KB<br>"
            info_text += f"<b>Canales:</b> {self.original_image.shape[2] if len(self.original_image.shape) > 2 else 1}"
            
            self.image_info.setText(info_text)
    
    def show_mask_crop_options(self):
        """Mostrar opciones para recortar con máscaras"""
        from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QCheckBox, 
                                     QRadioButton, QButtonGroup, QPushButton, QGroupBox)
        
        options_dialog = QDialog(self)
        options_dialog.setWindowTitle("🎭 Recortar con Máscaras de Afectación")
        options_dialog.resize(400, 350)
        
        layout = QVBoxLayout(options_dialog)
        
        # Título
        title = QLabel("🎭 Configurar Recorte con Máscaras")
        title.setFont(QFont("", 12, QFont.Weight.Bold))
        title.setStyleSheet("color: #000000; padding: 10px; background-color: #e8f5e8; border-radius: 5px;")
        layout.addWidget(title)
        
        # Descripción
        desc = QLabel("Seleccione las máscaras para crear un recorte de las zonas específicas:")
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #333; padding: 5px; font-style: italic;")
        layout.addWidget(desc)
        
        # Grupo de máscaras
        masks_group = QGroupBox("Seleccionar Máscaras para Recorte")
        masks_layout = QVBoxLayout(masks_group)
        
        self.mask_healthy = QCheckBox("🟢 Zona Saludable (Verde)")
        self.mask_healthy.setChecked(True)
        self.mask_healthy.setStyleSheet("color: #000000; font-weight: bold;")
        masks_layout.addWidget(self.mask_healthy)
        
        self.mask_disease1 = QCheckBox("🟡 Afectación Leve (Amarillo)")
        self.mask_disease1.setChecked(False) 
        self.mask_disease1.setStyleSheet("color: #000000; font-weight: bold;")
        masks_layout.addWidget(self.mask_disease1)
        
        self.mask_disease2 = QCheckBox("🔴 Afectación Severa (Rojo)")
        self.mask_disease2.setChecked(False)
        self.mask_disease2.setStyleSheet("color: #000000; font-weight: bold;")
        masks_layout.addWidget(self.mask_disease2)
        
        layout.addWidget(masks_group)
        
        # Grupo de color de fondo
        bg_group = QGroupBox("Color de Fondo para Recorte")
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
        
        self.bg_transparent = QRadioButton("🔳 Transparente (solo máscaras)")
        self.bg_transparent.setStyleSheet("color: #000000; font-weight: bold;")
        self.bg_button_group.addButton(self.bg_transparent)
        bg_layout.addWidget(self.bg_transparent)
        
        layout.addWidget(bg_group)
        
        # Botones
        buttons_layout = QHBoxLayout()
        
        apply_btn = QPushButton("✂️ Crear Recorte")
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
        apply_btn.clicked.connect(lambda: self.create_mask_crop(options_dialog))
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
    
    def create_mask_crop(self, dialog):
        """Crear recorte usando las máscaras seleccionadas"""
        try:
            # Cerrar diálogo
            dialog.close()
            
            # Verificar qué máscaras están seleccionadas
            masks_selected = []
            if self.mask_healthy.isChecked():
                masks_selected.append('healthy')
            if self.mask_disease1.isChecked():
                masks_selected.append('disease1')
            if self.mask_disease2.isChecked():
                masks_selected.append('disease2')
            
            if not masks_selected:
                QMessageBox.warning(self, "Advertencia", "Seleccione al menos una máscara")
                return
            
            # Determinar color de fondo
            if self.bg_white.isChecked():
                bg_color = [255, 255, 255]
                bg_name = "blanco"
            elif self.bg_black.isChecked():
                bg_color = [0, 0, 0]
                bg_name = "negro"
            else:
                bg_color = None
                bg_name = "transparente"
            
            # Crear recorte con máscara
            cropped_image = self.apply_mask_crop(self.current_image, masks_selected, bg_color)
            
            if cropped_image is not None:
                # Mostrar imagen recortada
                self.display_image(cropped_image)
                
                # Actualizar histograma
                if self.tab_widget.currentIndex() == 1:
                    self.update_histogram(cropped_image)
                
                # Habilitar comparación
                self.compare_btn.setEnabled(True)
                self.compare_btn.setText("⚖️ Ver Original")
                
                # Guardar imagen recortada para comparación
                self.cropped_image = cropped_image
                
                # Mostrar información
                masks_text = ", ".join([
                    m.replace('healthy', 'Sano')
                     .replace('disease1', 'Afectación Leve')
                     .replace('disease2', 'Afectación Severa') 
                    for m in masks_selected
                ])
                
                # Verificar si el recorte se realizó correctamente
                method = "análisis HSV (mismos rangos que el análisis de color)"
                print(f"✅ Recorte creado usando {method}")
                print(f"   Zonas: {masks_text}")
                print(f"   Fondo: {bg_name}")
                
                # Actualizar información de la imagen
                info_text = f"<b>🎭 Recorte con Máscaras HSV</b><br>"
                info_text += f"<b>Método:</b> {method}<br>"
                info_text += f"<b>Zonas:</b> {masks_text}<br>"
                info_text += f"<b>Fondo:</b> {bg_name}<br>"
                info_text += f"<b>Rangos HSV:</b><br>"
                info_text += f"  • Verde: [40,40,40] - [80,255,255]<br>"
                info_text += f"  • Amarillo: [15,40,40] - [35,255,255]<br>"
                info_text += f"  • Marrón: [0,0,0] - [15,255,100]"
                self.image_info.setText(info_text)
                
            else:
                QMessageBox.warning(self, "Error", "No se pudo crear el recorte con máscara")
                print("❌ Error: No se pudo crear el recorte")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error creando recorte: {str(e)}")
            print(f"Error en create_mask_crop: {e}")
    
    def apply_mask_crop(self, image, masks_selected, bg_color):
        """Aplicar recorte usando EXACTAMENTE las mismas máscaras del análisis de color"""
        try:
            # ⚠️ IMPORTANTE: Usar exactamente la misma conversión que analysis_tab.py
            # En analysis_tab.py:
            # 1. Lee imagen → BGR 
            # 2. Convierte BGR → RGB (para mostrar)
            # 3. Convierte BGR → HSV (para análisis) 
            
            # Aquí tenemos imagen RGB, así que:
            # 1. RGB → BGR (para simular lo que hace analysis_tab.py)
            # 2. BGR → HSV (EXACTAMENTE igual que analysis_tab.py)
            
            print(f"DEBUG: Imagen de entrada shape: {image.shape}")
            print(f"DEBUG: Imagen de entrada dtype: {image.dtype}")
            
            # Convertir RGB → BGR (como si fuera la imagen original leída)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Convertir BGR → HSV (EXACTAMENTE igual que analysis_tab.py)
            hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            
            print(f"DEBUG: HSV shape: {hsv.shape}")
            print(f"DEBUG: HSV dtype: {hsv.dtype}")
            print(f"DEBUG: HSV min/max: {hsv.min()}/{hsv.max()}")
            
            # USAR EXACTAMENTE LOS MISMOS RANGOS que analysis_tab.py
            # Verde saludable
            lower_healthy = np.array([40, 40, 40])
            upper_healthy = np.array([80, 255, 255])
            
            # Amarillo/marrón (enfermedad leve) 
            lower_disease1 = np.array([15, 40, 40])
            upper_disease1 = np.array([35, 255, 255])
            
            # Marrón oscuro/negro (necrosis severa)
            lower_disease2 = np.array([0, 0, 0])
            upper_disease2 = np.array([15, 255, 100])
            
            # Crear máscaras individuales (EXACTAMENTE igual que analysis_tab.py)
            mask_healthy_cv = cv2.inRange(hsv, lower_healthy, upper_healthy)
            mask_disease1_cv = cv2.inRange(hsv, lower_disease1, upper_disease1)
            mask_disease2_cv = cv2.inRange(hsv, lower_disease2, upper_disease2)
            
            print(f"DEBUG: Máscaras generadas (EXACTAS)")
            print(f"  - Healthy pixels: {np.sum(mask_healthy_cv > 0)}")
            print(f"  - Disease1 pixels: {np.sum(mask_disease1_cv > 0)}")
            print(f"  - Disease2 pixels: {np.sum(mask_disease2_cv > 0)}")
            
            # Combinar máscaras seleccionadas SIN operaciones morfológicas
            combined_mask = np.zeros_like(mask_healthy_cv)
            
            for mask_type in masks_selected:
                if mask_type == 'healthy':
                    combined_mask = cv2.bitwise_or(combined_mask, mask_healthy_cv)
                    print(f"  + Agregada máscara healthy: {np.sum(mask_healthy_cv > 0)} píxeles")
                elif mask_type == 'disease1':
                    combined_mask = cv2.bitwise_or(combined_mask, mask_disease1_cv)
                    print(f"  + Agregada máscara disease1: {np.sum(mask_disease1_cv > 0)} píxeles")
                elif mask_type == 'disease2':
                    combined_mask = cv2.bitwise_or(combined_mask, mask_disease2_cv)
                    print(f"  + Agregada máscara disease2: {np.sum(mask_disease2_cv > 0)} píxeles")
            
            print(f"  - Combined mask pixels ANTES de morfología: {np.sum(combined_mask > 0)}")
            
            # ⚠️ TEMPORAL: Sin operaciones morfológicas para ver si es eso lo que causa el problema
            # Si funciona bien sin morfología, luego podemos hacer operaciones más suaves
            
            print(f"  - Combined mask pixels DESPUÉS (sin morfología): {np.sum(combined_mask > 0)}")
            
            # Aplicar máscara EXACTAMENTE como debe ser
            if bg_color is None:
                # Fondo transparente/gris
                result = image.copy()
                result[combined_mask == 0] = [128, 128, 128]  # Gris para zonas no seleccionadas
            else:
                # Fondo de color específico  
                result = np.full_like(image, bg_color, dtype=np.uint8)
                # ⚠️ CLAVE: Usar combined_mask == 255, no > 0
                result[combined_mask == 255] = image[combined_mask == 255]
            
            if np.sum(combined_mask > 0) == 0:
                print("❌ ERROR: No se encontraron píxeles en las máscaras seleccionadas")
                print("   Posibles causas:")
                print("   1. Los rangos HSV no detectan esta imagen")
                print("   2. Problema en la conversión RGB↔BGR↔HSV")
                return None
            else:
                print(f"✅ Máscara aplicada correctamente: {np.sum(combined_mask > 0)} píxeles procesados")
            
            return result
            
        except Exception as e:
            print(f"❌ Error aplicando máscara: {e}")
            import traceback
            traceback.print_exc()
            return None
