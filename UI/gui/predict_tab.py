"""
Pestaña de predicción con modelos YOLO
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QPushButton, QLabel, QComboBox, QDoubleSpinBox, 
                             QProgressBar, QTextEdit, QFormLayout, QCheckBox,
                             QFileDialog, QMessageBox, QSplitter, QFrame,
                             QScrollArea)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

import os
from pathlib import Path
from utils.config import config
from utils.yolo_utils import YOLOProcessor

class PredictionWorker(QThread):
    """Worker thread para predicción"""
    
    progress_update = pyqtSignal(str)
    prediction_completed = pyqtSignal(bool, str)
    
    def __init__(self, source_path, model_path, conf, save_crops, classes):
        super().__init__()
        self.source_path = source_path
        self.model_path = model_path
        self.conf = conf
        self.save_crops = save_crops
        self.classes = classes
        self.yolo_processor = YOLOProcessor()
    
    def run(self):
        """Ejecutar predicción"""
        try:
            self.progress_update.emit("Iniciando predicción...")
            
            results_path = self.yolo_processor.predict_images(
                self.source_path, self.model_path, self.conf, 
                self.save_crops, self.classes
            )
            
            if results_path:
                self.progress_update.emit("¡Predicción completada exitosamente!")
                self.prediction_completed.emit(True, str(results_path))
            else:
                self.progress_update.emit("Error durante la predicción")
                self.prediction_completed.emit(False, "")
                
        except Exception as e:
            self.progress_update.emit(f"Error: {str(e)}")
            self.prediction_completed.emit(False, "")

class PredictTab(QWidget):
    """Pestaña para predicción con modelos"""
    
    prediction_completed = pyqtSignal(bool, str)
    
    def __init__(self):
        super().__init__()
        self.yolo_processor = YOLOProcessor()
        self.prediction_worker = None
        self.current_source_path = ""
        self.setup_ui()
        self.load_existing_predictions()
    
    def setup_ui(self):
        """
        CONFIGURAR INTERFAZ DE PREDICCIÓN
        EDITAR AQUÍ: Layout principal, distribución de paneles
        """
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)  # Márgenes reducidos
        layout.setSpacing(5)
        
        # SPLITTER PRINCIPAL - Divide la pantalla en dos paneles
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # PANEL IZQUIERDO - Controles y configuración (más estrecho)
        left_panel = self.create_config_panel()
        splitter.addWidget(left_panel)
        
        # PANEL DERECHO - Progreso y resultados (más ancho)
        right_panel = self.create_results_panel()
        splitter.addWidget(right_panel)
        
        # PROPORCIONES - Ajustadas para pantallas pequeñas
        # 300px para controles, resto para resultados
        splitter.setSizes([300, 800])
    
    def create_config_panel(self):
        """
        PANEL DE CONFIGURACIÓN - Controles del lado izquierdo
        EDITAR AQUÍ: Controles de predicción, parámetros, botones
        """
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        panel.setMaximumWidth(320)  # Ancho máximo para pantallas pequeñas
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)  # Espaciado reducido
        
        # TÍTULO DE LA SECCIÓN
        title = QLabel("🔍 Configuración de Predicción")
        title_font = QFont()
        title_font.setPointSize(12)  # Tamaño reducido
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #000000; background-color: #e8f5e8; padding: 8px; border-radius: 4px; font-weight: bold;")  # EDITAR COLOR TÍTULO - FIJO PARA MODO OSCURO
        layout.addWidget(title)
        
        # GRUPO: SELECCIÓN DE MODELO
        model_group = QGroupBox("Selección de Modelo")
        model_group.setMaximumHeight(100)  # Altura limitada
        model_layout = QFormLayout(model_group)
        model_layout.setSpacing(5)  # Espaciado reducido
        
        self.model_combo = QComboBox()
        self.model_combo.setMinimumHeight(25)  # Altura mínima para visibilidad
        self.load_available_models()
        model_layout.addRow("Modelo:", self.model_combo)
        
        refresh_models_btn = QPushButton("🔄 Actualizar")
        refresh_models_btn.setMaximumHeight(30)  # Altura controlada
        refresh_models_btn.clicked.connect(self.load_available_models)
        model_layout.addRow("", refresh_models_btn)
        
        layout.addWidget(model_group)
        
        # GRUPO: ORIGEN DE IMÁGENES
        source_group = QGroupBox("Origen de Imágenes")
        source_group.setMaximumHeight(120)  # Altura controlada
        source_layout = QVBoxLayout(source_group)
        source_layout.setSpacing(5)
        
        # BOTONES DE SELECCIÓN - Opciones para elegir fuente
        source_buttons_layout = QHBoxLayout()
        source_buttons_layout.setSpacing(3)  # Espaciado reducido
        
        # Botones más pequeños para pantallas pequeñas
        self.use_test_images_btn = QPushButton("📁 Test")
        self.use_test_images_btn.setMaximumHeight(30)
        self.use_test_images_btn.clicked.connect(self.use_test_images)
        
        self.select_folder_btn = QPushButton("📂 Carpeta")
        self.select_folder_btn.setMaximumHeight(30)
        self.select_folder_btn.clicked.connect(self.select_image_folder)
        
        self.select_image_btn = QPushButton("🖼️ Imagen")
        self.select_image_btn.setMaximumHeight(30)
        self.select_image_btn.clicked.connect(self.select_single_image)
        
        source_buttons_layout.addWidget(self.use_test_images_btn)
        source_buttons_layout.addWidget(self.select_folder_btn)
        source_buttons_layout.addWidget(self.select_image_btn)
        
        source_layout.addLayout(source_buttons_layout)
        
        # RUTA SELECCIONADA - Mostrar fuente elegida
        self.source_path_label = QLabel("Ninguna fuente seleccionada")
        self.source_path_label.setWordWrap(True)
        self.source_path_label.setMaximumHeight(40)  # Altura limitada
        self.source_path_label.setStyleSheet("""
            color: #000000; 
            background-color: #ffffff;
            padding: 4px; 
            border: 1px solid #cccccc; 
            border-radius: 3px;
            font-size: 8pt;
        """)  # EDITAR ESTILO RUTA AQUÍ
        source_layout.addWidget(self.source_path_label)
        
        layout.addWidget(source_group)
        
        # GRUPO: PARÁMETROS DE PREDICCIÓN
        params_group = QGroupBox("Parámetros")
        params_group.setMaximumHeight(100)  # Altura compacta
        params_layout = QFormLayout(params_group)
        params_layout.setSpacing(5)
        
        # Control de confianza
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.1, 1.0)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        self.conf_spin.setMinimumHeight(25)
        params_layout.addRow("Confianza:", self.conf_spin)
        
        # CHECKBOX PARA GUARDAR RECORTES - Opción para generar crops de detecciones
        self.save_crops_check = QCheckBox("Guardar recortes de detecciones")
        self.save_crops_check.setChecked(True)
        self.save_crops_check.setStyleSheet("""
            QCheckBox {
                font-size: 10pt;
                padding: 5px;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
        """)  # EDITAR ESTILO CHECKBOX RECORTES
        params_layout.addRow("Opciones:", self.save_crops_check)
        
        layout.addWidget(params_group)
        
        # GRUPO: FILTRAR POR CLASES
        classes_group = QGroupBox("Filtrar por Clases")
        classes_group.setMaximumHeight(150)  # Altura controlada
        classes_layout = QVBoxLayout(classes_group)
        classes_layout.setSpacing(3)
        
        # Scroll area para clases si son muchas
        scroll_area = QScrollArea()
        scroll_area.setMaximumHeight(100)
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(2)
        
        # CHECKBOXES DE CLASES - Una por cada clase del modelo
        self.class_checkboxes = {}
        for class_id, class_name in config.CLASSES.items():
            checkbox = QCheckBox(f"{class_name} (ID: {class_id})")
            checkbox.setStyleSheet("font-size: 8pt;")  # EDITAR TAMAÑO FUENTE
            self.class_checkboxes[class_id] = checkbox
            scroll_layout.addWidget(checkbox)
        
        scroll_area.setWidget(scroll_widget)
        classes_layout.addWidget(scroll_area)
        
        # BOTONES DE SELECCIÓN DE CLASES
        class_buttons_layout = QHBoxLayout()
        class_buttons_layout.setSpacing(3)
        
        select_all_btn = QPushButton("Todas")
        select_all_btn.setMaximumHeight(25)
        select_all_btn.clicked.connect(self.select_all_classes)
        
        select_none_btn = QPushButton("Ninguna")
        select_none_btn.setMaximumHeight(25)
        select_none_btn.clicked.connect(self.select_no_classes)
        
        class_buttons_layout.addWidget(select_all_btn)
        class_buttons_layout.addWidget(select_none_btn)
        
        classes_layout.addLayout(class_buttons_layout)
        layout.addWidget(classes_group)
        
        # BOTONES PRINCIPALES - Acciones de predicción
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(5)
        
        self.predict_btn = QPushButton("🚀 Predecir")  # Texto más corto
        self.predict_btn.setMinimumHeight(35)  # Altura mínima para visibilidad
        self.predict_btn.clicked.connect(self.start_prediction)
        
        self.stop_btn = QPushButton("⏹️ Parar")  # Texto más corto
        self.stop_btn.setMinimumHeight(35)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_prediction)
        
        buttons_layout.addWidget(self.predict_btn)
        buttons_layout.addWidget(self.stop_btn)
        
        layout.addLayout(buttons_layout)
        layout.addStretch()  # Empujar todo hacia arriba
        
        return panel
    
    def create_results_panel(self):
        """
        PANEL DE RESULTADOS - Área derecha para mostrar progreso y resultados
        EDITAR AQUÍ: Disposición de resultados, logs, historial
        """
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # TÍTULO DEL PANEL
        title = QLabel("📊 Resultados de Predicción")
        title_font = QFont()
        title_font.setPointSize(12)  # Tamaño reducido
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #000000; background-color: #e8f5e8; padding: 8px; border-radius: 4px; font-weight: bold;")  # EDITAR COLOR - FIJO
        layout.addWidget(title)
        
        # BARRA DE PROGRESO - Mostrar progreso de operaciones
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumHeight(20)  # Altura reducida
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # ESTADO ACTUAL - Información del proceso
        self.status_label = QLabel("Listo para predecir")
        self.status_label.setStyleSheet("""
            font-weight: bold; 
            color: #000000; 
            background-color: #ffffff;
            padding: 3px;
            font-size: 9pt;
        """)  # EDITAR ESTILO STATUS - COLORES FIJOS
        layout.addWidget(self.status_label)
        
        # LOG DE SALIDA - Área de texto para mensajes del sistema
        log_group = QGroupBox("Log de Predicción")
        log_group.setMaximumHeight(300)  # Altura más compacta
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 8))  # Fuente más pequeña
        self.log_text.setStyleSheet("""
            color: #000000;
            background-color: #ffffff;
            border: 1px solid #cccccc;
            border-radius: 3px;
            font-size: 8pt;
        """)  # EDITAR COLOR FONDO - FIJO
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        # PREDICCIONES ANTERIORES - Historial de predicciones
        history_group = QGroupBox("Predicciones Anteriores")
        history_group.setMaximumHeight(120)  # Altura compacta
        history_layout = QVBoxLayout(history_group)
        history_layout.setSpacing(5)
        
        # Combo para seleccionar predicción anterior
        self.history_combo = QComboBox()
        self.history_combo.setMinimumHeight(25)
        self.history_combo.currentTextChanged.connect(self.on_history_selection_changed)
        history_layout.addWidget(self.history_combo)
        
        # BOTONES DE ACCIÓN - Acciones sobre predicciones anteriores
        history_buttons_layout = QHBoxLayout()
        history_buttons_layout.setSpacing(3)
        
        self.view_results_btn = QPushButton("👁️ Ver")  # Texto más corto
        self.view_results_btn.setMaximumHeight(30)
        self.view_results_btn.clicked.connect(self.view_prediction_results)
        
        self.analyze_btn = QPushButton("🔬 Analizar")
        self.analyze_btn.setMaximumHeight(30)
        self.analyze_btn.clicked.connect(self.analyze_predictions)
        
        history_buttons_layout.addWidget(self.view_results_btn)
        history_buttons_layout.addWidget(self.analyze_btn)
        
        history_layout.addLayout(history_buttons_layout)
        layout.addWidget(history_group)
        
        return panel
    
    def load_available_models(self):
        """Cargar modelos disponibles"""
        self.model_combo.clear()
        
        # Modelos pre-entrenados
        for model in config.AVAILABLE_MODELS:
            model_path = config.MODELS_DIR / model
            if model_path.exists():
                self.model_combo.addItem(f"Pre-entrenado: {model}")
        
        # Modelos entrenados
        train_runs = config.get_all_train_runs()
        for run in train_runs:
            best_model = run / "weights" / "best.pt"
            if best_model.exists():
                self.model_combo.addItem(f"Entrenado: {run.name}")
    
    def use_test_images(self):
        """Usar carpeta de test images"""
        if config.TEST_IMAGES_DIR.exists():
            self.current_source_path = str(config.TEST_IMAGES_DIR)
            self.source_path_label.setText(f"📁 {self.current_source_path}")
        else:
            QMessageBox.warning(self, "Error", "Carpeta test_images no encontrada")
    
    def select_image_folder(self):
        """Seleccionar carpeta de imágenes"""
        folder = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de imágenes")
        if folder:
            self.current_source_path = folder
            self.source_path_label.setText(f"📁 {folder}")
    
    def select_single_image(self):
        """Seleccionar una sola imagen"""
        image_file, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar imagen", "",
            "Imágenes (*.jpg *.jpeg *.png *.bmp *.tiff)"
        )
        if image_file:
            self.current_source_path = image_file
            self.source_path_label.setText(f"🖼️ {os.path.basename(image_file)}")
    
    def select_all_classes(self):
        """Seleccionar todas las clases"""
        for checkbox in self.class_checkboxes.values():
            checkbox.setChecked(True)
    
    def select_no_classes(self):
        """Deseleccionar todas las clases"""
        for checkbox in self.class_checkboxes.values():
            checkbox.setChecked(False)
    
    def get_selected_classes(self):
        """Obtener clases seleccionadas"""
        selected = []
        for class_id, checkbox in self.class_checkboxes.items():
            if checkbox.isChecked():
                selected.append(class_id)
        return selected if selected else None
    
    def get_selected_model_path(self):
        """Obtener ruta del modelo seleccionado"""
        selection = self.model_combo.currentText()
        
        if selection.startswith("Pre-entrenado:"):
            model_name = selection.replace("Pre-entrenado: ", "")
            return str(config.MODELS_DIR / model_name)
        elif selection.startswith("Entrenado:"):
            run_name = selection.replace("Entrenado: ", "")
            run_path = config.RUNS_DIR / run_name
            return str(run_path / "weights" / "best.pt")
        
        return None
    
    def start_prediction(self):
        """Iniciar predicción"""
        if not self.current_source_path:
            QMessageBox.warning(self, "Error", "Seleccione una fuente de imágenes")
            return
        
        model_path = self.get_selected_model_path()
        if not model_path or not Path(model_path).exists():
            QMessageBox.warning(self, "Error", "Modelo seleccionado no encontrado")
            return
        
        # Configurar UI
        self.predict_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.log_text.clear()
        
        # Obtener parámetros
        conf = self.conf_spin.value()
        save_crops = self.save_crops_check.isChecked()
        selected_classes = self.get_selected_classes()
        
        # Crear y iniciar worker
        self.prediction_worker = PredictionWorker(
            self.current_source_path, model_path, conf, save_crops, selected_classes
        )
        
        self.prediction_worker.progress_update.connect(self.update_progress)
        self.prediction_worker.prediction_completed.connect(self.on_prediction_completed)
        
        self.prediction_worker.start()
    
    def stop_prediction(self):
        """Detener predicción"""
        if self.prediction_worker and self.prediction_worker.isRunning():
            self.prediction_worker.terminate()
            self.prediction_worker.wait()
        
        self.reset_ui_after_prediction()
        self.update_progress("Predicción detenida por el usuario")
    
    def update_progress(self, message):
        """Actualizar progreso"""
        self.status_label.setText(message)
        self.log_text.append(f"{message}\n")
    
    def on_prediction_completed(self, success, results_path):
        """Manejar completación de predicción"""
        self.reset_ui_after_prediction()
        
        if success:
            self.update_progress("✅ Predicción completada exitosamente")
            self.load_existing_predictions()
        else:
            self.update_progress("❌ Error durante la predicción")
        
        # Emitir señal para ventana principal
        self.prediction_completed.emit(success, results_path)
    
    def reset_ui_after_prediction(self):
        """Resetear UI después de predicción"""
        self.predict_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
    
    def load_existing_predictions(self):
        """Cargar predicciones existentes"""
        self.history_combo.clear()
        
        predict_runs = config.get_all_predict_runs()
        if predict_runs:
            for run in predict_runs:
                self.history_combo.addItem(f"{run.name}")
    
    def on_history_selection_changed(self):
        """Manejar cambio de selección en historial"""
        pass
    
    def view_prediction_results(self):
        """Ver resultados de predicción seleccionada"""
        current_selection = self.history_combo.currentText()
        if current_selection:
            results_path = config.RUNS_DIR / current_selection
            if results_path.exists():
                # Abrir carpeta de resultados
                import subprocess
                subprocess.Popen(f'explorer "{results_path}"')
    
    def analyze_predictions(self):
        """Analizar predicciones (ir a la pestaña de análisis)"""
        current_selection = self.history_combo.currentText()
        if current_selection:
            results_path = config.RUNS_DIR / current_selection
            # Aquí podrías emitir una señal para cambiar a la pestaña de análisis
            QMessageBox.information(self, "Análisis", 
                                   f"Cambiando a análisis para: {current_selection}")
