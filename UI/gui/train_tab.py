"""
Pestaña de entrenamiento de modelos YOLO
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QPushButton, QLabel, QComboBox, QSpinBox, 
                             QProgressBar, QTextEdit, QFormLayout, QMessageBox, QSplitter, QFrame)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

import os
import subprocess
import time
from pathlib import Path
from datetime import datetime

class TrainingWorker(QThread):
    """Worker thread para entrenamiento YOLO"""
    
    progress_update = pyqtSignal(str)
    training_completed = pyqtSignal(bool, str)
    
    def __init__(self, model_name, epochs, imgsz, batch, data_path, output_dir):
        super().__init__()
        self.model_name = model_name  # Nombre del archivo (ej: yolov9s.pt)
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.data_path = data_path    # Ruta al data.yaml
        self.output_dir = output_dir  # Directorio de salida
        self.should_stop = False
        self.process = None
    
    def stop(self):
        """Detener entrenamiento"""
        self.should_stop = True
        if self.process:
            self.process.terminate()
        
    def run(self):
        """Ejecutar entrenamiento con múltiples métodos de fallback"""
        try:
            # Cambiar al directorio correcto para rutas relativas
            base_dir = Path(self.data_path).parent.parent  # My-First-Project-3 -> content
            os.chdir(str(base_dir))
            
            self.progress_update.emit(f"� Cambiando al directorio: {base_dir}")
            
            # Método 1: Intentar con CLI de YOLO (como en Jupyter)
            success = self._try_yolo_cli_method()
            
            if not success:
                # Método 2: Fallback con ultralytics Python
                self.progress_update.emit("🔄 CLI falló, intentando método Python...")
                success = self._try_python_method()
            
            if success:
                self.progress_update.emit("✅ Entrenamiento completado exitosamente!")
                
                # Buscar el modelo entrenado
                possible_paths = [
                    Path(self.output_dir) / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}" / "weights" / "best.pt",
                    Path("runs") / "detect" / "train" / "weights" / "best.pt",
                    Path("runs") / "detect" / "train2" / "weights" / "best.pt",
                    Path("runs") / "detect" / "train3" / "weights" / "best.pt"
                ]
                
                model_path = None
                for path in possible_paths:
                    if path.exists():
                        model_path = str(path)
                        break
                
                self.training_completed.emit(True, model_path or "Entrenamiento completado")
            else:
                self.progress_update.emit("❌ Todos los métodos de entrenamiento fallaron")
                self.training_completed.emit(False, "Error: No se pudo completar el entrenamiento")
                
        except Exception as e:
            error_msg = f"Error durante entrenamiento: {str(e)}"
            self.progress_update.emit(f"❌ {error_msg}")
            self.training_completed.emit(False, error_msg)
    
    def _try_yolo_cli_method(self):
        """Intentar entrenamiento con CLI de YOLO (método preferido)"""
        try:
            # Construir comando exactamente como en Jupyter
            # !yolo task=detect mode=train model=yolov9s.pt data=My-First-Project-3/data.yaml epochs=125 imgsz=640 plots=True
            cmd = [
                "yolo", "task=detect", "mode=train", 
                f"model={self.model_name}",
                f"data=My-First-Project-3/data.yaml",
                f"epochs={self.epochs}",
                f"imgsz={self.imgsz}",
                f"batch={self.batch}",
                "plots=True"
            ]
            
            self.progress_update.emit(f"🚀 Ejecutando: {' '.join(cmd)}")
            
            # Ejecutar comando
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Procesar salida en tiempo real
            while True:
                if self.should_stop:
                    self.process.terminate()
                    return False
                
                output = self.process.stdout.readline()
                if output == '' and self.process.poll() is not None:
                    break
                
                if output:
                    line = output.strip()
                    if line:
                        self.progress_update.emit(f"📊 {line}")
            
            # Verificar código de salida
            return_code = self.process.poll()
            if return_code == 0:
                return True
            else:
                self.progress_update.emit(f"❌ CLI terminó con código {return_code}")
                return False
                
        except FileNotFoundError:
            self.progress_update.emit("⚠️ CLI 'yolo' no encontrado, probando método alternativo...")
            return False
        except Exception as e:
            self.progress_update.emit(f"⚠️ Error con CLI: {str(e)}")
            return False
    
    def _try_python_method(self):
        """Método fallback usando ultralytics directamente"""
        try:
            from ultralytics import YOLO
            
            self.progress_update.emit("🐍 Usando método Python directo...")
            
            # Cargar modelo
            model = YOLO(self.model_name)
            
            # Entrenar con los mismos parámetros
            results = model.train(
                data="My-First-Project-3/data.yaml",
                epochs=self.epochs,
                imgsz=self.imgsz,
                batch=self.batch,
                plots=True,
                project=self.output_dir,
                name=f"train_python_{int(time.time())}"
            )
            
            return True
            
        except ImportError:
            self.progress_update.emit("❌ ultralytics no está instalado")
            return False
        except Exception as e:
            self.progress_update.emit(f"❌ Error método Python: {str(e)}")
            return False

class TrainTab(QWidget):
    """Pestaña para entrenamiento de modelos"""
    
    training_completed = pyqtSignal(bool, str)
    
    def __init__(self):
        super().__init__()
        self.training_worker = None
        
        # Configuración por defecto (sin dependencia del config)
        self.available_models = ['yolov9n.pt', 'yolov9s.pt', 'yolov9m.pt', 'yolov9l.pt']
        self.default_epochs = 125
        self.default_imgsz = 640
        self.default_batch = 8
        
        self.setup_ui()
        self.load_existing_models()
    
    def setup_ui(self):
        """
        CONFIGURAR INTERFAZ DE ENTRENAMIENTO
        EDITAR AQUÍ: Layout principal, distribución de paneles
        """
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)  # Márgenes reducidos
        layout.setSpacing(5)
        
        # SPLITTER PRINCIPAL - Divide en panel de controles y resultados
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # PANEL IZQUIERDO - Configuración del entrenamiento
        left_panel = self.create_config_panel()
        splitter.addWidget(left_panel)
        
        # PANEL DERECHO - Progreso y logs del entrenamiento
        right_panel = self.create_progress_panel()
        splitter.addWidget(right_panel)
        
        # PROPORCIONES - Adaptadas para pantallas pequeñas
        splitter.setSizes([320, 780])
    
    def create_config_panel(self):
        """
        PANEL DE CONFIGURACIÓN - Controles para configurar entrenamiento
        EDITAR AQUÍ: Parámetros de entrenamiento, selección de modelo
        """
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        panel.setMaximumWidth(340)  # Ancho controlado para pantallas pequeñas
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)  # Espaciado reducido
        
        # TÍTULO DE LA SECCIÓN
        title = QLabel("⚙️ Configuración de Entrenamiento")
        title_font = QFont()
        title_font.setPointSize(12)  # Tamaño reducido para pantallas pequeñas
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #000000; background-color: #e8f5e8; padding: 8px; border-radius: 4px; font-weight: bold;")  # EDITAR COLOR TÍTULO - FIJO
        layout.addWidget(title)
        
        # GRUPO: SELECCIÓN DE MODELO BASE
        model_group = QGroupBox("Selección de Modelo")
        model_group.setMaximumHeight(80)  # Altura compacta
        model_layout = QFormLayout(model_group)
        model_layout.setSpacing(5)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.available_models)
        self.model_combo.setMinimumHeight(25)  # Altura mínima para visibilidad
        model_layout.addRow("Modelo base:", self.model_combo)
        
        layout.addWidget(model_group)
        
        # GRUPO: PARÁMETROS DE ENTRENAMIENTO
        params_group = QGroupBox("Parámetros de Entrenamiento")
        params_group.setMaximumHeight(150)  # Altura controlada para pantallas pequeñas
        params_layout = QFormLayout(params_group)
        params_layout.setSpacing(5)
        
        # ÉPOCAS - Número de iteraciones de entrenamiento
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(self.default_epochs)
        self.epochs_spin.setMinimumHeight(25)
        params_layout.addRow("Épocas:", self.epochs_spin)
        
        # TAMAÑO DE IMAGEN - Resolución para entrenamiento
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(320, 1280)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(self.default_imgsz)
        self.imgsz_spin.setMinimumHeight(25)
        params_layout.addRow("Tamaño imagen:", self.imgsz_spin)
        
        # TAMAÑO DE LOTE - Número de imágenes por iteración
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(self.default_batch)
        self.batch_spin.setMinimumHeight(25)
        params_layout.addRow("Batch size:", self.batch_spin)
        
        layout.addWidget(params_group)
        
        # GRUPO: INFORMACIÓN DEL DATASET
        dataset_group = QGroupBox("Información del Dataset")
        dataset_group.setMaximumHeight(120)  # Altura compacta
        dataset_layout = QVBoxLayout(dataset_group)
        dataset_layout.setSpacing(5)
        
        # ETIQUETA DE INFORMACIÓN - Muestra detalles del dataset cargado
        self.dataset_info = QLabel()
        self.dataset_info.setStyleSheet("""
            QLabel {
                background-color: #ffffff;
                color: #000000;
                padding: 8px;
                border: 1px solid #cccccc;
                border-radius: 4px;
                font-size: 10px;  /* Tamaño compacto para pantallas pequeñas */
            }
        """)  # EDITAR ESTILO DE INFORMACIÓN DEL DATASET - COLORES FIJOS
        self.dataset_info.setWordWrap(True)
        self.update_dataset_info()
        dataset_layout.addWidget(self.dataset_info)
        
        layout.addWidget(dataset_group)
        
        # BOTONES DE CONTROL
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(8)
        
        # BOTÓN INICIAR ENTRENAMIENTO
        self.start_btn = QPushButton("🚀 Iniciar Entrenamiento")
        self.start_btn.setMinimumHeight(35)  # Altura mínima para visibilidad
        self.start_btn.setStyleSheet("""
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
        """)  # EDITAR ESTILO BOTÓN INICIAR
        self.start_btn.clicked.connect(self.start_training)
        
        # BOTÓN DETENER ENTRENAMIENTO
        self.stop_btn = QPushButton("⏹️ Detener")
        self.stop_btn.setMinimumHeight(35)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                font-size: 11px;
                border: none;
                border-radius: 6px;
                padding: 8px;
            }
            QPushButton:hover:enabled {
                background-color: #da190b;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)  # EDITAR ESTILO BOTÓN DETENER
        self.stop_btn.clicked.connect(self.stop_training)
        
        buttons_layout.addWidget(self.start_btn)
        buttons_layout.addWidget(self.stop_btn)
        
        layout.addLayout(buttons_layout)
        layout.addStretch()  # Espacio flexible al final
        
        return panel
    
    def create_progress_panel(self):
        """
        PANEL DE PROGRESO - Muestra el avance del entrenamiento en tiempo real
        EDITAR AQUÍ: Estilos de barra de progreso, métricas mostradas
        """
        panel = QFrame()
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        
        # TÍTULO DE LA SECCIÓN
        title = QLabel("📊 Progreso del Entrenamiento")
        title_font = QFont()
        title_font.setPointSize(12)  # Tamaño reducido para pantallas pequeñas
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet("color: #000000; background-color: #e8f5e8; padding: 8px; border-radius: 4px; font-weight: bold;")  # EDITAR COLOR TÍTULO - FIJO
        layout.addWidget(title)
        
        # BARRA DE PROGRESO PRINCIPAL
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(25)  # Altura visible en pantallas pequeñas
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #cccccc;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                font-size: 10px;  /* Texto compacto */
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 6px;
            }
        """)  # EDITAR ESTILO BARRA DE PROGRESO
        layout.addWidget(self.progress_bar)
        
        # ESTADO ACTUAL - Información del proceso
        self.status_label = QLabel("Listo para entrenar")
        self.status_label.setStyleSheet("""
            font-weight: bold; 
            color: #000000; 
            background-color: #ffffff;
            padding: 3px;
            font-size: 10pt;
        """)  # EDITAR ESTILO STATUS - COLORES FIJOS
        layout.addWidget(self.status_label)
        
        # LOG DE SALIDA - Área de texto para mensajes del sistema
        log_group = QGroupBox("Log de Entrenamiento")
        log_group.setMaximumHeight(300)  # Altura más compacta
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 8))  # Fuente más pequeña
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                color: #000000;
                border: 1px solid #cccccc;
                font-size: 8pt;
            }
        """)  # EDITAR ESTILO LOG - COLORES FIJOS
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        # ENTRENAMIENTOS ANTERIORES - Historial de entrenamientos
        history_group = QGroupBox("Entrenamientos Anteriores")
        history_group.setMaximumHeight(120)  # Altura compacta
        history_layout = QVBoxLayout(history_group)
        history_layout.setSpacing(5)
        
        self.history_combo = QComboBox()
        self.history_combo.setMinimumHeight(25)
        self.history_combo.currentTextChanged.connect(self.on_history_selection_changed)
        history_layout.addWidget(self.history_combo)
        
        # BOTÓN VER RESULTADOS
        self.view_results_btn = QPushButton("📈 Ver Resultados")
        self.view_results_btn.setMinimumHeight(25)
        self.view_results_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-size: 10px;
                border: none;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)  # EDITAR ESTILO BOTÓN RESULTADOS
        self.view_results_btn.clicked.connect(self.view_training_results)
        history_layout.addWidget(self.view_results_btn)
        
        layout.addWidget(history_group)
        
        return panel
    
    def update_dataset_info(self):
        """Actualizar información del dataset"""
        try:
            # Usar rutas relativas
            base_dir = Path(__file__).parent.parent.parent  # UI/gui -> UI -> Segmentacion
            data_yaml_path = base_dir / "content" / "My-First-Project-3" / "data.yaml"
            
            if data_yaml_path.exists():
                # Leer información del data.yaml
                import yaml
                with open(data_yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                
                nc = data.get('nc', 0)
                names = data.get('names', [])
                
                info_text = f"📊 Clases: {nc}\n"
                if names:
                    info_text += f"🏷️ Nombres: {', '.join(names[:3])}{'...' if len(names) > 3 else ''}\n"
                
                # Verificar directorios del dataset
                dataset_dir = base_dir / "content" / "My-First-Project-3"
                train_dir = dataset_dir / 'train' / 'images'
                valid_dir = dataset_dir / 'valid' / 'images'
                
                if train_dir.exists():
                    train_count = len(list(train_dir.glob('*.jpg')) + list(train_dir.glob('*.jpeg')))
                    info_text += f"🏋️ Entrenamiento: {train_count} imágenes\n"
                
                if valid_dir.exists():
                    valid_count = len(list(valid_dir.glob('*.jpg')) + list(valid_dir.glob('*.jpeg')))
                    info_text += f"✅ Validación: {valid_count} imágenes"
                
                self.dataset_info.setText(info_text)
            else:
                self.dataset_info.setText("⚠️ Dataset no encontrado\nVerificar configuración")
        except Exception as e:
            self.dataset_info.setText(f"❌ Error al cargar dataset:\n{str(e)[:50]}...")
    
    def load_existing_models(self):
        """Cargar modelos existentes"""
        self.history_combo.clear()
        
        # Usar rutas relativas
        base_dir = Path(__file__).parent.parent.parent  # UI/gui -> UI -> Segmentacion
        runs_dir = base_dir / "content" / "runs"
        
        if runs_dir.exists():
            train_dirs = []
            for item in runs_dir.iterdir():
                if item.is_dir() and item.name.startswith("train"):
                    train_dirs.append(item)
            
            # Ordenar por fecha de modificación (más recientes primero)
            train_dirs = sorted(train_dirs, key=lambda x: x.stat().st_mtime, reverse=True)
            
            for run_dir in train_dirs:
                self.history_combo.addItem(run_dir.name)
    
    def start_training(self):
        """Iniciar entrenamiento"""
        if self.training_worker and self.training_worker.isRunning():
            QMessageBox.warning(self, "Advertencia", "Ya hay un entrenamiento en progreso")
            return
        
        try:
            # Obtener parámetros
            model_name = self.model_combo.currentText()
            epochs = self.epochs_spin.value()
            imgsz = self.imgsz_spin.value()
            batch = self.batch_spin.value()
            
            # Configurar rutas usando rutas relativas
            base_dir = Path(__file__).parent.parent.parent  # UI/gui -> UI -> Segmentacion
            data_yaml_path = base_dir / "content" / "My-First-Project-3" / "data.yaml"
            output_dir = base_dir / "content" / "runs"
            
            # Verificar que el data.yaml existe
            if not data_yaml_path.exists():
                QMessageBox.critical(self, "Error", f"Dataset no encontrado en:\n{data_yaml_path}")
                return
            
            # Verificar que el modelo existe en content
            model_path = base_dir / "content" / model_name
            if not model_path.exists():
                QMessageBox.critical(self, "Error", f"Modelo no encontrado en:\n{model_path}\n\nAsegúrate de tener {model_name} en la carpeta content/")
                return
            
            # Crear directorio de salida
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Mostrar configuración al usuario antes de entrenar
            config_msg = f"""🎯 Configuración de Entrenamiento:
            
📁 Modelo: {model_name}
📊 Dataset: {data_yaml_path.name}
⚡ Épocas: {epochs}
🖼️ Tamaño imagen: {imgsz}
📦 Batch size: {batch}
💾 Salida: {output_dir}

¿Continuar con el entrenamiento?"""
            
            reply = QMessageBox.question(self, "Confirmar Entrenamiento", config_msg,
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            
            if reply != QMessageBox.StandardButton.Yes:
                return
            
            # Crear worker thread con los parámetros correctos
            self.training_worker = TrainingWorker(
                model_name, epochs, imgsz, batch, 
                str(data_yaml_path), str(output_dir)
            )
            self.training_worker.progress_update.connect(self.update_progress)
            self.training_worker.training_completed.connect(self.on_training_completed)
            
            # Actualizar UI
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.log_text.clear()
            self.status_label.setText("🚀 Iniciando entrenamiento...")
            
            # Iniciar entrenamiento
            self.training_worker.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al iniciar entrenamiento: {str(e)}")
    
    def stop_training(self):
        """Detener entrenamiento"""
        if self.training_worker and self.training_worker.isRunning():
            reply = QMessageBox.question(
                self, "Confirmar", 
                "¿Detener el entrenamiento actual?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self.training_worker.stop()
                self.status_label.setText("⏹️ Deteniendo entrenamiento...")
    
    def update_progress(self, message):
        """Actualizar progreso"""
        self.log_text.append(message)
        self.status_label.setText(message)
        
        # Auto scroll
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)
    
    def on_training_completed(self, success, model_path):
        """Manejar completación del entrenamiento"""
        self.progress_bar.setVisible(False)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        if success:
            self.status_label.setText(f"✅ Entrenamiento completado: {model_path}")
            QMessageBox.information(self, "Éxito", f"Entrenamiento completado exitosamente.\nModelo guardado en: {model_path}")
            self.training_completed.emit(True, model_path)
        else:
            self.status_label.setText(f"❌ Entrenamiento falló: {model_path}")
            QMessageBox.warning(self, "Advertencia", f"Entrenamiento no completado: {model_path}")
        
        # Actualizar lista de modelos
        self.load_existing_models()
    
    def reset_ui_after_training(self):
        """Resetear UI después del entrenamiento"""
        self.progress_bar.setVisible(False)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("Listo para entrenar")
    
    def on_history_selection_changed(self):
        """Manejar cambio de selección en historial"""
        selection = self.history_combo.currentText()
        if selection:
            self.status_label.setText(f"📁 Seleccionado: {selection}")
    
    def view_training_results(self):
        """Ver resultados del entrenamiento seleccionado"""
        selection = self.history_combo.currentText()
        if not selection:
            QMessageBox.information(self, "Información", "Seleccione un entrenamiento del historial")
            return
        
        try:
            # Abrir carpeta de resultados usando rutas relativas
            base_dir = Path(__file__).parent.parent.parent  # UI/gui -> UI -> Segmentacion
            runs_dir = base_dir / "content" / "runs"
            results_path = runs_dir / selection
            
            if results_path.exists():
                os.startfile(str(results_path))  # Windows
            else:
                QMessageBox.warning(self, "Advertencia", "Carpeta de resultados no encontrada")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al abrir resultados: {str(e)}")
