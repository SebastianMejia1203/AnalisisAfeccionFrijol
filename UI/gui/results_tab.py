"""
Pestaña de visualización de resultados y comparaciones
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, 
                             QPushButton, QLabel, QComboBox, QScrollArea,
                             QGridLayout, QFrame, QSplitter, QTabWidget,
                             QTextEdit, QTableWidget, QTableWidgetItem,
                             QHeaderView, QMessageBox, QCheckBox, QFormLayout)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QPainter, QPen, QColor

import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

from utils.config import config

class ClickableImageLabel(QLabel):
    """Label de imagen con funcionalidad de clic"""
    
    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
        self.setFixedSize(200, 150)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid #ddd;
                border-radius: 5px;
                background-color: #f9f9f9;
                cursor: pointer;
            }
            QLabel:hover {
                border: 2px solid #4CAF50;
                background-color: #f0f8ff;
            }
        """)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Cargar imagen
        self.load_image()
    
    def load_image(self):
        """Cargar y mostrar imagen"""
        pixmap = QPixmap(str(self.image_path))
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            self.setPixmap(scaled_pixmap)
            self.setToolTip(f"Clic para ampliar: {self.image_path.name}")
        else:
            self.setText("Error cargando imagen")
    
    def mousePressEvent(self, event):
        """Manejar clic en la imagen"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.open_image_detail()
        super().mousePressEvent(event)
    
    def open_image_detail(self):
        """Abrir imagen en ventana de detalle con zoom"""
        self.create_image_detail_window(str(self.image_path), f"Imagen: {self.image_path.name}")
    
    def create_image_detail_window(self, image_path, title):
        """Crear ventana de detalle de imagen con zoom"""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QScrollArea, QPushButton, QSlider
        
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

class MetricsWidget(QWidget):
    """Widget para mostrar métricas de entrenamiento"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.current_results_path = None
        self.graph_index = 0  # Índice de la gráfica actual
        self.total_graphs = 4  # Total de gráficas disponibles
    
    def setup_ui(self):
        """Configurar interfaz"""
        layout = QVBoxLayout(self)
        
        # Controles de navegación
        nav_layout = QHBoxLayout()
        
        self.prev_btn = QPushButton("◀ Anterior")
        self.prev_btn.clicked.connect(self.show_previous_graph)
        self.prev_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_btn)
        
        self.graph_label = QLabel("Gráfica 1 de 4")
        self.graph_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.graph_label.setStyleSheet("font-weight: bold; padding: 5px; color: black;")
        nav_layout.addWidget(self.graph_label)
        
        self.next_btn = QPushButton("Siguiente ▶")
        self.next_btn.clicked.connect(self.show_next_graph)
        nav_layout.addWidget(self.next_btn)
        
        nav_layout.addStretch()
        
        layout.addLayout(nav_layout)
        
        # Canvas para las gráficas
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Información adicional
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(80)
        self.info_text.setReadOnly(True)
        self.info_text.setStyleSheet("background-color: #f8f8f8; border: 1px solid #ddd; color: #000; font-size: 9pt;")
        layout.addWidget(self.info_text)
    
    def show_previous_graph(self):
        """Mostrar gráfica anterior"""
        if self.graph_index > 0:
            self.graph_index -= 1
            self.update_graph()
            self.update_navigation_buttons()
    
    def show_next_graph(self):
        """Mostrar siguiente gráfica"""
        if self.graph_index < self.total_graphs - 1:
            self.graph_index += 1
            self.update_graph()
            self.update_navigation_buttons()
    
    def update_navigation_buttons(self):
        """Actualizar estado de botones de navegación"""
        self.prev_btn.setEnabled(self.graph_index > 0)
        self.next_btn.setEnabled(self.graph_index < self.total_graphs - 1)
        self.graph_label.setText(f"Gráfica {self.graph_index + 1} de {self.total_graphs}")
    
    def plot_training_metrics(self, results_path):
        """Graficar métricas de entrenamiento"""
        self.current_results_path = results_path
        self.graph_index = 0  # Resetear a la primera gráfica
        self.update_graph()
        self.update_navigation_buttons()
    
    def update_graph(self):
        """Actualizar gráfica actual"""
        if not self.current_results_path:
            return
        
        results_file = Path(self.current_results_path) / "results.csv"
        
        if not results_file.exists():
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No se encontraron métricas de entrenamiento', 
                   ha='center', va='center', transform=ax.transAxes)
            self.canvas.draw()
            return
        
        try:
            import pandas as pd
            df = pd.read_csv(results_file)
            df = df.fillna(0)
            
            self.figure.clear()
            
            # Mostrar gráfica según índice actual
            if self.graph_index == 0:
                self.show_loss_graph(df)
            elif self.graph_index == 1:
                self.show_accuracy_graph(df)
            elif self.graph_index == 2:
                self.show_precision_recall_graph(df)
            elif self.graph_index == 3:
                self.show_learning_rate_graph(df)
            
            self.canvas.draw()
            
        except Exception as e:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f'Error cargando métricas:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            self.canvas.draw()
    
    def show_loss_graph(self, df):
        """Mostrar gráfica de pérdidas"""
        ax = self.figure.add_subplot(111)
        
        if 'train/box_loss' in df.columns:
            ax.plot(df.index, df['train/box_loss'], label='Box Loss (Train)', color='red', linewidth=2)
        if 'train/cls_loss' in df.columns:
            ax.plot(df.index, df['train/cls_loss'], label='Class Loss (Train)', color='blue', linewidth=2)
        if 'val/box_loss' in df.columns:
            ax.plot(df.index, df['val/box_loss'], label='Box Loss (Val)', color='red', linestyle='--', linewidth=2)
        if 'val/cls_loss' in df.columns:
            ax.plot(df.index, df['val/cls_loss'], label='Class Loss (Val)', color='blue', linestyle='--', linewidth=2)
        
        ax.set_title('Pérdidas de Entrenamiento y Validación', fontsize=14, fontweight='bold')
        ax.set_xlabel('Época')
        ax.set_ylabel('Pérdida')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.info_text.setText("Gráfica de pérdidas: Muestra cómo disminuyen los errores del modelo durante el entrenamiento. Las líneas sólidas son entrenamiento, las punteadas son validación.")
    
    def show_accuracy_graph(self, df):
        """Mostrar gráfica de precisión"""
        ax = self.figure.add_subplot(111)
        
        if 'metrics/mAP50' in df.columns:
            ax.plot(df.index, df['metrics/mAP50'], label='mAP@0.5', color='green', linewidth=2)
        if 'metrics/mAP50-95' in df.columns:
            ax.plot(df.index, df['metrics/mAP50-95'], label='mAP@0.5:0.95', color='orange', linewidth=2)
        
        ax.set_title('Precisión Media (mAP)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Época')
        ax.set_ylabel('mAP')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        self.info_text.setText("Precisión Media (mAP): Métrica principal para evaluar la calidad de detección. mAP@0.5 es más permisiva, mAP@0.5:0.95 es más estricta.")
    
    def show_precision_recall_graph(self, df):
        """Mostrar gráfica de precisión y recall"""
        ax = self.figure.add_subplot(111)
        
        if 'metrics/precision' in df.columns:
            ax.plot(df.index, df['metrics/precision'], label='Precisión', color='purple', linewidth=2)
        if 'metrics/recall' in df.columns:
            ax.plot(df.index, df['metrics/recall'], label='Recall', color='brown', linewidth=2)
        
        ax.set_title('Precisión y Recall', fontsize=14, fontweight='bold')
        ax.set_xlabel('Época')
        ax.set_ylabel('Valor')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        self.info_text.setText("Precisión: De todas las detecciones, ¿cuántas fueron correctas? Recall: De todos los objetos reales, ¿cuántos fueron detectados?")
    
    def show_learning_rate_graph(self, df):
        """Mostrar gráfica de tasa de aprendizaje"""
        ax = self.figure.add_subplot(111)
        
        if 'lr/pg0' in df.columns:
            ax.plot(df.index, df['lr/pg0'], label='Learning Rate', color='cyan', linewidth=2)
        
        ax.set_title('Tasa de Aprendizaje', fontsize=14, fontweight='bold')
        ax.set_xlabel('Época')
        ax.set_ylabel('Learning Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.info_text.setText("Tasa de aprendizaje: Controla qué tan grandes son los pasos que toma el modelo al aprender. Valores altos aprenden rápido pero pueden ser inestables.")

class ComparisonWidget(QWidget):
    """Widget para comparar diferentes modelos"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.comparison_data = {}
    
    def setup_ui(self):
        """Configurar interfaz"""
        layout = QVBoxLayout(self)
        
        # Controles de comparación
        controls_layout = QHBoxLayout()
        
        controls_layout.addWidget(QLabel("Comparar modelos:"))
        
        self.model1_combo = QComboBox()
        self.model2_combo = QComboBox()
        
        controls_layout.addWidget(QLabel("Modelo 1:"))
        controls_layout.addWidget(self.model1_combo)
        controls_layout.addWidget(QLabel("Modelo 2:"))
        controls_layout.addWidget(self.model2_combo)
        
        compare_btn = QPushButton("📊 Comparar")
        compare_btn.clicked.connect(self.compare_models)
        controls_layout.addWidget(compare_btn)
        
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Área de resultados
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("Consolas", 10))
        layout.addWidget(self.results_text)
    
    def load_available_models(self):
        """Cargar modelos disponibles para comparación"""
        self.model1_combo.clear()
        self.model2_combo.clear()
        
        train_runs = config.get_all_train_runs()
        for run in train_runs:
            self.model1_combo.addItem(run.name)
            self.model2_combo.addItem(run.name)
    
    def compare_models(self):
        """Comparar dos modelos seleccionados"""
        model1 = self.model1_combo.currentText()
        model2 = self.model2_combo.currentText()
        
        if not model1 or not model2:
            QMessageBox.warning(self, "Error", "Seleccione dos modelos para comparar")
            return
        
        if model1 == model2:
            QMessageBox.warning(self, "Error", "Seleccione dos modelos diferentes")
            return
        
        # Cargar datos de ambos modelos
        model1_data = self.load_model_data(model1)
        model2_data = self.load_model_data(model2)
        
        # Generar comparación
        comparison_text = self.generate_comparison_text(model1, model1_data, model2, model2_data)
        self.results_text.setText(comparison_text)
    
    def load_model_data(self, model_name):
        """Cargar datos de un modelo"""
        model_path = config.RUNS_DIR / model_name
        results_file = model_path / "results.csv"
        
        if not results_file.exists():
            return None
        
        try:
            import pandas as pd
            df = pd.read_csv(results_file)
            
            # Obtener métricas finales
            final_metrics = df.iloc[-1] if len(df) > 0 else None
            
            return {
                "path": model_path,
                "results_df": df,
                "final_metrics": final_metrics,
                "epochs": len(df)
            }
        except:
            return None
    
    def generate_comparison_text(self, name1, data1, name2, data2):
        """Generar texto de comparación"""
        text = f"COMPARACIÓN DE MODELOS\n"
        text += "=" * 50 + "\n\n"
        
        text += f"Modelo 1: {name1}\n"
        text += f"Modelo 2: {name2}\n\n"
        
        if data1 is None or data2 is None:
            text += "Error: No se pudieron cargar los datos de uno o ambos modelos.\n"
            return text
        
        # Comparar métricas finales
        text += "MÉTRICAS FINALES:\n"
        text += "-" * 20 + "\n"
        
        metrics_to_compare = [
            ('train/box_loss', 'Pérdida de Caja'),
            ('train/cls_loss', 'Pérdida de Clase'),
            ('metrics/mAP50', 'mAP@0.5'),
            ('metrics/mAP50-95', 'mAP@0.5:0.95'),
            ('metrics/precision', 'Precisión'),
            ('metrics/recall', 'Recall')
        ]
        
        for metric_key, metric_name in metrics_to_compare:
            if metric_key in data1["final_metrics"] and metric_key in data2["final_metrics"]:
                val1 = data1["final_metrics"][metric_key]
                val2 = data2["final_metrics"][metric_key]
                
                better = "🟢" if val1 > val2 else "🔴"
                if "loss" in metric_key:  # Para las pérdidas, menor es mejor
                    better = "🟢" if val1 < val2 else "🔴"
                
                text += f"{metric_name}:\n"
                text += f"  {name1}: {val1:.4f} {better if (val1 > val2 and 'loss' not in metric_key) or (val1 < val2 and 'loss' in metric_key) else '🔴'}\n"
                text += f"  {name2}: {val2:.4f} {'🟢' if better == '🔴' else '🔴'}\n\n"
        
        # Información adicional
        text += "INFORMACIÓN ADICIONAL:\n"
        text += "-" * 25 + "\n"
        text += f"Épocas entrenadas:\n"
        text += f"  {name1}: {data1['epochs']} épocas\n"
        text += f"  {name2}: {data2['epochs']} épocas\n\n"
        
        return text

class ResultsTab(QWidget):
    """Pestaña para visualización de resultados"""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.load_available_results()
    
    def setup_ui(self):
        """Configurar interfaz de usuario"""
        layout = QVBoxLayout(self)
        
        # Header
        header = QLabel("📈 Visualización de Resultados")
        header_font = QFont()
        header_font.setPointSize(16)
        header_font.setBold(True)
        header.setFont(header_font)
        header.setStyleSheet("color: #000000; background-color: #e8f5e8; padding: 10px; border-radius: 4px; font-weight: bold;")
        layout.addWidget(header)
        
        # Tabs secundarios
        self.result_tabs = QTabWidget()
        layout.addWidget(self.result_tabs)
        
        # Tab 1: Entrenamientos
        self.training_tab = self.create_training_tab()
        self.result_tabs.addTab(self.training_tab, "🏋️ Entrenamientos")
        
        # Tab 2: Predicciones
        self.predictions_tab = self.create_predictions_tab()
        self.result_tabs.addTab(self.predictions_tab, "🔍 Predicciones")
        
        # Tab 3: Comparaciones
        self.comparison_tab = self.create_comparison_tab()
        self.result_tabs.addTab(self.comparison_tab, "⚖️ Comparaciones")
    
    def create_training_tab(self):
        """Crear tab de entrenamientos"""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Panel izquierdo - Controles
        left_panel = QFrame()
        left_panel.setFrameStyle(QFrame.Shape.StyledPanel)
        left_panel.setMaximumWidth(300)
        left_layout = QVBoxLayout(left_panel)
        
        # Selección de entrenamiento
        selection_group = QGroupBox("Seleccionar Entrenamiento")
        selection_layout = QFormLayout(selection_group)
        
        self.training_combo = QComboBox()
        self.training_combo.currentTextChanged.connect(self.on_training_selection_changed)
        selection_layout.addRow("Entrenamiento:", self.training_combo)
        
        refresh_btn = QPushButton("🔄 Actualizar")
        refresh_btn.clicked.connect(self.load_available_results)
        selection_layout.addRow("", refresh_btn)
        
        left_layout.addWidget(selection_group)
        
        # Información del entrenamiento
        info_group = QGroupBox("Información")
        info_layout = QVBoxLayout(info_group)
        
        self.training_info = QLabel("Seleccione un entrenamiento")
        self.training_info.setWordWrap(True)
        self.training_info.setStyleSheet("padding: 10px; color: #000000; background-color: #ffffff; border: 1px solid #cccccc; border-radius: 3px;")
        info_layout.addWidget(self.training_info)
        
        left_layout.addWidget(info_group)
        
        # Botones de acción
        actions_group = QGroupBox("Acciones")
        actions_layout = QVBoxLayout(actions_group)
        
        self.view_folder_btn = QPushButton("📁 Abrir Carpeta")
        self.view_folder_btn.clicked.connect(self.view_training_folder)
        actions_layout.addWidget(self.view_folder_btn)
        
        self.export_model_btn = QPushButton("💾 Exportar Modelo")
        self.export_model_btn.clicked.connect(self.export_model)
        actions_layout.addWidget(self.export_model_btn)
        
        left_layout.addWidget(actions_group)
        left_layout.addStretch()
        
        layout.addWidget(left_panel)
        
        # Panel derecho - Métricas
        self.metrics_widget = MetricsWidget()
        layout.addWidget(self.metrics_widget)
        
        return tab
    
    def create_predictions_tab(self):
        """Crear tab de predicciones"""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Panel izquierdo - Controles
        left_panel = QFrame()
        left_panel.setFrameStyle(QFrame.Shape.StyledPanel)
        left_panel.setMaximumWidth(300)
        left_layout = QVBoxLayout(left_panel)
        
        # Selección de predicción
        selection_group = QGroupBox("Seleccionar Predicción")
        selection_layout = QFormLayout(selection_group)
        
        self.prediction_combo = QComboBox()
        self.prediction_combo.currentTextChanged.connect(self.on_prediction_selection_changed)
        selection_layout.addRow("Predicción:", self.prediction_combo)
        
        left_layout.addWidget(selection_group)
        
        # Información de la predicción
        info_group = QGroupBox("Información")
        info_layout = QVBoxLayout(info_group)
        
        self.prediction_info = QLabel("Seleccione una predicción")
        self.prediction_info.setWordWrap(True)
        self.prediction_info.setStyleSheet("padding: 10px; color: #000000; background-color: #ffffff; border: 1px solid #cccccc; border-radius: 3px;")
        info_layout.addWidget(self.prediction_info)
        
        left_layout.addWidget(info_group)
        
        # Opciones de visualización
        view_group = QGroupBox("Opciones de Visualización")
        view_layout = QVBoxLayout(view_group)
        
        self.show_labels_check = QCheckBox("Mostrar etiquetas")
        self.show_labels_check.setChecked(True)
        self.show_labels_check.stateChanged.connect(self.update_gallery_view)
        view_layout.addWidget(self.show_labels_check)
        
        self.show_conf_check = QCheckBox("Mostrar confianza")
        self.show_conf_check.setChecked(True)
        self.show_conf_check.stateChanged.connect(self.update_gallery_view)
        view_layout.addWidget(self.show_conf_check)
        
        self.show_crops_check = QCheckBox("Mostrar recortes")
        self.show_crops_check.setChecked(False)
        self.show_crops_check.stateChanged.connect(self.update_gallery_view)
        view_layout.addWidget(self.show_crops_check)
        
        left_layout.addWidget(view_group)
        
        # Botones de acción
        actions_group = QGroupBox("Acciones")
        actions_layout = QVBoxLayout(actions_group)
        
        self.view_pred_folder_btn = QPushButton("📁 Abrir Carpeta")
        self.view_pred_folder_btn.clicked.connect(self.view_prediction_folder)
        actions_layout.addWidget(self.view_pred_folder_btn)
        
        self.analyze_pred_btn = QPushButton("🔬 Analizar")
        self.analyze_pred_btn.clicked.connect(self.analyze_prediction)
        actions_layout.addWidget(self.analyze_pred_btn)
        
        left_layout.addWidget(actions_group)
        left_layout.addStretch()
        
        layout.addWidget(left_panel)
        
        # Panel derecho - Galería de imágenes
        right_panel = self.create_image_gallery()
        layout.addWidget(right_panel)
        
        return tab
    
    def create_image_gallery(self):
        """Crear galería de imágenes"""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        self.gallery_widget = QWidget()
        self.gallery_layout = QGridLayout(self.gallery_widget)
        
        scroll_area.setWidget(self.gallery_widget)
        return scroll_area
    
    def create_comparison_tab(self):
        """Crear tab de comparaciones"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Widget de comparación
        self.comparison_widget = ComparisonWidget()
        layout.addWidget(self.comparison_widget)
        
        return tab
    
    def load_available_results(self):
        """Cargar resultados disponibles"""
        # Cargar entrenamientos
        self.training_combo.clear()
        train_runs = config.get_all_train_runs()
        for run in train_runs:
            self.training_combo.addItem(run.name)
        
        # Cargar predicciones
        self.prediction_combo.clear()
        predict_runs = config.get_all_predict_runs()
        for run in predict_runs:
            self.prediction_combo.addItem(run.name)
        
        # Cargar en widget de comparación
        self.comparison_widget.load_available_models()
    
    def on_training_selection_changed(self):
        """Manejar cambio de selección de entrenamiento"""
        selection = self.training_combo.currentText()
        if not selection:
            return
        
        # La ruta correcta ya está en RUNS_DIR que apunta a content/runs/detect
        training_path = config.RUNS_DIR / selection
        
        # Actualizar información
        info_text = f"📁 Entrenamiento: {selection}\n"
        info_text += f"📍 Ubicación: {training_path}\n"
        
        # Verificar archivos importantes
        weights_dir = training_path / "weights"
        if weights_dir.exists():
            best_model = weights_dir / "best.pt"
            last_model = weights_dir / "last.pt"
            
            info_text += "🎯 Modelos disponibles:\n"
            if best_model.exists():
                size_mb = best_model.stat().st_size / (1024 * 1024)
                info_text += f"  • best.pt ({size_mb:.1f} MB)\n"
            if last_model.exists():
                size_mb = last_model.stat().st_size / (1024 * 1024)
                info_text += f"  • last.pt ({size_mb:.1f} MB)\n"
        
        results_file = training_path / "results.csv"
        if results_file.exists():
            info_text += "📊 Métricas disponibles: ✓\n"
            
            # También verificar si hay gráficas de entrenamiento
            graphs_available = []
            possible_graphs = [
                "confusion_matrix.png",
                "F1_curve.png",
                "P_curve.png",
                "R_curve.png",
                "PR_curve.png",
                "results.png"
            ]
            
            for graph_name in possible_graphs:
                graph_file = training_path / graph_name
                if graph_file.exists():
                    graphs_available.append(graph_name)
            
            if graphs_available:
                info_text += f"📈 Gráficas disponibles ({len(graphs_available)}): ✓\n"
                info_text += "  • " + "\n  • ".join(graphs_available[:3])
                if len(graphs_available) > 3:
                    info_text += f"\n  • y {len(graphs_available) - 3} más..."
            else:
                info_text += "📈 Gráficas disponibles: ❌\n"
        else:
            info_text += "📊 Métricas disponibles: ❌\n"
        
        self.training_info.setText(info_text)
        
        # Actualizar gráficas
        self.metrics_widget.plot_training_metrics(training_path)
    
    def on_prediction_selection_changed(self):
        """Manejar cambio de selección de predicción"""
        selection = self.prediction_combo.currentText()
        if not selection:
            return
        
        prediction_path = config.RUNS_DIR / selection
        
        # Actualizar información
        info_text = f"📁 Predicción: {selection}\n"
        info_text += f"📍 Ubicación: {prediction_path}\n"
        
        # Contar imágenes procesadas
        images_dir = prediction_path
        if images_dir.exists():
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            info_text += f"🖼️ Imágenes procesadas: {len(image_files)}\n"
        
        # Verificar crops
        crops_dir = prediction_path / "crops"
        if crops_dir.exists():
            crop_count = 0
            for category_dir in crops_dir.iterdir():
                if category_dir.is_dir():
                    category_crops = len(list(category_dir.glob("*.jpg"))) + len(list(category_dir.glob("*.png")))
                    crop_count += category_crops
            info_text += f"✂️ Recortes generados: {crop_count}\n"
        
        self.prediction_info.setText(info_text)
        
        # Actualizar galería
        self.load_prediction_gallery(prediction_path)
    
    def load_prediction_gallery(self, prediction_path):
        """Cargar galería de imágenes de predicción"""
        # Limpiar galería actual
        for i in reversed(range(self.gallery_layout.count())): 
            child = self.gallery_layout.itemAt(i)
            if child:
                widget = child.widget()
                if widget:
                    widget.setParent(None)
        
        # Buscar imágenes según opciones de visualización
        image_files = []
        
        # Siempre incluir imágenes principales (con predicciones dibujadas)
        main_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            main_images.extend(list(prediction_path.glob(ext)))
        
        # Filtrar por opciones de visualización
        if self.show_labels_check.isChecked() or self.show_conf_check.isChecked():
            image_files.extend(main_images)
        else:
            # Si no se muestran etiquetas ni confianza, mostrar imágenes originales si existen
            image_files.extend(main_images)
        
        # Agregar recortes si está habilitado
        crops_images = []
        if self.show_crops_check.isChecked():
            crops_dir = prediction_path / "crops"
            if crops_dir.exists():
                for category_dir in crops_dir.iterdir():
                    if category_dir.is_dir():
                        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                            crops_images.extend(list(category_dir.glob(ext)))
            
            image_files.extend(crops_images)
        
        # Si no hay opciones seleccionadas o no hay imágenes, mostrar al menos las principales
        if not image_files:
            image_files = main_images
        
        # Mostrar TODAS las imágenes encontradas (sin límite de 24)
        total_images = len(image_files)
        
        if not image_files:
            # Mostrar mensaje si no hay imágenes
            no_images_label = QLabel(f"No se encontraron imágenes en:\n{prediction_path}")
            no_images_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            no_images_label.setStyleSheet("color: #666; font-size: 12px; padding: 20px; background-color: #f0f0f0; border: 1px solid #ccc;")
            self.gallery_layout.addWidget(no_images_label, 0, 0, 1, 4)
            return
        
        # Mostrar información de la galería
        info_label = QLabel(f"Mostrando {total_images} imágenes:")
        info_parts = []
        if len(main_images) > 0:
            info_parts.append(f"{len(main_images)} predicciones")
        if len(crops_images) > 0:
            info_parts.append(f"{len(crops_images)} recortes")
        if info_parts:
            info_label.setText(f"Mostrando {total_images} imágenes: " + ", ".join(info_parts))
        
        info_label.setStyleSheet("font-weight: bold; padding: 5px; color: black; background-color: #e8f5e8;")
        self.gallery_layout.addWidget(info_label, 0, 0, 1, 4)
        
        # Crear grid de imágenes clickeables (4 columnas)
        row, col = 1, 0
        for image_file in image_files:
            # Crear widget de imagen clickeable
            image_widget = ClickableImageLabel(image_file)
            
            # Agregar información del tipo de imagen
            if image_file.parent.name == "crops" or "crops" in str(image_file):
                image_widget.setToolTip(f"Recorte: {image_file.name}\nCategoría: {image_file.parent.name}")
            else:
                image_widget.setToolTip(f"Predicción: {image_file.name}")
            
            self.gallery_layout.addWidget(image_widget, row, col)
            
            # Avanzar posición en grid (4 columnas)
            col += 1
            if col >= 4:
                col = 0
                row += 1
            
            # Agregar título con información del archivo
            image_title = image_file.parent.name if image_file.parent.name != prediction_path.name else ""
            if image_title:
                image_widget.setToolTip(f"Categoría: {image_title}\nArchivo: {image_file.name}\nClic para ampliar")
            
            # Agregar al grid
            self.gallery_layout.addWidget(image_widget, row, col)
            
            col += 1
            if col >= 4:  # 4 columnas
                col = 0
                row += 1
    
    def update_gallery_view(self):
        """Actualizar vista de galería según opciones seleccionadas"""
        # Obtener predicción actual
        current_prediction = self.prediction_combo.currentText()
        if current_prediction:
            prediction_path = config.RUNS_DIR / current_prediction
            if prediction_path.exists():
                self.load_prediction_gallery(prediction_path)
    
    def view_training_folder(self):
        """Abrir carpeta de entrenamiento"""
        selection = self.training_combo.currentText()
        if selection:
            folder_path = config.RUNS_DIR / selection
            if folder_path.exists():
                import subprocess
                subprocess.Popen(f'explorer "{folder_path}"')
    
    def view_prediction_folder(self):
        """Abrir carpeta de predicción"""
        selection = self.prediction_combo.currentText()
        if selection:
            folder_path = config.RUNS_DIR / selection
            if folder_path.exists():
                import subprocess
                subprocess.Popen(f'explorer "{folder_path}"')
    
    def export_model(self):
        """Exportar modelo entrenado"""
        selection = self.training_combo.currentText()
        if not selection:
            QMessageBox.warning(self, "Error", "Seleccione un entrenamiento")
            return
        
        model_path = config.RUNS_DIR / selection / "weights" / "best.pt"
        if not model_path.exists():
            QMessageBox.warning(self, "Error", "Modelo no encontrado")
            return
        
        from PyQt6.QtWidgets import QFileDialog
        import shutil
        
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Exportar modelo", f"{selection}_best.pt",
            "Archivos PyTorch (*.pt)"
        )
        
        if save_path:
            try:
                shutil.copy2(model_path, save_path)
                QMessageBox.information(self, "Éxito", f"Modelo exportado a:\n{save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error exportando modelo:\n{str(e)}")
    
    def analyze_prediction(self):
        """Analizar predicción seleccionada - cambiar a pestaña de análisis"""
        selection = self.prediction_combo.currentText()
        if selection:
            # Emitir señal para que la ventana principal cambie a la pestaña de análisis
            # y seleccione esta predicción
            try:
                # Obtener la ventana principal y cambiar a la pestaña de análisis
                main_window = self.window()
                if hasattr(main_window, 'tab_widget'):
                    # Buscar el índice de la pestaña de análisis
                    for i in range(main_window.tab_widget.count()):
                        if "Análisis" in main_window.tab_widget.tabText(i):
                            main_window.tab_widget.setCurrentIndex(i)
                            
                            # Actualizar la selección en la pestaña de análisis
                            if hasattr(main_window, 'analysis_tab'):
                                analysis_tab = main_window.analysis_tab
                                if hasattr(analysis_tab, 'prediction_combo'):
                                    # Buscar y seleccionar la predicción
                                    for j in range(analysis_tab.prediction_combo.count()):
                                        if analysis_tab.prediction_combo.itemText(j) == selection:
                                            analysis_tab.prediction_combo.setCurrentIndex(j)
                                            break
                            break
                
                QMessageBox.information(self, "Análisis", 
                                       f"Cambiando a pestaña de análisis para: {selection}")
                
            except Exception as e:
                QMessageBox.warning(self, "Error", 
                                   f"No se pudo cambiar a la pestaña de análisis:\n{str(e)}")
        else:
            QMessageBox.warning(self, "Advertencia", "Seleccione una predicción para analizar")
    
    def load_training_results(self, model_path):
        """Cargar resultados de entrenamiento desde otra pestaña"""
        self.load_available_results()
        
        # Seleccionar el entrenamiento más reciente
        if self.training_combo.count() > 0:
            self.training_combo.setCurrentIndex(0)
