"""
Configuración global de la aplicación
"""

import os
from pathlib import Path

class Config:
    """Clase para manejar la configuración de la aplicación"""
    
    def __init__(self):
        self.BASE_DIR = Path(__file__).parent.parent.parent
        self.CONTENT_DIR = self.BASE_DIR / "content"
        self.UI_DIR = self.BASE_DIR / "UI"
        
        # Directorios de datos
        self.DATASET_DIR = self.CONTENT_DIR / "My-First-Project-3"
        self.TEST_IMAGES_DIR = self.CONTENT_DIR / "test_images"
        self.RUNS_DIR = self.CONTENT_DIR / "runs" / "detect"
        self.MODELS_DIR = self.CONTENT_DIR
        
        # Archivos de configuración
        self.DATA_YAML = self.DATASET_DIR / "data.yaml"
        
        # Modelos disponibles
        self.AVAILABLE_MODELS = [
            "yolov9s.pt",
            "yolo11n.pt"
        ]
        
        # Clases del dataset
        self.CLASSES = {
            0: "Root",
            1: "Stem", 
            2: "healty_leaves",
            3: "rooten_leaf"
        }
        
        # Configuración de entrenamiento por defecto
        self.DEFAULT_TRAIN_CONFIG = {
            "epochs": 125,
            "imgsz": 640,
            "batch": 16,
            "conf": 0.25
        }
    
    def get_latest_train_run(self):
        """Obtener el directorio del último entrenamiento"""
        train_dirs = []
        if self.RUNS_DIR.exists():
            for item in self.RUNS_DIR.iterdir():
                if item.is_dir() and item.name.startswith("train"):
                    train_dirs.append(item)
        
        if train_dirs:
            # Ordenar por fecha de modificación
            return max(train_dirs, key=lambda x: x.stat().st_mtime)
        return None
    
    def get_latest_predict_run(self):
        """Obtener el directorio de la última predicción"""
        predict_dirs = []
        if self.RUNS_DIR.exists():
            for item in self.RUNS_DIR.iterdir():
                if item.is_dir() and item.name.startswith("predict"):
                    predict_dirs.append(item)
        
        if predict_dirs:
            # Ordenar por fecha de modificación
            return max(predict_dirs, key=lambda x: x.stat().st_mtime)
        return None
    
    def get_all_train_runs(self):
        """Obtener todos los directorios de entrenamiento"""
        train_dirs = []
        if self.RUNS_DIR.exists():
            for item in self.RUNS_DIR.iterdir():
                if item.is_dir() and item.name.startswith("train"):
                    train_dirs.append(item)
        
        return sorted(train_dirs, key=lambda x: x.stat().st_mtime, reverse=True)
    
    def get_all_predict_runs(self):
        """Obtener todos los directorios de predicción"""
        predict_dirs = []
        if self.RUNS_DIR.exists():
            for item in self.RUNS_DIR.iterdir():
                if item.is_dir() and item.name.startswith("predict"):
                    predict_dirs.append(item)
        
        return sorted(predict_dirs, key=lambda x: x.stat().st_mtime, reverse=True)

# Instancia global de configuración
config = Config()
