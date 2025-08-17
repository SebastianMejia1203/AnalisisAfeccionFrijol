"""
Utilidades para el procesamiento con YOLO
"""

import subprocess
import os
import shutil
from pathlib import Path
from typing import Optional, List, Dict
from utils.config import config

class YOLOProcessor:
    """Clase para manejar operaciones de YOLO"""
    
    def __init__(self):
        self.config = config
    
    def train_model(self, model_name: str, epochs: int = 125, 
                   imgsz: int = 640, batch: int = 16) -> bool:
        """
        Entrenar modelo YOLO
        
        Args:
            model_name: Nombre del modelo a usar
            epochs: Número de épocas
            imgsz: Tamaño de imagen
            batch: Tamaño del batch
            
        Returns:
            bool: True si el entrenamiento fue exitoso
        """
        try:
            model_path = self.config.MODELS_DIR / model_name
            data_yaml = self.config.DATA_YAML
            
            if not model_path.exists():
                raise FileNotFoundError(f"Modelo {model_name} no encontrado")
            
            if not data_yaml.exists():
                raise FileNotFoundError("Archivo data.yaml no encontrado")
            
            # Cambiar al directorio content para ejecutar el comando
            original_dir = os.getcwd()
            os.chdir(self.config.CONTENT_DIR)
            
            # Comando de entrenamiento
            cmd = [
                "yolo", "task=detect", "mode=train",
                f"model={model_path}",
                f"data={data_yaml}",
                f"epochs={epochs}",
                f"imgsz={imgsz}",
                f"batch={batch}",
                "plots=True"
            ]
            
            # Ejecutar comando
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            # Restaurar directorio original
            os.chdir(original_dir)
            
            return process.returncode == 0
            
        except Exception as e:
            print(f"Error en entrenamiento: {e}")
            return False
    
    def predict_images(self, source_path: str, model_path: Optional[str] = None,
                      conf: float = 0.25, save_crops: bool = True,
                      classes: Optional[List[int]] = None) -> Optional[Path]:
        """
        Realizar predicciones en imágenes
        
        Args:
            source_path: Ruta de las imágenes fuente
            model_path: Ruta del modelo (usa el último entrenado si es None)
            conf: Confianza mínima
            save_crops: Guardar recortes de las detecciones
            classes: Lista de clases específicas a detectar
            
        Returns:
            Path: Directorio de resultados o None si falla
        """
        try:
            # Usar modelo del último entrenamiento si no se especifica
            if model_path is None:
                latest_train = self.config.get_latest_train_run()
                if latest_train:
                    model_path = latest_train / "weights" / "best.pt"
                else:
                    # Usar modelo pre-entrenado por defecto
                    model_path = self.config.MODELS_DIR / "yolov9s.pt"
            
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Modelo {model_path} no encontrado")
            
            # Cambiar al directorio content
            original_dir = os.getcwd()
            os.chdir(self.config.CONTENT_DIR)
            
            # Construir comando
            cmd = [
                "yolo", "task=detect", "mode=predict",
                f"model={model_path}",
                f"conf={conf}",
                f"source={source_path}"
            ]
            
            if save_crops:
                cmd.append("save_crop=True")
            
            if classes:
                classes_str = "[" + ",".join(map(str, classes)) + "]"
                cmd.append(f"classes={classes_str}")
            
            # Ejecutar comando
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            # Restaurar directorio
            os.chdir(original_dir)
            
            if process.returncode == 0:
                return self.config.get_latest_predict_run()
            else:
                print(f"Error en predicción: {process.stderr}")
                return None
                
        except Exception as e:
            print(f"Error en predicción: {e}")
            return None
    
    def get_model_info(self, model_path: str) -> Dict:
        """
        Obtener información de un modelo
        
        Args:
            model_path: Ruta del modelo
            
        Returns:
            Dict: Información del modelo
        """
        info = {
            "exists": Path(model_path).exists(),
            "size": 0,
            "modified": None
        }
        
        if info["exists"]:
            path = Path(model_path)
            info["size"] = path.stat().st_size
            info["modified"] = path.stat().st_mtime
        
        return info
    
    def get_available_runs(self, run_type: str = "all") -> List[Path]:
        """
        Obtener runs disponibles
        
        Args:
            run_type: Tipo de run ("train", "predict", "all")
            
        Returns:
            List[Path]: Lista de directorios de runs
        """
        if run_type == "train":
            return self.config.get_all_train_runs()
        elif run_type == "predict":
            return self.config.get_all_predict_runs()
        else:
            train_runs = self.config.get_all_train_runs()
            predict_runs = self.config.get_all_predict_runs()
            all_runs = train_runs + predict_runs
            return sorted(all_runs, key=lambda x: x.stat().st_mtime, reverse=True)
