"""
Utilidades de validación para la aplicación
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from utils.config import config

class SystemValidator:
    """Validador del sistema y requisitos"""
    
    @staticmethod
    def validate_environment() -> Dict[str, bool]:
        """Validar el entorno del sistema"""
        results = {
            "python_version": False,
            "pytorch": False,
            "opencv": False,
            "ultralytics": False,
            "pyqt6": False,
            "dataset": False,
            "directories": False
        }
        
        # Validar versión de Python
        import sys
        if sys.version_info >= (3, 8):
            results["python_version"] = True
        
        # Validar PyTorch
        try:
            import torch
            results["pytorch"] = True
        except ImportError:
            pass
        
        # Validar OpenCV
        try:
            import cv2
            results["opencv"] = True
        except ImportError:
            pass
        
        # Validar Ultralytics
        try:
            import ultralytics
            results["ultralytics"] = True
        except ImportError:
            pass
        
        # Validar PyQt6
        try:
            import PyQt6
            results["pyqt6"] = True
        except ImportError:
            pass
        
        # Validar dataset
        if config.DATA_YAML.exists():
            results["dataset"] = True
        
        # Validar directorios
        required_dirs = [
            config.CONTENT_DIR,
            config.RUNS_DIR.parent,  # runs/detect debería crearse automáticamente
            config.TEST_IMAGES_DIR
        ]
        
        if all(d.exists() for d in required_dirs):
            results["directories"] = True
        
        return results
    
    @staticmethod
    def get_validation_report() -> str:
        """Obtener reporte de validación"""
        results = SystemValidator.validate_environment()
        
        report = "VALIDACIÓN DEL SISTEMA\n"
        report += "=" * 30 + "\n\n"
        
        status_map = {True: "✅", False: "❌"}
        
        report += f"Python (>= 3.8): {status_map[results['python_version']]}\n"
        report += f"PyTorch: {status_map[results['pytorch']]}\n"
        report += f"OpenCV: {status_map[results['opencv']]}\n"
        report += f"Ultralytics: {status_map[results['ultralytics']]}\n"
        report += f"PyQt6: {status_map[results['pyqt6']]}\n"
        report += f"Dataset: {status_map[results['dataset']]}\n"
        report += f"Directorios: {status_map[results['directories']]}\n\n"
        
        # Información adicional
        report += "INFORMACIÓN ADICIONAL:\n"
        report += "-" * 25 + "\n"
        
        try:
            import torch
            report += f"PyTorch versión: {torch.__version__}\n"
            report += f"CUDA disponible: {torch.cuda.is_available()}\n"
            if torch.cuda.is_available():
                report += f"GPUs disponibles: {torch.cuda.device_count()}\n"
        except:
            pass
        
        try:
            import cv2
            report += f"OpenCV versión: {cv2.__version__}\n"
        except:
            pass
        
        report += f"Dataset ubicación: {config.DATASET_DIR}\n"
        report += f"Contenido ubicación: {config.CONTENT_DIR}\n"
        
        return report
    
    @staticmethod
    def validate_model_file(model_path: str) -> bool:
        """Validar archivo de modelo"""
        path = Path(model_path)
        return path.exists() and path.suffix == '.pt' and path.stat().st_size > 0
    
    @staticmethod
    def validate_image_file(image_path: str) -> bool:
        """Validar archivo de imagen"""
        path = Path(image_path)
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        return path.exists() and path.suffix.lower() in valid_extensions
    
    @staticmethod
    def validate_directory_structure() -> List[str]:
        """Validar estructura de directorios"""
        issues = []
        
        # Verificar directorios principales
        if not config.CONTENT_DIR.exists():
            issues.append(f"Directorio content no encontrado: {config.CONTENT_DIR}")
        
        if not config.DATASET_DIR.exists():
            issues.append(f"Directorio del dataset no encontrado: {config.DATASET_DIR}")
        
        if not config.TEST_IMAGES_DIR.exists():
            issues.append(f"Directorio test_images no encontrado: {config.TEST_IMAGES_DIR}")
        
        # Verificar estructura del dataset
        if config.DATASET_DIR.exists():
            required_subdirs = ['train', 'valid', 'test']
            for subdir in required_subdirs:
                subdir_path = config.DATASET_DIR / subdir
                if not subdir_path.exists():
                    issues.append(f"Subdirectorio del dataset faltante: {subdir_path}")
                else:
                    images_path = subdir_path / "images"
                    labels_path = subdir_path / "labels"
                    
                    if not images_path.exists():
                        issues.append(f"Directorio de imágenes faltante: {images_path}")
                    
                    if not labels_path.exists():
                        issues.append(f"Directorio de etiquetas faltante: {labels_path}")
        
        # Verificar archivo data.yaml
        if not config.DATA_YAML.exists():
            issues.append(f"Archivo data.yaml no encontrado: {config.DATA_YAML}")
        
        return issues

class InputValidator:
    """Validador de entrada de usuario"""
    
    @staticmethod
    def validate_training_params(epochs: int, imgsz: int, batch: int) -> Tuple[bool, str]:
        """Validar parámetros de entrenamiento"""
        if epochs < 1 or epochs > 1000:
            return False, "Las épocas deben estar entre 1 y 1000"
        
        if imgsz < 320 or imgsz > 1280 or imgsz % 32 != 0:
            return False, "El tamaño de imagen debe ser múltiplo de 32, entre 320 y 1280"
        
        if batch < 1 or batch > 64:
            return False, "El batch size debe estar entre 1 y 64"
        
        return True, "Parámetros válidos"
    
    @staticmethod
    def validate_prediction_params(conf: float) -> Tuple[bool, str]:
        """Validar parámetros de predicción"""
        if conf < 0.1 or conf > 1.0:
            return False, "La confianza debe estar entre 0.1 y 1.0"
        
        return True, "Parámetros válidos"
    
    @staticmethod
    def validate_class_selection(selected_classes: List[int]) -> Tuple[bool, str]:
        """Validar selección de clases"""
        if not selected_classes:
            return True, "Sin filtro de clases (se detectarán todas)"
        
        valid_classes = set(config.CLASSES.keys())
        invalid_classes = set(selected_classes) - valid_classes
        
        if invalid_classes:
            return False, f"Clases inválidas: {invalid_classes}"
        
        return True, f"Clases válidas seleccionadas: {selected_classes}"
