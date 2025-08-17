"""
Sistema de logging para la aplicación
"""

import logging
import os
from pathlib import Path
from datetime import datetime

class AppLogger:
    """Clase para manejar logging de la aplicación"""
    
    def __init__(self, name="PlantAnalysis", log_dir="logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Crear logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Evitar duplicar handlers
        if not self.logger.handlers:
            self.setup_handlers()
    
    def setup_handlers(self):
        """Configurar handlers de logging"""
        # Formato de mensaje
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Handler para archivo
        log_file = self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Handler para consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Agregar handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def debug(self, message):
        """Log debug"""
        self.logger.debug(message)
    
    def info(self, message):
        """Log info"""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning"""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error"""
        self.logger.error(message)
    
    def critical(self, message):
        """Log critical"""
        self.logger.critical(message)

# Instancia global del logger
app_logger = AppLogger()
