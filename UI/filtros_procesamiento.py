"""
Módulo de filtros de procesamiento de imágenes para análisis de enfermedades en plantas
Adaptado de los filtros existentes para integración con PyQt6
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple

class FiltrosProcessamiento:
    """Clase unificada para aplicar diferentes filtros de procesamiento a imágenes"""
    
    def __init__(self):
        self.filtros_disponibles = {
            'Gaussiano': self._filtro_gaussiano,
            'Laplaciano': self._filtro_laplaciano, 
            'Media': self._filtro_media,
            'Mediana': self._filtro_mediana,
            'Sobel': self._filtro_sobel,
            'Prewitt': self._filtro_prewitt
        }
        
        # Parámetros por defecto para cada filtro
        self.parametros_default = {
            'Gaussiano': {'kernel_size': 5, 'sigma': 1.0},
            'Laplaciano': {'tipo': '4-conectado'},
            'Media': {'kernel_size': 5},
            'Mediana': {'kernel_size': 5},
            'Sobel': {'direccion': 'ambas'},
            'Prewitt': {'direccion': 'ambas'}
        }
    
    def aplicar_filtro(self, imagen: np.ndarray, tipo_filtro: str, 
                      parametros: Dict[str, Any] = None) -> np.ndarray:
        """
        Aplica el filtro especificado a la imagen
        
        Args:
            imagen: Imagen de entrada (BGR)
            tipo_filtro: Tipo de filtro a aplicar
            parametros: Parámetros específicos del filtro
            
        Returns:
            Imagen filtrada
        """
        if tipo_filtro not in self.filtros_disponibles:
            raise ValueError(f"Filtro '{tipo_filtro}' no disponible")
        
        # Usar parámetros por defecto si no se proporcionan
        if parametros is None:
            parametros = self.parametros_default.get(tipo_filtro, {})
        
        # Aplicar filtro
        return self.filtros_disponibles[tipo_filtro](imagen, parametros)
    
    def get_filtros_disponibles(self) -> list:
        """Retorna lista de filtros disponibles"""
        return list(self.filtros_disponibles.keys())
    
    def get_parametros_filtro(self, tipo_filtro: str) -> Dict[str, Any]:
        """Retorna parámetros por defecto de un filtro"""
        return self.parametros_default.get(tipo_filtro, {}).copy()
    
    # ==================== FILTROS PASA-BAJO (Suavizado) ====================
    
    def _filtro_gaussiano(self, imagen: np.ndarray, params: Dict) -> np.ndarray:
        """
        Filtro Gaussiano - PASA-BAJO
        Reduce ruido manteniendo los bordes relativamente intactos
        """
        kernel_size = params.get('kernel_size', 5)
        sigma = params.get('sigma', 1.0)
        
        # Asegurar que el kernel sea impar
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        return cv2.GaussianBlur(imagen, (kernel_size, kernel_size), sigma)
    
    def _filtro_media(self, imagen: np.ndarray, params: Dict) -> np.ndarray:
        """
        Filtro de Media - PASA-BAJO
        Suaviza la imagen reduciendo ruido de alta frecuencia
        """
        kernel_size = params.get('kernel_size', 5)
        
        # Asegurar que el kernel sea impar
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        return cv2.blur(imagen, (kernel_size, kernel_size))
    
    def _filtro_mediana(self, imagen: np.ndarray, params: Dict) -> np.ndarray:
        """
        Filtro de Mediana - PASA-BAJO
        Excelente para eliminar ruido de sal y pimienta
        """
        kernel_size = params.get('kernel_size', 5)
        
        # Asegurar que el kernel sea impar
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        return cv2.medianBlur(imagen, kernel_size)
    
    # ==================== FILTROS PASA-ALTO (Detección de bordes) ====================
    
    def _filtro_laplaciano(self, imagen: np.ndarray, params: Dict) -> np.ndarray:
        """
        Filtro Laplaciano - PASA-ALTO
        Detecta bordes en todas las direcciones
        """
        # Convertir a escala de grises si es necesario
        if len(imagen.shape) == 3:
            imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            imagen_gris = imagen
        
        # Aplicar Laplaciano
        laplaciano = cv2.Laplacian(imagen_gris, cv2.CV_64F)
        
        # Normalizar y convertir a uint8
        laplaciano = np.abs(laplaciano)
        laplaciano = np.clip(laplaciano, 0, 255).astype(np.uint8)
        
        # Convertir de vuelta a BGR para consistencia
        if len(imagen.shape) == 3:
            return cv2.cvtColor(laplaciano, cv2.COLOR_GRAY2BGR)
        
        return laplaciano
    
    def _filtro_sobel(self, imagen: np.ndarray, params: Dict) -> np.ndarray:
        """
        Filtro Sobel - PASA-ALTO
        Detecta bordes direccionales (X, Y o ambas)
        """
        direccion = params.get('direccion', 'ambas')
        
        # Convertir a escala de grises
        if len(imagen.shape) == 3:
            imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            imagen_gris = imagen
        
        if direccion == 'x':
            sobel = cv2.Sobel(imagen_gris, cv2.CV_64F, 1, 0, ksize=3)
        elif direccion == 'y':
            sobel = cv2.Sobel(imagen_gris, cv2.CV_64F, 0, 1, ksize=3)
        else:  # ambas
            sobel_x = cv2.Sobel(imagen_gris, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(imagen_gris, cv2.CV_64F, 0, 1, ksize=3)
            sobel = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Normalizar
        sobel = np.clip(sobel, 0, 255).astype(np.uint8)
        
        # Convertir de vuelta a BGR
        if len(imagen.shape) == 3:
            return cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
            
        return sobel
    
    def _filtro_prewitt(self, imagen: np.ndarray, params: Dict) -> np.ndarray:
        """
        Filtro Prewitt - PASA-ALTO  
        Similar a Sobel pero con diferentes pesos en el kernel
        """
        direccion = params.get('direccion', 'ambas')
        
        # Convertir a escala de grises
        if len(imagen.shape) == 3:
            imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            imagen_gris = imagen
        
        # Kernels Prewitt
        kernel_x = np.array([[-1, 0, 1],
                           [-1, 0, 1], 
                           [-1, 0, 1]], dtype=np.float32)
        
        kernel_y = np.array([[-1, -1, -1],
                           [ 0,  0,  0],
                           [ 1,  1,  1]], dtype=np.float32)
        
        if direccion == 'x':
            prewitt = cv2.filter2D(imagen_gris, cv2.CV_64F, kernel_x)
        elif direccion == 'y':
            prewitt = cv2.filter2D(imagen_gris, cv2.CV_64F, kernel_y)
        else:  # ambas
            prewitt_x = cv2.filter2D(imagen_gris, cv2.CV_64F, kernel_x)
            prewitt_y = cv2.filter2D(imagen_gris, cv2.CV_64F, kernel_y)
            prewitt = np.sqrt(prewitt_x**2 + prewitt_y**2)
        
        # Normalizar
        prewitt = np.abs(prewitt)
        prewitt = np.clip(prewitt, 0, 255).astype(np.uint8)
        
        # Convertir de vuelta a BGR
        if len(imagen.shape) == 3:
            return cv2.cvtColor(prewitt, cv2.COLOR_GRAY2BGR)
            
        return prewitt

    def calcular_histograma_rgb(self, imagen: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calcula histogramas RGB de una imagen
        
        Returns:
            Tupla con histogramas (B, G, R)
        """
        if len(imagen.shape) == 3:
            hist_b = cv2.calcHist([imagen], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([imagen], [1], None, [256], [0, 256])
            hist_r = cv2.calcHist([imagen], [2], None, [256], [0, 256])
            return hist_b.flatten(), hist_g.flatten(), hist_r.flatten()
        else:
            # Imagen en escala de grises
            hist = cv2.calcHist([imagen], [0], None, [256], [0, 256])
            hist = hist.flatten()
            return hist, hist, hist
    
    def get_info_filtro(self, tipo_filtro: str) -> Dict[str, Any]:
        """
        Retorna información descriptiva sobre el filtro
        """
        info_filtros = {
            'Gaussiano': {
                'tipo': 'Pasa-bajo',
                'descripcion': 'Suaviza la imagen reduciendo ruido mientras preserva los bordes',
                'uso': 'Ideal para preprocesamiento y reducción de ruido gaussiano',
                'parametros': 'kernel_size: tamaño del kernel, sigma: desviación estándar'
            },
            'Media': {
                'tipo': 'Pasa-bajo', 
                'descripcion': 'Promedia los píxeles vecinos para suavizar la imagen',
                'uso': 'Reducción de ruido simple y rápida',
                'parametros': 'kernel_size: tamaño del kernel de promediado'
            },
            'Mediana': {
                'tipo': 'Pasa-bajo',
                'descripcion': 'Reemplaza cada píxel con la mediana de sus vecinos',
                'uso': 'Excelente para eliminar ruido de sal y pimienta',
                'parametros': 'kernel_size: tamaño del kernel'
            },
            'Laplaciano': {
                'tipo': 'Pasa-alto',
                'descripcion': 'Detecta bordes y cambios de intensidad en todas las direcciones',
                'uso': 'Realce de bordes y detección de características',
                'parametros': 'tipo: 4-conectado o 8-conectado'
            },
            'Sobel': {
                'tipo': 'Pasa-alto',
                'descripcion': 'Detecta bordes direccionales usando gradientes',
                'uso': 'Detección de bordes con información direccional',
                'parametros': 'direccion: x, y, o ambas'
            },
            'Prewitt': {
                'tipo': 'Pasa-alto',
                'descripcion': 'Similar a Sobel con diferentes pesos en el kernel',
                'uso': 'Detección de bordes alternativa a Sobel',
                'parametros': 'direccion: x, y, o ambas'
            }
        }
        
        return info_filtros.get(tipo_filtro, {})
