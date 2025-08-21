"""
Utilidades para análisis de afectación en hojas
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class PlantAnalyzer:
    """Clase para análisis de afectación en plantas"""
    
    def __init__(self):
        # Rangos de color en HSV para diferentes estados, ajustados según las imágenes
        self.color_ranges = {
            "sano": {
                # Verdes vibrantes y saludables
                "lower": np.array([35, 50, 50]),
                "upper": np.array([85, 255, 255])
            },
            "afectado": {
                # Tonos amarillentos y verdes pálidos
                "lower": np.array([20, 50, 50]),
                "upper": np.array([34, 255, 255])
            },
            "severo": {
                # Tonos marrones, cafés y áreas necróticas (secas/quemadas)
                "lower": np.array([5, 30, 20]),
                "upper": np.array([19, 255, 200])
            }
        }
        
        # Colores para visualización (BGR)
        self.visualization_colors = {
            "sano": [0, 255, 0],      # Verde
            "afectado": [0, 255, 255], # Amarillo
            "severo": [0, 0, 128]      # Marrón oscuro
        }
    
    def analyze_leaf_damage(self, image_path: str) -> Optional[Dict]:
        """
        Analizar daño en una hoja individual
        
        Args:
            image_path: Ruta a la imagen de la hoja
            
        Returns:
            Dict: Resultados del análisis o None si falla
        """
        try:
            # Cargar imagen
            img = cv2.imread(str(image_path))
            if img is None:
                return None
            
            # Convertir a HSV
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Crear máscaras para cada estado
            masks = {}
            pixel_counts = {}
            
            for state, color_range in self.color_ranges.items():
                mask = cv2.inRange(hsv_img, color_range["lower"], color_range["upper"])
                masks[state] = mask
                pixel_counts[state] = cv2.countNonZero(mask)
            
            # Calcular total de píxeles de la planta
            total_plant_pixels = sum(pixel_counts.values())
            
            if total_plant_pixels == 0:
                return {
                    "sano": 0.0,
                    "afectado": 0.0,
                    "severo": 0.0,
                    "afectacion_total": 0.0,
                    "total_pixels": 0,
                    "masks": masks,
                    "original_image": img
                }
            
            # Calcular porcentajes
            percentages = {}
            for state, count in pixel_counts.items():
                percentages[state] = (count / total_plant_pixels) * 100
            
            # Afectación total
            afectacion_total = percentages["afectado"] + percentages["severo"]
            
            # Crear máscara superpuesta para visualización
            overlay_mask = np.zeros_like(img)
            for state, mask in masks.items():
                color = self.visualization_colors[state]
                overlay_mask[mask > 0] = color
            
            return {
                "sano": percentages["sano"],
                "afectado": percentages["afectado"],
                "severo": percentages["severo"],
                "afectacion_total": afectacion_total,
                "total_pixels": total_plant_pixels,
                "pixel_counts": pixel_counts,
                "masks": masks,
                "overlay_mask": overlay_mask,
                "original_image": img
            }
            
        except Exception as e:
            print(f"Error analizando imagen {image_path}: {e}")
            return None
    
    def analyze_crops_directory(self, crops_dir: str) -> Dict:
        """
        Analizar todas las imágenes recortadas en un directorio
        
        Args:
            crops_dir: Directorio con subcarpetas de crops por categoría
            
        Returns:
            Dict: Resultados organizados por imagen y categoría
        """
        results = {}
        crops_path = Path(crops_dir)
        
        if not crops_path.exists():
            return results
        
        # Iterar por cada categoría (subcarpeta)
        for category_dir in crops_path.iterdir():
            if not category_dir.is_dir():
                continue
                
            category_name = category_dir.name
            
            # Procesar cada imagen en la categoría
            for image_file in category_dir.iterdir():
                if image_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue
                
                # Extraer número de imagen del nombre del archivo
                try:
                    # Asumir formato imgXXX...
                    image_num = int(image_file.stem[3:6])
                except (ValueError, IndexError):
                    continue
                
                # Analizar la imagen
                analysis_result = self.analyze_leaf_damage(str(image_file))
                if analysis_result:
                    # Organizar resultados
                    if image_num not in results:
                        results[image_num] = {}
                    
                    if category_name not in results[image_num]:
                        results[image_num][category_name] = []
                    
                    results[image_num][category_name].append(analysis_result)
        
        return results
    
    def calculate_summary_statistics(self, analysis_results: Dict) -> Dict:
        """
        Calcular estadísticas resumen de los análisis
        
        Args:
            analysis_results: Resultados del análisis por imagen/categoría
            
        Returns:
            Dict: Estadísticas resumen
        """
        summary = {}
        
        for image_num, categories in analysis_results.items():
            summary[image_num] = {}
            
            for category, results_list in categories.items():
                if not results_list:
                    continue
                
                # Calcular promedios para esta categoría e imagen
                avg_sano = np.mean([r["sano"] for r in results_list])
                avg_afectado = np.mean([r["afectado"] for r in results_list])
                avg_severo = np.mean([r["severo"] for r in results_list])
                avg_afectacion_total = np.mean([r["afectacion_total"] for r in results_list])
                
                summary[image_num][category] = {
                    "sano": avg_sano,
                    "afectado": avg_afectado,
                    "severo": avg_severo,
                    "afectacion_total": avg_afectacion_total,
                    "count": len(results_list)
                }
        
        return summary
    
    def save_analysis_visualization(self, image_path: str, analysis_result: Dict, 
                                  output_path: str) -> bool:
        """
        Guardar visualización del análisis
        
        Args:
            image_path: Ruta de la imagen original
            analysis_result: Resultado del análisis
            output_path: Ruta donde guardar la visualización
            
        Returns:
            bool: True si se guardó correctamente
        """
        try:
            if "overlay_mask" not in analysis_result:
                return False
            
            original = analysis_result["original_image"]
            overlay = analysis_result["overlay_mask"]
            
            # Crear visualización combinada
            alpha = 0.6
            combined = cv2.addWeighted(original, 1-alpha, overlay, alpha, 0)
            
            # Agregar texto con estadísticas
            text_info = [
                f"Sano: {analysis_result['sano']:.1f}%",
                f"Afectado: {analysis_result['afectado']:.1f}%",
                f"Severo: {analysis_result['severo']:.1f}%",
                f"Total Afectacion: {analysis_result['afectacion_total']:.1f}%"
            ]
            
            y_offset = 30
            for i, text in enumerate(text_info):
                y_pos = y_offset + (i * 25)
                cv2.putText(combined, text, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Guardar imagen
            cv2.imwrite(str(output_path), combined)
            return True
            
        except Exception as e:
            print(f"Error guardando visualización: {e}")
            return False