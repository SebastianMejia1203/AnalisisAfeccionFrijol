# 🌱 Sistema de Análisis de Plantas de Frijol con YOLO v9

Una aplicación moderna con interfaz gráfica para la detección y análisis de enfermedades en plantas de frijol utilizando YOLOv9.

## 🌟 Características

### 🏋️ Entrenamiento de Modelos
- **Entrenamiento personalizado** con YOLOv9 y YOLO11
- **Configuración flexible** de parámetros (épocas, batch size, tamaño de imagen)
- **Monitoreo en tiempo real** del progreso de entrenamiento
- **Visualización de métricas** (pérdidas, mAP, precisión, recall)
- **Gestión de modelos** entrenados

### 🔍 Predicción y Detección
- **Predicción en lote** o imágenes individuales
- **Configuración de confianza** y filtros por clase
- **Generación automática de recortes** (crops)
- **Visualización de resultados** con bounding boxes
- **Soporte para múltiples formatos** de imagen

### 🔬 Análisis de Afectación
- **Análisis por color HSV** para determinar estado de la planta
- **Clasificación automática**: Sano, Afectado, Severamente afectado
- **Estadísticas detalladas** por imagen y categoría
- **Visualización de máscaras** de color
- **Exportación de resultados** a CSV

### 📊 Visualización de Resultados
- **Gráficas de entrenamiento** interactivas
- **Galería de imágenes** procesadas
- **Comparación entre modelos** entrenados
- **Reportes estadísticos** completos
- **Exportación de modelos** entrenados

## 🛠️ Requisitos del Sistema

### Software Necesario
- **Python 3.8** o superior
- **Windows 10/11** (recomendado)
- **8GB RAM** mínimo, 16GB recomendado
- **GPU NVIDIA** (opcional, para entrenamiento acelerado)

### Estructura de Proyecto Requerida
```
Vision Computacional/
└── Segmentacion/
    ├── content/                    # Datos del proyecto
    │   ├── My-First-Project-3/    # Dataset de Roboflow
    │   ├── test_images/           # Imágenes de prueba
    │   ├── runs/                  # Resultados de entrenamientos/predicciones
    │   ├── yolov9s.pt            # Modelo pre-entrenado
    │   └── yolo11n.pt            # Modelo pre-entrenado
    └── UI/                        # Aplicación con interfaz gráfica
        ├── main.py               # Punto de entrada
        ├── setup.py              # Script de instalación
        └── requirements.txt      # Dependencias
```

## 🚀 Instalación

### Método 1: Instalación Automática
```bash
cd "e:\Python\Vision Computacional\Segmentacion\UI"
python setup.py
```

### Método 2: Instalación Manual
```bash
# 1. Navegar al directorio
cd "e:\Python\Vision Computacional\Segmentacion\UI"

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar aplicación
python main.py
```

## 📋 Dependencias Principales

- **PyQt6** - Interfaz gráfica moderna
- **ultralytics** - YOLOv9 y YOLO11
- **OpenCV** - Procesamiento de imágenes
- **PyTorch** - Machine learning backend
- **matplotlib** - Visualización de gráficas
- **pandas** - Análisis de datos
- **numpy** - Computación numérica

## 💻 Uso de la Aplicación

### 1. 🏋️ Entrenamiento
1. **Ir a la pestaña "Entrenamiento"**
2. **Seleccionar modelo base** (yolov9s.pt o yolo11n.pt)
3. **Configurar parámetros**:
   - Épocas: 125 (recomendado)
   - Tamaño imagen: 640
   - Batch size: 16 (ajustar según GPU)
4. **Hacer clic en "Iniciar Entrenamiento"**
5. **Monitorear progreso** en tiempo real

### 2. 🔍 Predicción
1. **Ir a la pestaña "Predicción"**
2. **Seleccionar modelo** (pre-entrenado o recién entrenado)
3. **Elegir fuente de imágenes**:
   - Usar test images
   - Seleccionar carpeta personalizada
   - Imagen individual
4. **Configurar parámetros**:
   - Confianza: 0.25 (recomendado)
   - Guardar recortes: ✓
   - Filtrar por clases (opcional)
5. **Hacer clic en "Iniciar Predicción"**

### 3. 🔬 Análisis de Afectación
1. **Ir a la pestaña "Análisis"**
2. **Seleccionar predicción** con recortes
3. **Hacer clic en "Iniciar Análisis"**
4. **Revisar resultados**:
   - Tabla resumen por imagen
   - Estadísticas generales
   - Distribución de afectación
5. **Exportar resultados** si es necesario

### 4. 📊 Visualización de Resultados
1. **Ir a la pestaña "Resultados"**
2. **Explorar sub-pestañas**:
   - **Entrenamientos**: Métricas y gráficas
   - **Predicciones**: Galería de imágenes
   - **Comparaciones**: Comparar modelos
3. **Exportar modelos** entrenados

## 🎯 Clases de Detección

El modelo detecta **4 clases principales**:
- **Root** (Raíz) - ID: 0
- **Stem** (Tallo) - ID: 1  
- **healty_leaves** (Hojas sanas) - ID: 2
- **rooten_leaf** (Hoja con pudrición) - ID: 3

## 🔬 Análisis de Afectación por Color

### Rangos HSV Utilizados
- **Sano (Verde)**: H[35-85], S[40-255], V[40-255]
- **Afectado (Amarillo)**: H[20-30], S[100-255], V[100-255]
- **Severo (Marrón)**: H[10-20], S[50-255], V[20-200]

### Clasificación de Severidad
- **Prácticamente sano**: < 10% afectación
- **Afectación leve**: 10-30% afectación  
- **Afectación moderada**: 30-70% afectación
- **Afectación severa**: ≥ 70% afectación

## 📁 Estructura de Archivos Generados

```
content/runs/detect/
├── train/                     # Entrenamientos
│   ├── weights/
│   │   ├── best.pt           # Mejor modelo
│   │   └── last.pt           # Último modelo
│   ├── results.csv           # Métricas de entrenamiento
│   └── plots/                # Gráficas generadas
├── predict/                  # Predicciones
│   ├── *.jpg                # Imágenes con detecciones
│   └── crops/               # Recortes por clase
│       ├── Root/
│       ├── Stem/
│       ├── healty_leaves/
│       └── rooten_leaf/
└── predict2/, predict3/...   # Predicciones adicionales
```

## 🔧 Configuración Avanzada

### Ajustar Sensibilidad de Análisis de Color
Modifica los rangos HSV en `utils/plant_analyzer.py`:

```python
self.color_ranges = {
    "sano": {
        "lower": np.array([35, 40, 40]),    # H, S, V mínimos
        "upper": np.array([85, 255, 255])   # H, S, V máximos
    },
    # ... más rangos
}
```

### Personalizar Clases de Detección
Modifica `utils/config.py`:

```python
self.CLASSES = {
    0: "Root",
    1: "Stem", 
    2: "healty_leaves",
    3: "rooten_leaf"
}
```

## 🐛 Solución de Problemas

### Error: "Dataset no encontrado"
- Verifica que existe `content/My-First-Project-3/data.yaml`
- Asegúrate de que las carpetas `train`, `valid`, `test` existan

### Error: "Modelo no encontrado"  
- Descarga los modelos pre-entrenados en `content/`
- Verifica que los archivos `.pt` no estén corruptos

### Error de memoria durante entrenamiento
- Reduce el `batch_size` (ej: de 16 a 8 o 4)
- Reduce el `imgsz` (ej: de 640 a 416)

### Error: "CUDA out of memory"
- Usar CPU: modifica código para usar `device='cpu'`
- Cerrar otras aplicaciones que usen GPU
- Reducir batch size

## 📊 Interpretación de Métricas

### Métricas de Entrenamiento
- **mAP@0.5**: Precisión media a IoU 0.5 (objetivo: > 0.7)
- **mAP@0.5:0.95**: Precisión media promediada (objetivo: > 0.5)
- **Box Loss**: Pérdida de localización (debe decrecer)
- **Class Loss**: Pérdida de clasificación (debe decrecer)

### Métricas de Análisis
- **Porcentaje Sano**: Tejido vegetal saludable
- **Porcentaje Afectado**: Tejido con síntomas iniciales
- **Porcentaje Severo**: Tejido severamente dañado
- **Afectación Total**: Suma de afectado + severo

## 🤝 Contribución

Para contribuir al proyecto:

1. **Fork del repositorio**
2. **Crear rama de feature** (`git checkout -b feature/AmazingFeature`)
3. **Commit cambios** (`git commit -m 'Add AmazingFeature'`)
4. **Push a la rama** (`git push origin feature/AmazingFeature`)
5. **Abrir Pull Request**

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo `LICENSE` para más detalles.

## 👥 Autores

- **Sistema de Visión Computacional** - Desarrollo inicial

## 📞 Soporte

Para soporte técnico:
- **Crear issue** en el repositorio
- **Documentar el error** con screenshots
- **Incluir logs** de la aplicación (carpeta `UI/logs/`)

---

**🌱 ¡Disfruta analizando plantas de frijol con IA!** 🤖
