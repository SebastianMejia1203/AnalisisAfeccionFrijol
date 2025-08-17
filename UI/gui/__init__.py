"""
Archivo __init__.py para el paquete gui
"""

from .main_window import MainWindow
from .train_tab import TrainTab
from .predict_tab import PredictTab
from .analysis_tab import AnalysisTab
from .results_tab import ResultsTab

__all__ = ['MainWindow', 'TrainTab', 'PredictTab', 'AnalysisTab', 'ResultsTab']
