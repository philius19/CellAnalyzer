"""CellAnalyzer: Quantitative 3D morphology analysis for cell surface reconstructions."""
import logging

from .analyzer import CellAnalyzer
from .io import load_cell_directory
from .datatypes import (
    MeshFrame,
    CellData,
    ProcessingMetadata,
    MeshParameters,
    AnalysisResults,
    MeshStatistics,
    CurvatureStatistics,
    QualityMetrics
)
from .visualization import basic_spatial_plot, publication_style

__version__ = "2.0.0"

def load_cell(cell_directory: str, channel: int = 1) -> CellAnalyzer:
    """
    Load cell data from u-shape3D output directory.

    Shorthand for CellAnalyzer(). Loads all frames into memory.

    Parameters:
        cell_directory: Path to cell directory
        channel: Channel number (default: 1)

    Returns:
        CellAnalyzer with all frames loaded

    Example:
        >>> from CellAnalyzer import load_cell
        >>> cell = load_cell('/path/to/cell')
        >>> print(f"Loaded {cell.n_frames} frames")
    """
    return CellAnalyzer(cell_directory, channel)

__all__ = [
    'CellAnalyzer',
    'load_cell',
    'MeshFrame',
    'CellData',
    'ProcessingMetadata',
    'AnalysisResults',
    'basic_spatial_plot',
    'publication_style'
]

def setup_logging(level: str = 'INFO', format: str = None) -> None:
    """Configure logging for CellAnalyzer."""
    if format is None:
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
