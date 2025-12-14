"""Cell-level mesh analysis orchestrator.

CellAnalyzer is a COORDINATOR - it delegates work to specialized modules:
- io.py: Data loading
- analysis.py: Statistical computations
- visualization.py: Plotting

Design principle: CellAnalyzer has NO business logic, only delegation.
"""
import logging
from pathlib import Path
from typing import Dict, List, Iterator, Tuple

from .datatypes import (
    CellData,
    MeshFrame,
    ProcessingMetadata,
    AnalysisResults
)
from .io import load_cell_directory

logger = logging.getLogger(__name__)


class CellAnalyzer:
    """
    Central orchestrator for u-shape3D mesh analysis pipeline.

    Coordinates data loading, analysis, and visualization by delegating
    to specialized modules. Provides clean dict-like interface.

    Example:
        >>> cell = CellAnalyzer('/path/to/cell_directory')
        >>> print(f"Loaded {cell.n_frames} frames")
        >>> frame = cell[1]  # Access frame
        >>> stats = cell.calculate_statistics(1)  # Delegates to analysis.py
        >>> fig = cell.plot_frame(1)  # Delegates to visualization.py
    """

    def __init__(self, cell_directory: str, channel: int = 1,
                 load_auxiliary: bool = False):
        """
        Initialize by loading all frames from u-shape3D output directory.

        Parameters:
            cell_directory: Path to cell directory with MovieData and mesh files
            channel: Channel number (default: 1)
            load_auxiliary: Load auxiliary data (normals, Gaussian curvature)
        """
        self._data = load_cell_directory(Path(cell_directory), channel, load_auxiliary)
        logger.info(f"CellAnalyzer initialized with {self.n_frames} frames")

    # Data Access (delegate to CellData)

    def __getitem__(self, time_index: int) -> MeshFrame:
        """Access frame by time index."""
        return self._data[time_index]

    def __iter__(self) -> Iterator[Tuple[int, MeshFrame]]:
        """Iterate over (time_index, frame) pairs."""
        return iter(self._data)

    def __len__(self) -> int:
        """Number of frames."""
        return len(self._data)

    def __contains__(self, time_index: int) -> bool:
        """Check if time index exists."""
        return time_index in self._data

    # Properties (simple delegation, no computation) 

    @property
    def data(self) -> CellData:
        """Access underlying CellData container."""
        return self._data

    @property
    def metadata(self) -> ProcessingMetadata:
        """Processing metadata (acquisition params, mesh params, provenance)."""
        return self._data.metadata

    @property
    def time_indices(self) -> List[int]:
        """Available time indices (sorted)."""
        return self._data.time_indices

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return self._data.n_frames

    @property
    def pixel_size_xy_um(self) -> float:
        """XY pixel size in micrometers."""
        return self.metadata.pixel_size_xy_nm / 1000.0

    @property
    def pixel_size_z_um(self) -> float:
        """Z pixel size in micrometers."""
        return self.metadata.pixel_size_z_nm / 1000.0

    # Analysis Methods (delegate to analysis.py with lazy import)

    def calculate_statistics(self, time_index: int) -> AnalysisResults:
        """
        Calculate statistics for a single frame.

        Delegates to analysis.calculate_frame_statistics().
        """
        from .analysis import calculate_frame_statistics
        return calculate_frame_statistics(self[time_index])

    def calculate_all_statistics(self) -> Dict[int, AnalysisResults]:
        """
        Calculate statistics for all frames.

        Returns:
            Dict mapping time_index -> AnalysisResults
        """
        from .analysis import calculate_frame_statistics
        return {t: calculate_frame_statistics(frame) for t, frame in self}

    # Visualization Methods (delegate to visualization.py with lazy import) 

    def plot_frame(self, time_index: int, **kwargs) -> 'plt.Figure':
        """
        Plot spatial curvature distribution for a frame.

        Delegates to visualization.basic_spatial_plot().

        Parameters:
            time_index: Frame to plot
            **kwargs: Passed to basic_spatial_plot()

        Returns:
            matplotlib Figure
        """
        from .visualization import basic_spatial_plot
        return basic_spatial_plot(self[time_index], **kwargs)

    # String Representations 

    def __str__(self) -> str:
        if self.n_frames == 0:
            return "CellAnalyzer(no frames)"
        t_min, t_max = min(self.time_indices), max(self.time_indices)
        return f"CellAnalyzer({self.n_frames} frames: T{t_min}-T{t_max})"

    def __repr__(self) -> str:
        return f"CellAnalyzer('{self._data.cell_directory}', n_frames={self.n_frames})"
