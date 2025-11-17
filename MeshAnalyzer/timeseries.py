"""
Time-series analysis extension for MeshAnalyzer.

Handles multi-frame (time-lapse) mesh data while maintaining backward compatibility
with single-frame analysis.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Iterator, Tuple, List
from collections import OrderedDict
import numpy as np
import warnings

from .analyzer import MeshAnalyzer
from .datatypes import AnalysisResults


@dataclass(frozen=True)
class TimeFrameInfo:
    """
    Metadata for a single time frame.

    Attributes:
        time_index: Temporal index (frame number)
        surface_path: Path to surface .mat file
        curvature_path: Path to curvature .mat file
        timestamp_sec: Optional real-time timestamp in seconds
    """
    time_index: int
    surface_path: Path
    curvature_path: Path
    timestamp_sec: Optional[float] = None

    def __post_init__(self):
        """Validate file paths after initialization."""
        if not self.surface_path.exists():
            raise FileNotFoundError(f"Surface file not found: {self.surface_path}")
        if not self.curvature_path.exists():
            raise FileNotFoundError(f"Curvature file not found: {self.curvature_path}")

    def __lt__(self, other):
        """Enable sorting by time_index."""
        return self.time_index < other.time_index


@dataclass
class TimeSeriesData:
    """
    Loaded data for a single time frame.

    Attributes:
        time_index: Temporal index
        analyzer: MeshAnalyzer instance with loaded data
        face_centers: Pre-calculated face centers (N_faces, 3)
        curvature: Copy of curvature data for quick access
    """
    time_index: int
    analyzer: MeshAnalyzer
    face_centers: np.ndarray
    curvature: np.ndarray

    @property
    def n_faces(self) -> int:
        """Number of faces in this frame."""
        return len(self.analyzer.faces)

    @property
    def n_vertices(self) -> int:
        """Number of vertices in this frame."""
        return len(self.analyzer.vertices)


class TimeSeriesManager:
    """
    Manages time-lapse mesh data with lazy loading and caching.

    Provides dict-like interface for accessing individual frames while handling
    memory management efficiently through LRU eviction.

    Example:
        >>> manager = TimeSeriesManager(data_dir, pixel_size_xy=0.103, pixel_size_z=0.217)
        >>> manager.discover_frames()
        >>> print(f"Found {len(manager)} frames")
        >>>
        >>> # Dict-like access
        >>> frame_data = manager[10]  # Load frame 10
        >>>
        >>> # Iteration
        >>> for time_idx, data in manager:
        ...     print(f"T{time_idx}: {data.n_faces} faces")
        >>>
        >>> # Global normalization
        >>> curvature_norm = manager.get_normalized_curvature()
    """

    def __init__(
        self,
        data_dir: Path,
        pixel_size_xy: float,
        pixel_size_z: float,
        cache_mode: str = 'lazy',
        max_cached_frames: int = 10,
        verbose: bool = True
    ):
        """
        Initialize TimeSeriesManager.

        Parameters:
            data_dir: Directory containing surface_*.mat and meanCurvature_*.mat files
            pixel_size_xy: XY pixel size in micrometers
            pixel_size_z: Z pixel size in micrometers
            cache_mode: 'none' (no caching), 'lazy' (load on demand), 'all' (preload)
            max_cached_frames: Maximum frames to keep in cache (for 'lazy' mode)
            verbose: Print progress messages

        Raises:
            ValueError: If cache_mode is invalid
        """
        self.data_dir = Path(data_dir)
        self.pixel_size_xy = pixel_size_xy
        self.pixel_size_z = pixel_size_z
        self.verbose = verbose

        # Validate cache mode
        if cache_mode not in ('none', 'lazy', 'all'):
            raise ValueError(f"cache_mode must be 'none', 'lazy', or 'all', got: {cache_mode}")
        self.cache_mode = cache_mode
        self.max_cached_frames = max_cached_frames

        # Storage for frame metadata (always in memory)
        self._frame_info: OrderedDict[int, TimeFrameInfo] = OrderedDict()

        # Cache for loaded data (managed by cache_mode)
        self._loaded_frames: OrderedDict[int, TimeSeriesData] = OrderedDict()

        # Access tracking for LRU eviction
        self._access_order: List[int] = []

        # Global statistics cache
        self._global_stats_cache: Optional[Dict] = None

    def discover_frames(self, pattern: str = 'surface_*.mat') -> int:
        """
        Discover all time frames in data directory.

        Follows the naming convention from test.py:
        - surface_<channel>_<timeframe>.mat
        - meanCurvature_<channel>_<timeframe>.mat

        Parameters:
            pattern: Glob pattern for surface files

        Returns:
            Number of frames discovered

        Raises:
            FileNotFoundError: If no surface files found
            ValueError: If paired curvature files are missing
        """
        if self.verbose:
            print("=" * 70)
            print("DISCOVERING TIME-SERIES FRAMES".center(70))
            print("=" * 70)
            print(f"\nSearching in: {self.data_dir}")

        surface_files = sorted(self.data_dir.glob(pattern))

        if not surface_files:
            raise FileNotFoundError(f"No surface files found in {self.data_dir}")

        if self.verbose:
            print(f"Found {len(surface_files)} surface files\n")

        discovered_frames = []

        for surface_path in surface_files:
            # Parse filename: surface_1_10.mat → channel=1, time_index=10
            try:
                parts = surface_path.stem.split('_')
                if len(parts) < 3:
                    raise ValueError("Invalid filename format")

                time_index = int(parts[2])

                # Construct paired curvature file path
                curv_path = surface_path.parent / surface_path.name.replace('surface', 'meanCurvature')

                if not curv_path.exists():
                    if self.verbose:
                        print(f"  ⚠ Skipping T{time_index:02d}: Missing {curv_path.name}")
                    continue

                # Create frame info
                frame_info = TimeFrameInfo(
                    time_index=time_index,
                    surface_path=surface_path,
                    curvature_path=curv_path
                )

                discovered_frames.append((time_index, frame_info))

                if self.verbose:
                    print(f"  ✓ T{time_index:02d}: {surface_path.name}")

            except (IndexError, ValueError) as e:
                if self.verbose:
                    print(f"  ⚠ Skipping {surface_path.name}: Invalid filename format")
                continue

        # Store in chronological order
        self._frame_info = OrderedDict(sorted(discovered_frames))

        if not self._frame_info:
            raise ValueError("No valid time frames found!")

        if self.verbose:
            time_indices = list(self._frame_info.keys())
            print(f"\n{'SUCCESS'.center(70)}")
            print(f"Discovered {len(self._frame_info)} frames: "
                  f"T{min(time_indices):02d} - T{max(time_indices):02d}")
            print("=" * 70)

        # Preload all frames if cache_mode is 'all'
        if self.cache_mode == 'all':
            self._preload_all_frames()

        return len(self._frame_info)

    def _preload_all_frames(self) -> None:
        """Preload all frames into cache (for cache_mode='all')."""
        if self.verbose:
            print("\n" + "=" * 70)
            print("PRELOADING ALL FRAMES".center(70))
            print("=" * 70 + "\n")

        for time_idx in self._frame_info.keys():
            data = self._load_frame_internal(time_idx)
            self._loaded_frames[time_idx] = data

        if self.verbose:
            print(f"\n✓ Preloaded {len(self._loaded_frames)} frames")
            print("=" * 70)

    def _load_frame_internal(self, time_index: int) -> TimeSeriesData:
        """
        Load a single frame into memory.

        Parameters:
            time_index: Frame to load

        Returns:
            TimeSeriesData for the frame
        """
        frame_info = self._frame_info[time_index]

        if self.verbose:
            print(f"  Loading T{time_index:02d}...")

        # Create MeshAnalyzer instance
        analyzer = MeshAnalyzer(
            str(frame_info.surface_path),
            str(frame_info.curvature_path),
            pixel_size_xy=self.pixel_size_xy,
            pixel_size_z=self.pixel_size_z
        )
        analyzer.load_data(verbose=False)

        # Pre-calculate face centers for visualization
        vertices = analyzer.vertices
        faces = analyzer.faces
        face_centers = np.array([np.mean(vertices[face], axis=0) for face in faces])

        # Create TimeSeriesData
        data = TimeSeriesData(
            time_index=time_index,
            analyzer=analyzer,
            face_centers=face_centers,
            curvature=analyzer.curvature.copy()
        )

        return data

    def _evict_lru_frame(self) -> None:
        """Evict least recently used frame from cache."""
        if not self._access_order:
            return

        # Find LRU frame that's still in cache
        for time_idx in self._access_order:
            if time_idx in self._loaded_frames:
                if self.verbose:
                    print(f"  Evicting T{time_idx:02d} (LRU)")
                del self._loaded_frames[time_idx]
                self._access_order.remove(time_idx)
                break

    def load_frame(self, time_index: int) -> TimeSeriesData:
        """
        Load a specific time frame.

        Handles caching according to cache_mode:
        - 'none': Always reload
        - 'lazy': Cache with LRU eviction
        - 'all': All frames preloaded

        Parameters:
            time_index: Frame to load

        Returns:
            TimeSeriesData for the frame

        Raises:
            KeyError: If time_index not found
        """
        if time_index not in self._frame_info:
            available = list(self._frame_info.keys())
            raise KeyError(f"Time index {time_index} not found. Available: {available}")

        # Update access order for LRU
        if time_index in self._access_order:
            self._access_order.remove(time_index)
        self._access_order.append(time_index)

        # Check cache
        if time_index in self._loaded_frames:
            return self._loaded_frames[time_index]

        # Load frame
        data = self._load_frame_internal(time_index)

        # Cache management
        if self.cache_mode == 'none':
            # Don't cache
            return data

        elif self.cache_mode == 'lazy':
            # Cache with LRU eviction
            if len(self._loaded_frames) >= self.max_cached_frames:
                self._evict_lru_frame()

            self._loaded_frames[time_index] = data

        elif self.cache_mode == 'all':
            # All frames preloaded, just store
            self._loaded_frames[time_index] = data

        return data

    def validate_frames(self) -> Dict[str, any]:
        """
        Validate consistency across all frames.

        Checks:
        - Topology consistency (same number of vertices/faces)
        - Data quality (no NaN values)
        - Temporal continuity (no missing frames)

        Returns:
            Dictionary with validation results
        """
        if not self._frame_info:
            raise RuntimeError("No frames discovered. Call discover_frames() first.")

        if self.verbose:
            print("\n" + "=" * 70)
            print("VALIDATING TIME-SERIES CONSISTENCY".center(70))
            print("=" * 70 + "\n")

        results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'topology_consistent': True,
            'temporal_gaps': []
        }

        # Load first frame to get reference topology
        first_idx = next(iter(self._frame_info.keys()))
        first_data = self.load_frame(first_idx)
        ref_n_vertices = first_data.n_vertices
        ref_n_faces = first_data.n_faces

        if self.verbose:
            print(f"Reference topology (T{first_idx:02d}):")
            print(f"  • Vertices: {ref_n_vertices:,}")
            print(f"  • Faces: {ref_n_faces:,}\n")
            print("Checking all frames...")

        # Check each frame
        for time_idx in self._frame_info.keys():
            data = self.load_frame(time_idx)

            # Topology check - treat as warning for biological data
            # (cells naturally change topology as they move and deform)
            if data.n_vertices != ref_n_vertices or data.n_faces != ref_n_faces:
                msg = (f"T{time_idx:02d}: Topology mismatch "
                       f"({data.n_vertices} vertices, {data.n_faces} faces)")
                results['warnings'].append(msg)
                results['topology_consistent'] = False

                if self.verbose:
                    print(f"  ⚠ {msg}")

            # Data quality check - NaN values are critical errors
            if np.any(np.isnan(data.curvature)):
                msg = f"T{time_idx:02d}: NaN values in curvature"
                results['errors'].append(msg)
                results['is_valid'] = False

                if self.verbose:
                    print(f"  ✗ {msg}")

            elif self.verbose:
                print(f"  ✓ T{time_idx:02d}: OK")

        # Check temporal continuity
        time_indices = sorted(self._frame_info.keys())
        for i in range(len(time_indices) - 1):
            gap = time_indices[i + 1] - time_indices[i]
            if gap > 1:
                results['temporal_gaps'].append((time_indices[i], time_indices[i + 1]))

        if results['temporal_gaps']:
            msg = f"Temporal gaps detected: {results['temporal_gaps']}"
            results['warnings'].append(msg)

            if self.verbose:
                print(f"\n  ⚠ {msg}")

        if self.verbose:
            print(f"\n{'VALIDATION COMPLETE'.center(70)}")
            if results['is_valid']:
                print("✓ All frames valid".center(70))
            else:
                print(f"✗ {len(results['errors'])} errors found".center(70))
            print("=" * 70)

        return results

    def get_normalized_curvature(
        self,
        method: str = 'symmetric',
        percentile_range: Optional[Tuple[float, float]] = None
    ) -> Dict[int, np.ndarray]:
        """
        Get globally normalized curvature across all frames.

        Parameters:
            method: 'symmetric' (symmetric around 0) or 'full' (min-max)
            percentile_range: Optional (low, high) percentiles for robust scaling
                            e.g., (5, 95) to exclude outliers

        Returns:
            Dictionary mapping time_index -> normalized curvature array
        """
        if not self._frame_info:
            raise RuntimeError("No frames discovered. Call discover_frames() first.")

        if self.verbose:
            print("\n" + "=" * 70)
            print("CALCULATING GLOBAL CURVATURE NORMALIZATION".center(70))
            print("=" * 70 + "\n")

        # Collect all curvature values
        all_curvatures = []
        for time_idx in self._frame_info.keys():
            data = self.load_frame(time_idx)
            all_curvatures.append(data.curvature)

        all_curv_concat = np.concatenate(all_curvatures)

        # Calculate normalization bounds
        if percentile_range:
            low, high = percentile_range
            vmin = np.percentile(all_curv_concat, low)
            vmax = np.percentile(all_curv_concat, high)

            if self.verbose:
                print(f"Using percentile range: [{low}, {high}]")
                print(f"  • Raw range: [{np.min(all_curv_concat):.6f}, {np.max(all_curv_concat):.6f}]")
                print(f"  • Percentile range: [{vmin:.6f}, {vmax:.6f}]")
        else:
            vmin = np.min(all_curv_concat)
            vmax = np.max(all_curv_concat)

            if self.verbose:
                print(f"Using full range: [{vmin:.6f}, {vmax:.6f}]")

        # Apply normalization method
        if method == 'symmetric':
            # Make symmetric around zero
            vmax_sym = max(abs(vmin), abs(vmax))
            vmin_sym = -vmax_sym

            if self.verbose:
                print(f"Symmetric normalization: [{vmin_sym:.6f}, {vmax_sym:.6f}]")

            normalized = {}
            for time_idx, curv in zip(self._frame_info.keys(), all_curvatures):
                # Normalize to [-1, +1]
                normalized[time_idx] = curv / vmax_sym

        elif method == 'full':
            # Min-max normalization
            if self.verbose:
                print(f"Min-max normalization: [0, 1]")

            normalized = {}
            for time_idx, curv in zip(self._frame_info.keys(), all_curvatures):
                # Normalize to [0, 1]
                normalized[time_idx] = (curv - vmin) / (vmax - vmin)

        else:
            raise ValueError(f"Unknown method: {method}. Use 'symmetric' or 'full'")

        if self.verbose:
            print("=" * 70)

        return normalized

    # ========== DICT-LIKE INTERFACE ==========

    def __getitem__(self, time_index: int) -> TimeSeriesData:
        """Enable dictionary-style access: manager[10]."""
        return self.load_frame(time_index)

    def __iter__(self) -> Iterator[Tuple[int, TimeSeriesData]]:
        """Enable iteration: for time_idx, data in manager."""
        for time_idx in self._frame_info.keys():
            yield time_idx, self.load_frame(time_idx)

    def __len__(self) -> int:
        """Return number of frames."""
        return len(self._frame_info)

    def __contains__(self, time_index: int) -> bool:
        """Check if time_index exists: if 10 in manager."""
        return time_index in self._frame_info

    def keys(self) -> Iterator[int]:
        """Return iterator over time indices."""
        return iter(self._frame_info.keys())

    def items(self) -> Iterator[Tuple[int, TimeSeriesData]]:
        """Return iterator over (time_index, data) pairs."""
        return self.__iter__()

    # ========== STRING REPRESENTATION ==========

    def __str__(self) -> str:
        """String representation."""
        if not self._frame_info:
            return "TimeSeriesManager(no frames)"

        time_indices = list(self._frame_info.keys())
        return (f"TimeSeriesManager({len(self)} frames: "
                f"T{min(time_indices):02d}-T{max(time_indices):02d}, "
                f"cache_mode='{self.cache_mode}')")

    def __repr__(self) -> str:
        """Detailed representation."""
        return (f"TimeSeriesManager(data_dir='{self.data_dir}', "
                f"n_frames={len(self)}, cache_mode='{self.cache_mode}', "
                f"cached={len(self._loaded_frames)})")

    # ========== CACHE MANAGEMENT ==========

    def clear_cache(self) -> None:
        """Clear all cached frames."""
        self._loaded_frames.clear()
        self._access_order.clear()
        self._global_stats_cache = None

        if self.verbose:
            print("Cache cleared")

    def get_cache_stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        return {
            'cache_mode': self.cache_mode,
            'max_cached_frames': self.max_cached_frames,
            'currently_cached': len(self._loaded_frames),
            'cached_indices': list(self._loaded_frames.keys()),
            'total_frames': len(self._frame_info)
        }
