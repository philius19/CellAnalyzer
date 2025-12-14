"""
Data loading functions for u-shape3D mesh analysis.

Loads:
- MATLAB .mat files (v7.3 HDF5 and legacy formats)
- Metadata from metadata_export.mat
- Surface meshes (vertices + faces)
- Curvature data
- Auxiliary data (optional)
"""
from pathlib import Path
from typing import Tuple, List, Optional
from datetime import datetime
import json

import numpy as np
import scipy.io as sio
import vedo
from mat73 import loadmat as loadmat_v73

from .datatypes import (
    MeshParameters,
    ProcessingMetadata,
    AuxiliaryMeshData,
    MeshFrame,
    DEFAULT_MESH_PARAMETERS,
    DEFAULT_PIXEL_SIZE_XY_NM,
    DEFAULT_PIXEL_SIZE_Z_NM,
    DEFAULT_PIXEL_SIZE_XY_UM,
    DEFAULT_PIXEL_SIZE_Z_UM
)
from ._validation import (
    validate_vertices,
    validate_faces,
    validate_curvature,
    validate_face_indices_safe_for_int32
)

import logging
logger = logging.getLogger(__name__)
import re


def loadmat(filepath: str) -> dict:
    """Load MATLAB file, automatically handling both v7.3 (HDF5) and older formats."""
    try:
        return loadmat_v73(filepath)
    except (TypeError, OSError):
        return sio.loadmat(filepath, struct_as_record=False, squeeze_me=True)


def load_metadata_file(cell_dir: Path) -> ProcessingMetadata:
    """Load metadata from metadata_export.mat."""
    metadata_path = cell_dir / 'metadata_export.mat'

    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata_export.mat not found in {cell_dir}")

    data = loadmat(str(metadata_path))
    m = data['metadata']

    # Scalar extraction helper
    def scalar(key, default=None):
        val = m.get(key, default)
        if val is None:
            return default
        arr = np.asarray(val).squeeze()
        if arr.size == 1:
            return float(arr.flat[0]) if arr.ndim > 0 else float(arr)
        return float(arr)

    # Extract acquisition params
    pixel_size_xy_nm = scalar('pixel_size_xy_nm', DEFAULT_PIXEL_SIZE_XY_NM)
    pixel_size_z_nm = scalar('pixel_size_z_nm', DEFAULT_PIXEL_SIZE_Z_NM)
    time_interval_sec = scalar('time_interval_sec', 1.0)

    # Extract provenance
    source_image_path = str(m.get('source_image_path', ''))
    matlab_version = str(m.get('matlab_version', ''))
    processing_date_str = m.get('processing_date', None)

    processing_date = None
    if processing_date_str:
        try:
            processing_date = datetime.fromisoformat(str(processing_date_str))
        except:
            pass

    # Extract mesh parameters
    mp = m.get('mesh_parameters', {})

    def param(key, default_val):
        val = mp.get(key, default_val)
        if isinstance(default_val, bool):
            return bool(val)
        if isinstance(default_val, (int, np.integer)):
            arr = np.asarray(val).squeeze()
            return int(arr) if arr.size == 1 else arr.tolist()
        if isinstance(default_val, (float, np.floating)):
            arr = np.asarray(val).squeeze()
            return float(arr) if arr.size == 1 else arr.tolist()
        if isinstance(default_val, str):
            return str(val)
        if isinstance(default_val, (list, np.ndarray)):
            return np.asarray(val).squeeze().tolist() if hasattr(val, 'squeeze') else val
        return val

    mesh_parameters = MeshParameters(
        mesh_mode=param('mesh_mode', DEFAULT_MESH_PARAMETERS.mesh_mode),
        inside_gamma=param('inside_gamma', DEFAULT_MESH_PARAMETERS.inside_gamma),
        inside_blur=param('inside_blur', DEFAULT_MESH_PARAMETERS.inside_blur),
        filter_scales=param('filter_scales', DEFAULT_MESH_PARAMETERS.filter_scales),
        filter_num_std_surface=param('filter_num_std_surface', DEFAULT_MESH_PARAMETERS.filter_num_std_surface),
        inside_dilate_radius=param('inside_dilate_radius', DEFAULT_MESH_PARAMETERS.inside_dilate_radius),
        inside_erode_radius=param('inside_erode_radius', DEFAULT_MESH_PARAMETERS.inside_erode_radius),
        smooth_mesh_mode=param('smooth_mesh_mode', DEFAULT_MESH_PARAMETERS.smooth_mesh_mode),
        smooth_mesh_iterations=param('smooth_mesh_iterations', DEFAULT_MESH_PARAMETERS.smooth_mesh_iterations),
        use_undeconvolved=param('use_undeconvolved', DEFAULT_MESH_PARAMETERS.use_undeconvolved),
        image_gamma=param('image_gamma', DEFAULT_MESH_PARAMETERS.image_gamma),
        scale_otsu=param('scale_otsu', DEFAULT_MESH_PARAMETERS.scale_otsu),
        smooth_image_size=param('smooth_image_size', DEFAULT_MESH_PARAMETERS.smooth_image_size),
        curvature_median_filter_radius=param('curvature_median_filter_radius', DEFAULT_MESH_PARAMETERS.curvature_median_filter_radius),
        curvature_smooth_on_mesh_iterations=param('curvature_smooth_on_mesh_iterations', DEFAULT_MESH_PARAMETERS.curvature_smooth_on_mesh_iterations),
        register_images=param('register_images', DEFAULT_MESH_PARAMETERS.register_images),
        save_raw_images=param('save_raw_images', DEFAULT_MESH_PARAMETERS.save_raw_images),
        registration_mode=param('registration_mode', DEFAULT_MESH_PARAMETERS.registration_mode)
    )

    return ProcessingMetadata(
        pixel_size_xy_nm=pixel_size_xy_nm,
        pixel_size_z_nm=pixel_size_z_nm,
        time_interval_sec=time_interval_sec,
        source_image_path=source_image_path,
        source_image_name=Path(source_image_path).name if source_image_path else '',
        processing_date=processing_date,
        matlab_version=matlab_version,
        mesh_parameters=mesh_parameters
    )


def validate_file_paths(surface_path: Path, curvature_path: Path,
                       supported_formats: List[str]) -> None:
    """Validate input files exist and have correct format."""
    for path, name in [(surface_path, "Surface"), (curvature_path, "Curvature")]:
        if not path.exists():
            raise FileNotFoundError(f"{name} file not found: {path}")

        if path.suffix not in supported_formats:
            raise ValueError(f"{name} file format {path.suffix} not supported. "
                           f"Supported formats: {supported_formats}")


def load_surface_data(filepath: Path) -> Tuple[np.ndarray, np.ndarray, vedo.Mesh]:
    """
    Load surface mesh data from .mat file with comprehensive validation.

    Handles both MATLAB v7.3 (HDF5) and older formats. Converts MATLAB's
    1-indexed faces to Python's 0-indexed. Validates data integrity.

    Parameters:
        filepath: Path to surface .mat file

    Returns:
        Tuple of (vertices, faces, vedo_mesh)
            - vertices: Nx3 float32 array of vertex coordinates
            - faces: Mx3 int64 array of face indices (0-indexed)
            - vedo_mesh: vedo.Mesh object

    Raises:
        ValueError: If data is invalid or corrupted
    """
    surface_data = loadmat(str(filepath))
    surface = surface_data['surface']

    # Extract vertices and faces from either dict or object format
    if isinstance(surface, dict):
        vertices_raw = np.array(surface['vertices'], dtype=np.float32)
        faces_matlab = np.array(surface['faces'], dtype=np.int64)
    else:
        vertices_raw = np.array(surface.vertices, dtype=np.float32)
        faces_matlab = np.array(surface.faces, dtype=np.int64)

    # Validate vertices before processing
    validate_vertices(vertices_raw)

    # Convert MATLAB 1-indexed to Python 0-indexed
    # Do this BEFORE any potential int32 cast to avoid overflow
    faces_python = faces_matlab - 1

    # Validate faces with vertex count
    validate_faces(faces_python, len(vertices_raw))

    # Check if safe for int32 (for memory efficiency with vedo)
    # Most meshes fit in int32, but we keep int64 for safety
    if not validate_face_indices_safe_for_int32(faces_python):
        # Keep as int64 for very large meshes
        faces = faces_python
    else:
        # Safe to cast to int32 for memory efficiency
        faces = faces_python.astype(np.int32)

    vertices = vertices_raw
    mesh = vedo.Mesh([vertices, faces])
    return vertices, faces, mesh


def load_curvature_data(filepath: Path, expected_length: int) -> np.ndarray:
    """
    Load mean curvature data with comprehensive validation.

    Parameters:
        filepath: Path to curvature .mat file
        expected_length: Expected number of faces (for validation)

    Returns:
        1D array of curvature values (one per face)

    Raises:
        ValueError: If curvature data is invalid or length mismatch
    """
    curv_data = loadmat(str(filepath))
    curvature = np.array(curv_data['meanCurvature']).flatten()

    # Comprehensive validation
    validate_curvature(curvature, expected_length)

    return curvature


def load_curvature_data_raw(filepath: Path) -> np.ndarray:
    """Load raw (unsmoothed) mean curvature data."""
    curv_raw_data = loadmat(str(filepath))
    return np.array(curv_raw_data['meanCurvatureUnsmoothed']).flatten()


def load_gauss_data(filepath: Path) -> np.ndarray:
    """Load Gaussian curvature data."""
    gauss_data = loadmat(str(filepath))
    return np.array(gauss_data['gaussCurvatureUnsmoothed']).flatten()


def load_face_normals(filepath: Path) -> np.ndarray:
    """Load face normal vectors."""
    data = loadmat(str(filepath))
    return np.array(data['faceNormals'], dtype=np.float32)


def load_neighbors(filepath: Path) -> np.ndarray:
    """Load face neighbor adjacency matrix."""
    data = loadmat(str(filepath))
    return np.array(data['neighbors'])


def load_image_surface(filepath: Path) -> np.ndarray:
    """Load binary segmentation mask."""
    data = loadmat(str(filepath))
    return np.array(data['imageSurface'])


def load_auxiliary_data(mesh_dir: Path, channel: int, time_index: int) -> AuxiliaryMeshData:
    """
    Load all auxiliary data for a single timepoint.

    Parameters:
        mesh_dir: Path to Morphology/Analysis/Mesh/ch{channel}/
        channel: Channel number
        time_index: Time index

    Returns:
        AuxiliaryMeshData with all optional fields
    """
    base_pattern = f"{channel}_{time_index}"

    def safe_load(filename, loader_func):
        filepath = mesh_dir / filename
        if filepath.exists():
            try:
                return loader_func(filepath)
            except Exception as e:
                logger.warning(f"Failed to load {filename}: {e}")
                return None
        return None

    return AuxiliaryMeshData(
        face_normals=safe_load(f'faceNormals_{base_pattern}.mat', load_face_normals),
        gaussian_curvature=safe_load(f'gaussCurvatureUnsmoothed_{base_pattern}.mat', load_gauss_data),
        mean_curvature_raw=safe_load(f'meanCurvatureUnsmoothed_{base_pattern}.mat', load_curvature_data_raw),
        neighbors=safe_load(f'neighbors_{base_pattern}.mat', load_neighbors),
        image_surface=safe_load(f'imageSurface_{base_pattern}.mat', load_image_surface)
    )


def load_mesh_frame(
    surface_path: Path,
    curvature_path: Path,
    pixel_size_xy: float = DEFAULT_PIXEL_SIZE_XY_UM,
    pixel_size_z: float = DEFAULT_PIXEL_SIZE_Z_UM,
    metadata: Optional[ProcessingMetadata] = None,
    load_auxiliary: bool = False
) -> MeshFrame:
    """
    Load complete mesh frame data from MATLAB files.

    Unified loader for mesh frame data from MATLAB files.
    Automatically handles mesh validation and normal correction.

    Parameters:
        surface_path: Path to surface_*.mat file
        curvature_path: Path to meanCurvature_*.mat file
        pixel_size_xy: XY pixel size in micrometers (default from constants)
        pixel_size_z: Z pixel size in micrometers (default from constants)
        metadata: Optional ProcessingMetadata (auto-loaded if available)
        load_auxiliary: Load auxiliary data (normals, Gaussian curvature, etc.)

    Returns:
        MeshFrame with all data loaded and validated

    Example:
        frame = load_mesh_frame(
            Path('surface_1_1.mat'),
            Path('meanCurvature_1_1.mat'),
            pixel_size_xy=0.103,
            pixel_size_z=0.217
        )
    """
    # Load geometry
    surface_path = Path(surface_path)
    curvature_path = Path(curvature_path)
    vertices, faces, mesh = load_surface_data(surface_path)
    curvature = load_curvature_data(curvature_path, len(faces))

    # Auto-correct inverted normals
    if mesh.volume() < 0:
        logger.debug("Negative mesh volume detected, reversing normals")
        mesh = mesh.clone().reverse()

    # Extract time index from filename (e.g., surface_1_42.mat -> time_index=42)
    match = re.search(r'_(\d+)_(\d+)\.mat$', surface_path.name)
    time_index = int(match.group(2)) if match else 0

    # Load auxiliary data if requested
    auxiliary = None
    if load_auxiliary:
        match_ch_time = re.search(r'_(\d+)_(\d+)\.mat$', surface_path.name)
        if match_ch_time:
            channel = int(match_ch_time.group(1))
            time_idx = int(match_ch_time.group(2))
            auxiliary = load_auxiliary_data(surface_path.parent, channel, time_idx)

    # Create metadata if not provided (use defaults)
    if metadata is None:
        metadata = ProcessingMetadata(
            pixel_size_xy_nm=pixel_size_xy * 1000.0,
            pixel_size_z_nm=pixel_size_z * 1000.0,
            time_interval_sec=1.0,
            source_image_path='',
            source_image_name='',
            processing_date=None,
            matlab_version=None,
            mesh_parameters=DEFAULT_MESH_PARAMETERS
        )

    return MeshFrame(
        vertices=vertices,
        faces=faces,
        curvature=curvature,
        mesh=mesh,
        time_index=time_index,
        metadata=metadata,
        auxiliary=auxiliary
    )


def save_mesh_to_ply(mesh, filepath: Path) -> None:
    """Export mesh to PLY format."""
    mesh.export(str(filepath))


def save_results_to_json(results: dict, filepath: Path) -> None:
    """Save analysis results to JSON."""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)


def load_cell_directory(cell_dir: Path, channel: int = 1,
                       load_auxiliary: bool = False) -> 'CellData':
    """
    Load complete cell data from u-shape3D output directory.

    Loads ALL frames upfront into memory (eager loading).

    Parameters:
        cell_dir: Path to cell directory
        channel: Channel number (default: 1)
        load_auxiliary: Load auxiliary data (normals, Gaussian curv, etc.)

    Returns:
        CellData with all frames loaded

    Raises:
        FileNotFoundError: If MovieData file or mesh directory not found
        ValueError: If no valid frames discovered
    """
    from .datatypes import CellData

    cell_dir = Path(cell_dir)

    # Load metadata from metadata_export.mat
    metadata = load_metadata_file(cell_dir)

    # Get mesh directory
    mesh_dir = cell_dir / 'Morphology' / 'Analysis' / 'Mesh' / f'ch{channel}'
    if not mesh_dir.exists():
        raise FileNotFoundError(f"Mesh directory not found: {mesh_dir}")

    # Discover frames
    surface_files = sorted(mesh_dir.glob('surface_*.mat'))
    if not surface_files:
        raise FileNotFoundError(f"No surface files found in {mesh_dir}")

    pattern_regex = re.compile(r'surface_(\d+)_(\d+)\.mat')
    frames = {}

    pixel_size_xy = metadata.pixel_size_xy_nm / 1000.0
    pixel_size_z = metadata.pixel_size_z_nm / 1000.0

    for surface_path in surface_files:
        match = pattern_regex.match(surface_path.name)
        if not match:
            continue

        try:
            time_index = int(match.groups()[1])
            curv_path = surface_path.parent / surface_path.name.replace('surface', 'meanCurvature')

            if not curv_path.exists():
                logger.debug(f"Skipping T{time_index}: missing curvature file")
                continue

            frame = load_mesh_frame(
                surface_path=surface_path,
                curvature_path=curv_path,
                pixel_size_xy=pixel_size_xy,
                pixel_size_z=pixel_size_z,
                metadata=metadata,
                load_auxiliary=load_auxiliary
            )
            frames[time_index] = frame

        except Exception as e:
            logger.warning(f"Failed to load frame from {surface_path.name}: {e}")
            continue

    if not frames:
        raise ValueError("No valid frames loaded")

    logger.info(f"Loaded {len(frames)} frames from {cell_dir}")

    return CellData(
        cell_directory=cell_dir,
        metadata=metadata,
        frames=frames
    )