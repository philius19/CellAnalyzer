from mat73 import loadmat
from pathlib import Path
import vedo
from typing import Tuple, List
import numpy as np
import json

"""
File input/output operations for mesh analysis. Maybe rework to store all of the data in one single Library? 
"""

def validate_file_paths(surface_path: Path, curvature_path: Path, 
                       supported_formats: List[str]) -> None:
    """
    Validate input files exist and have correct format.
    
    Raises:
        FileNotFoundError: If files don't exist
        ValueError: If format not supported
    """
    for path, name in [(surface_path, "Surface"), (curvature_path, "Curvature")]:
        if not path.exists():
            raise FileNotFoundError(f"{name} file not found: {path}")
        
        if path.suffix not in supported_formats:
            raise ValueError(
                f"{name} file format {path.suffix} not supported. "
                f"Supported formats: {supported_formats}"
            )


def load_surface_data(filepath: Path) -> Tuple[np.ndarray, np.ndarray, vedo.Mesh]:
    """
    Load surface mesh data from .mat file.
    
    Returns:
        Tuple of (vertices, faces, mesh)
    """
    surface_data = loadmat(str(filepath))
    surface = surface_data['surface']
    
    vertices = np.array(surface['vertices'], dtype=np.float32)
    faces = np.array(surface['faces'], dtype=np.int32) - 1  # Convert to 0-based
    mesh = vedo.Mesh([vertices, faces])
    
    return vertices, faces, mesh


def load_curvature_data(filepath: Path, expected_length: int) -> np.ndarray:

    curv_data = loadmat(str(filepath))
    curvature = np.array(curv_data['meanCurvature']).flatten()


    
    if len(curvature) != expected_length:
        raise ValueError(
            f"Curvature length ({len(curvature)}) doesn't match "
            f"expected count ({expected_length})"
        )
    
    return curvature

def load_curvature_data_raw(filepath: Path) -> np.ndarray:

    curv_raw_data = loadmat(str(filepath))
    curv_raw_data = np.array(curv_raw_data['meanCurvatureUnsmoothed']).flatten()

    return curv_raw_data


def load_gauss_data(filepath: Path) -> np.ndarray:


    gauss_data = loadmat(str(filepath))
    gauss_data = np.array(gauss_data['gaussCurvatureUnsmoothed']).flatten()

    return gauss_data


def save_mesh_to_ply(mesh, filepath: Path) -> None:
    """Export mesh to PLY format."""
    mesh.export(str(filepath))


def save_results_to_json(results: dict, filepath: Path) -> None:
    """Save analysis results to JSON."""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)