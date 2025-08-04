"""
Utility functions for mesh analysis.
"""
from typing import Dict
import numpy as np
import vedo


def convert_pixels_to_um(value: float, pixel_size: float) -> float:
    """Convert pixel units to micrometers."""
    return value * pixel_size


def calculate_mesh_quality_metrics(mesh: vedo.Mesh, verbose: bool = False) -> Dict:
    """
    Calculate mesh quality metrics.
    
    Parameters:
        mesh: vedo Mesh object
        
    Returns:
        Dictionary with quality metrics
    """
    # Simplified quality metrics for vedo
    # Just return basic statistics
    
    if verbose:
        print("=== MESH INTEGRITY CHECK ===")
        print(f"Is watertight (closed): {mesh.is_closed()}")
        print(f"Volume: {mesh.volume()}")
        print(f"Euler number: {mesh.euler_characteristic()}")
    
    # Return simplified metrics
    return {
        'simplified': True,
        'note': 'Using simplified metrics with vedo'
    }



def calculate_surface_roughness(curvature: np.ndarray) -> float:
    """Calculate surface roughness from curvature."""
    return float(np.std(np.abs(curvature)))


def find_high_curvature_regions(curvature: np.ndarray, 
                               threshold: float = 2.0) -> np.ndarray:
    """
    Find indices of high curvature regions.
    
    Parameters:
        curvature: Curvature values
        threshold: Threshold for high curvature
        
    Returns:
        Boolean mask of high curvature vertices
    """
    return np.abs(curvature) > threshold