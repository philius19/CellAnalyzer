"""
Utility functions for mesh analysis.
"""
from typing import Dict
import numpy as np
import trimesh


def convert_pixels_to_um(value: float, pixel_size: float) -> float:
    """Convert pixel units to micrometers."""
    return value * pixel_size


def calculate_mesh_quality_metrics(mesh: trimesh.Trimesh, verbose: bool = False) -> Dict:
    """
    Calculate mesh quality metrics.
    
    Parameters:
        mesh: Trimesh object
        
    Returns:
        Dictionary with quality metrics
    """
    # Triangle areas
    areas = mesh.area_faces
    
    # Edge lengths
    edge_lengths = mesh.edges_unique_length

    if verbose:
        print("=== MESH INTEGRITY CHECK ===")
        print(f"Is watertight (closed): {mesh.is_watertight}")
        print(f"Is winding consistent: {mesh.is_winding_consistent}")
        print(f"Has vertex normals: {mesh.vertex_normals is not None}")
        print(f"Volume: {mesh.volume}")
        print(f"Euler number: {mesh.euler_number}")
    
    return {
        'triangle_area': {
            'mean': float(np.mean(areas)),
            'std': float(np.std(areas)),
            'min': float(np.min(areas)),
            'max': float(np.max(areas))
        },
        'edge_length': {
            'mean': float(np.mean(edge_lengths)),
            'std': float(np.std(edge_lengths)),
            'min': float(np.min(edge_lengths)),
            'max': float(np.max(edge_lengths))
        },
        'aspect_ratio': float(np.max(edge_lengths) / np.min(edge_lengths))
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