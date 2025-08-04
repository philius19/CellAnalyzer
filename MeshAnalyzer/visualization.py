"""
Visualization functions for mesh analysis.
"""
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt


# Set Nature-style plotting parameters
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.3,
    'ytick.minor.width': 0.3,
    'lines.linewidth': 1.0,
    'patch.linewidth': 0.5,
    'figure.dpi': 300
})

# Nature color palette
NATURE_COLORS = {
    'blue': '#2E86AB',
    'red': '#E63946', 
    'gray': '#6C757D',
    'light_gray': '#ADB5BD',
    'dark_gray': '#343A40',
    'green': '#028A0F',
    'orange': '#F77F00'
}



def plot_curvature_distribution(curvature: np.ndarray, 
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot curvature distribution with linear and log scale.
    
    Parameters:
        curvature: Array of curvature values
        save_path: Optional path to save figure
        
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Linear scale
    ax1.hist(curvature, bins=100, alpha=0.7)
    ax1.axvline(0, color=NATURE_COLORS['red'], linestyle='--', label='Zero')
    ax1.set_xlabel('Curvature (1/pixels)')
    ax1.set_ylabel('Count')
    ax1.set_title('Curvature Distribution')
    ax1.legend()

    # Log scale
    non_zero_curv = curvature[curvature != 0]
    ax2.hist(non_zero_curv, bins=100, alpha=0.7)
    ax2.set_yscale('log')
    ax2.set_xlabel('Curvature (1/pixels)')
    ax2.set_ylabel('Count (log scale)')
    ax2.set_title('Curvature Distribution (Log Scale)')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig

# ========== Enter Function for 3D Visualisation ==========
# ========== Enter Function for 3D Plot ==========

def basic_spatial_plot(mesh, curvature: np.ndarray, 
                       save_path: Optional[str] = None, 
                       title: str = "Spatial Curvature Distribution") -> plt.Figure:
    """
    Create a simple spatial visualization of curvature.
    
    This shows you where high/low curvatures are located.
    
    Parameters:
    -----------
    mesh : trimesh object
        The mesh with face centers
    curvature : np.ndarray
        Curvature values for each face
    
    Example:
    --------
    >>> basic_spatial_plot(analyzer_2d.mesh, analyzer_2d.curvature, "2D Neuron")
    """
    face_centers = mesh.triangles_center                                          # Each triangle face has a center point (centroid) --> Vertices: (A+B+C)/3
    
    fig, ax = plt.subplots(figsize=(10, 8)) 

    # Make the color scale symmetric around zero
    vmax = np.percentile(np.abs(curvature), 95)                                   # Use absolute value
    vmin = -vmax                                                                  # Make symmetric
    
    # Create scatter plot
    scatter = ax.scatter(face_centers[:, 0], face_centers[:, 1],                  # X coordinates (all rows, column 0) and Y  coordinates (all rows, column 1)
                        c=curvature, s=0.5, cmap='RdBu',                          # Color each point by its curvature value (s = size each dot)
                        vmin=vmin, vmax=vmax)                                     # Color scale limits at 5th and 95th percentiles (removes outliers from color scaling)
    
    ax.set_xlabel('X position (pixels)')
    ax.set_ylabel('Y position (pixels)')
    ax.set_title(title)
    ax.set_aspect('equal')                                                        # Equal aspect ratio: Ensures 1 pixel in X = 1 pixel in Y. Prevents distortion of the neuron shape
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)                                           # Add colorbar: Shows the color-to-curvature mapping. The colorbar is linked to the scatter plot and labeled with units.
    cbar.set_label('Mean Curvature (1/pixels)')
    
    # Add some statistics to the plot
    stats_text = f'Mean: {np.mean(curvature):.3f}\nStd: {np.std(curvature):.3f}'  # Create statistics text: Formats mean and standard deviation to 3 decimal places
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax