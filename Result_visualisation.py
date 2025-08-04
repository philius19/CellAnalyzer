from MeshAnalyzer import MeshAnalyzer
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.cm import RdBu
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

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

##################################################################################################################################

root = Path("/Users/philippkaintoch/Documents/Projects/01_Bleb3D/Analyze/2D/5_BAIAP2_OE")
mesh_dir = Path("Morphology/Analysis/Mesh/ch1")

surface_path = root / mesh_dir / "surface_1_1.mat"
curvature_path = root / mesh_dir / "meanCurvature_1_1.mat"

analyzer_3d = MeshAnalyzer(surface_path, curvature_path)
analyzer_3d.load_data()

##################################################################################################################################

def calculate_curvature_entropy(curvature: np.ndarray, n_bins: int = 50) -> float:
    """
    Calculate Shannon entropy of curvature distribution.
    
    Higher entropy = more diverse curvature values = complex shape
    
    Parameters:
    -----------
    curvature : np.ndarray
        Array of curvature values
    n_bins : int
        Number of bins for histogram
        
    Returns:
    --------
    float : Entropy value
    """
    # Create histogram
    hist, _ = np.histogram(curvature, bins=n_bins)
    
    # Normalize to probability distribution
    hist = hist / np.sum(hist)
    
    # Remove zeros (log(0) is undefined)
    hist = hist[hist > 0]
    
    # Calculate entropy
    entropy = -np.sum(hist * np.log(hist))
    
    return entropy

def create_individual_components_from_analyzer(analyzer, shannon_entropy,
                                              output_dir='/Users/philippkaintoch/Desktop/Results/Components', 
                                              dpi=500):
    """
    Create individual high-resolution components from MeshAnalyzer instance.
    
    Parameters:
    -----------
    analyzer : MeshAnalyzer
        Your MeshAnalyzer instance with loaded data
    shannon_entropy : float
        Shannon entropy value for histogram annotation
    output_dir : str
        Directory for output files
    dpi : int
        Resolution (500 for high quality)
    """
    
    # Check if data is loaded
    if not analyzer.is_loaded:
        raise RuntimeError("Analyzer must have data loaded. Call analyzer.load_data() first.")
    
    print("\n=== Creating Individual High-Resolution Components ===")
    print(f"Resolution: {dpi} DPI")
    
    # Create output directory
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get pixel sizes from analyzer
    pixel_size_xy = analyzer.pixel_size_xy
    pixel_size_z = analyzer.pixel_size_z
    
    # Convert mesh to physical units
    vertices_um = analyzer.vertices.copy()
    vertices_um[:, 0] *= pixel_size_xy  # X
    vertices_um[:, 1] *= pixel_size_xy  # Y
    vertices_um[:, 2] *= pixel_size_z   # Z
    
    # Convert curvature from pixel units to physical units
    curvature = analyzer.curvature
    curvature_um = curvature / pixel_size_xy  # Convert 1/pixel to 1/μm
    
    print(f"Physical dimensions: {analyzer.physical_dimensions}")
    print(f"Curvature range: [{curvature_um.min():.4f}, {curvature_um.max():.4f}] μm⁻¹")
    
    # Create face centers in physical units
    faces = analyzer.faces
    face_centers = np.mean(vertices_um[faces], axis=1)
    
    # Color mapping
    vmax = np.percentile(np.abs(curvature_um), 95)
    vmin = -vmax
    norm = Normalize(vmin=vmin, vmax=vmax)
    face_colors = RdBu(norm(curvature_um))
    triangles = vertices_um[faces]
    
    # Set consistent style
    plt.style.use('dark_background')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 20,
        'axes.linewidth': 1.5,
        'lines.linewidth': 2,
        'savefig.dpi': dpi,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'text.usetex': False,
        'mathtext.default': 'regular'
    })
    
    # 1. Create 3D Morphology Figure
    print("Creating 3D morphology figure...")
    fig_3d = plt.figure(figsize=(14, 12), facecolor='black')
    ax_3d = fig_3d.add_subplot(111, projection='3d', facecolor='black')
    
    mesh_collection = Poly3DCollection(
        triangles,
        facecolors=face_colors,
        edgecolors='none',
        alpha=0.95
    )
    ax_3d.add_collection3d(mesh_collection)
    ax_3d.set_xlim(vertices_um[:, 0].min(), vertices_um[:, 0].max())
    ax_3d.set_ylim(vertices_um[:, 1].min(), vertices_um[:, 1].max())
    ax_3d.set_zlim(vertices_um[:, 2].min(), vertices_um[:, 2].max())
    ax_3d.view_init(elev=20, azim=60)
    
    ax_3d.set_xlabel(r'X ($\mu$m)', color='white', fontsize=16)
    ax_3d.set_ylabel(r'Y ($\mu$m)', color='white', fontsize=16)
    ax_3d.set_zlabel(r'Z ($\mu$m)', color='white', fontsize=16)
    ax_3d.set_title('3D Neuron Morphology', fontsize=18, color='white', fontweight='bold')
    ax_3d.grid(False)
    ax_3d.xaxis.pane.fill = False
    ax_3d.yaxis.pane.fill = False
    ax_3d.zaxis.pane.fill = False
    
    # Add colorbar
    cbar_3d = fig_3d.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=RdBu), ax=ax_3d, shrink=0.6, aspect=20)
    cbar_3d.set_label(r'Mean Curvature ($\mu$m$^{-1}$)', fontsize=14, color='white', labelpad=15)
    cbar_3d.ax.tick_params(colors='white', labelsize=12)
    
    plt.tight_layout()
    fig_3d.savefig(output_dir / '3d_morphology.png', dpi=dpi, facecolor='black')
    plt.close(fig_3d)
    
    # 2. Create Curvature Distribution Figure with mesh-matched colors
    print("Creating curvature distribution figure...")
    fig_hist = plt.figure(figsize=(12, 10), facecolor='black')
    ax_hist = fig_hist.add_subplot(111, facecolor='black')
    
    # Use raw curvature data without conversion and without filtering
    raw_curvature = analyzer.curvature  # Raw curvature in 1/pixel units
    
    print(f"  Raw curvature range: [{raw_curvature.min():.4f}, {raw_curvature.max():.4f}] (1/pixel)")
    print(f"  Data points: {len(raw_curvature):,} (complete dataset)")
    
    # Create histogram with complete raw data
    n, bins, patches = ax_hist.hist(raw_curvature, bins=50, density=True, 
                                   color='steelblue', alpha=0.8, edgecolor='white')
    ax_hist.set_yscale('log')
    
    # Color patches with original simple coloring
    for i, patch in enumerate(patches):
        if bins[i] < 0:
            patch.set_facecolor('#FF6B6B')  # Red: cylindrical
        else:
            patch.set_facecolor('#4ECDC4')  # Cyan: protrusions
    
    # Add reference lines
    ax_hist.axvline(0, color='white', linestyle='--', linewidth=2, alpha=0.8, label='Zero Curvature')
    ax_hist.axvline(np.median(raw_curvature), color='orange', linestyle=':', linewidth=2,
                   label=f'Median: {np.median(raw_curvature):.4f} 1/pixel')
    
    # No text overlays - clean histogram for presentation
    
    ax_hist.set_xlabel(r'Mean Curvature (1/pixel)', fontsize=16, color='white')
    ax_hist.set_ylabel('Probability Density (log scale)', fontsize=16, color='white')
    ax_hist.set_title('Curvature Distribution', fontsize=18, color='white', fontweight='bold')
    ax_hist.tick_params(colors='white')
    ax_hist.spines['bottom'].set_color('white')
    ax_hist.spines['left'].set_color('white')
    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)
    ax_hist.legend(loc='upper left', frameon=True, labelcolor='white',
                   facecolor='black', edgecolor='white', framealpha=0.8)
    
    plt.tight_layout()
    fig_hist.savefig(output_dir / 'curvature_distribution.png', dpi=dpi, facecolor='black')
    plt.close(fig_hist)
    
    # 3. Create Individual Projection Figures
    projections = [
        (0, 1, 'XY Projection: Top View', 'X', 'Y', 'xy_projection.png'),
        (0, 2, 'XZ Projection: Side View', 'X', 'Z', 'xz_projection.png'),
        (1, 2, 'YZ Projection: Front View', 'Y', 'Z', 'yz_projection.png')
    ]
    
    for x_idx, y_idx, title, xlabel, ylabel, filename in projections:
        print(f"Creating {title.lower()}...")
        
        # Calculate density for optimization
        x_coords = face_centers[:, x_idx]
        y_coords = face_centers[:, y_idx]
        x_range = x_coords.max() - x_coords.min()
        y_range = y_coords.max() - y_coords.min()
        area = x_range * y_range
        density = len(face_centers) / area
        
        # Optimize point size based on density
        if density > 600:  # Very dense
            point_size = 2.5
            alpha = 0.5
        elif density > 300:  # Moderately dense
            point_size = 2.0
            alpha = 0.6
        else:  # Sparse
            point_size = 1.5
            alpha = 0.8
        
        fig_proj = plt.figure(figsize=(14, 12), facecolor='black')
        ax_proj = fig_proj.add_subplot(111, facecolor='black')
        
        scatter = ax_proj.scatter(x_coords, y_coords, c=curvature_um, 
                                s=point_size, cmap='RdBu', vmin=vmin, vmax=vmax, 
                                alpha=alpha, edgecolors='none')
        
        ax_proj.set_xlabel(f'{xlabel} (μm)', fontsize=16, color='white', fontweight='bold')
        ax_proj.set_ylabel(f'{ylabel} (μm)', fontsize=16, color='white', fontweight='bold')
        ax_proj.set_title(title, fontsize=18, color='white', fontweight='bold')
        ax_proj.set_aspect('equal')
        ax_proj.tick_params(colors='white', labelsize=14)
        ax_proj.grid(True, alpha=0.3, color='white')
        
        for spine in ax_proj.spines.values():
            spine.set_color('white')
            spine.set_linewidth(1.5)
        
        # Add colorbar
        cbar_proj = plt.colorbar(scatter, ax=ax_proj, shrink=0.8, aspect=30, pad=0.02)
        cbar_proj.set_label('Mean Curvature (μm⁻¹)', fontsize=14, color='white', 
                           fontweight='bold', labelpad=15)
        cbar_proj.ax.tick_params(colors='white', labelsize=12)
        
        plt.tight_layout()
        fig_proj.savefig(output_dir / filename, dpi=dpi, facecolor='black')
        plt.close(fig_proj)
    
    plt.rcdefaults()
    print(f"\n✅ All components saved to: {output_dir}")
    print(f"📁 Files created:")
    print(f"  - 3d_morphology.png")
    print(f"  - curvature_distribution.png")
    print(f"  - xy_projection.png")
    print(f"  - xz_projection.png")
    print(f"  - yz_projection.png")


# Usage with your MeshAnalyzer
create_individual_components_from_analyzer(
    analyzer_3d,
    calculate_curvature_entropy(analyzer_3d.curvature / analyzer_3d.pixel_size_xy),  # Convert for entropy
    output_dir='/Users/philippkaintoch/Desktop/Results/Components',
    dpi=500
)


