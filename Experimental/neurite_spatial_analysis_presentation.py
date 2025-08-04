#!/usr/bin/env python3
"""
Neurite Spatial Analysis - Presentation Style
Adapted to match your presentation color scheme:
- Red (#FF6B6B) for negative curvature (cylindrical)
- Cyan (#4ECDC4) for positive curvature (protrusions)
- Dark background with clean styling
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
from pathlib import Path
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from MeshAnalyzer import MeshAnalyzer

# Set consistent presentation style
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
    'savefig.dpi': 500,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'text.usetex': False,
    'mathtext.default': 'regular'
})

# Use RdBu colormap to match your other visualizations
from matplotlib.cm import RdBu

def analyze_neurite_region_presentation(mesh_analyzer, region_bounds, region_name=""):
    """
    Analyze a specific region of the neurite with presentation styling
    
    Parameters:
    -----------
    mesh_analyzer : MeshAnalyzer
        Loaded mesh analyzer object
    region_bounds : dict
        Dictionary with 'x', 'y', 'z' keys containing (min, max) tuples
    region_name : str
        Name of the region for labeling
    """
    
    # Get mesh data
    vertices = mesh_analyzer.vertices
    faces = mesh_analyzer.faces
    curvature = mesh_analyzer.curvature  # Raw curvature in 1/pixel
    
    # Convert to physical units
    pixel_size_xy = mesh_analyzer.pixel_size_xy
    pixel_size_z = mesh_analyzer.pixel_size_z
    
    vertices_um = vertices.copy()
    vertices_um[:, 0] *= pixel_size_xy
    vertices_um[:, 1] *= pixel_size_xy
    vertices_um[:, 2] *= pixel_size_z
    
    curvature_um = curvature / pixel_size_xy  # Convert to 1/μm
    
    # Calculate face centers
    face_centers = np.mean(vertices_um[faces], axis=1)
    
    # Print face center bounds for debugging
    print(f"  Face center bounds:")
    print(f"    X: [{face_centers[:, 0].min():.1f}, {face_centers[:, 0].max():.1f}]")
    print(f"    Y: [{face_centers[:, 1].min():.1f}, {face_centers[:, 1].max():.1f}]")
    print(f"    Z: [{face_centers[:, 2].min():.1f}, {face_centers[:, 2].max():.1f}]")
    
    # Create mask for region
    mask = np.ones(len(face_centers), dtype=bool)
    
    if 'x' in region_bounds:
        mask &= (face_centers[:, 0] >= region_bounds['x'][0]) & (face_centers[:, 0] <= region_bounds['x'][1])
    if 'y' in region_bounds:
        mask &= (face_centers[:, 1] >= region_bounds['y'][0]) & (face_centers[:, 1] <= region_bounds['y'][1])
    if 'z' in region_bounds:
        mask &= (face_centers[:, 2] >= region_bounds['z'][0]) & (face_centers[:, 2] <= region_bounds['z'][1])
    
    # Extract region data
    region_centers = face_centers[mask]
    region_curvature = curvature_um[mask]
    
    print(f"  Found {len(region_curvature)} points in region")
    
    # Check if region is empty
    if len(region_curvature) == 0:
        print(f"  WARNING: No data points found in region!")
        return None, None
    
    # Determine color limits
    vmax = np.percentile(np.abs(region_curvature), 95)
    vmin = -vmax
    
    # Create figure with presentation style
    fig = plt.figure(figsize=(16, 10), facecolor='black')
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.2, 1], width_ratios=[1, 1, 0.8], 
                  hspace=0.3, wspace=0.3)
    
    # 1. 3D region visualization (top left) - show selected region in 3D space
    ax_3d = fig.add_subplot(gs[0, 0], projection='3d', facecolor='black')
    
    # Sample vertices for 3D visualization (subsample for performance)
    sample_indices = np.random.choice(len(region_centers), min(5000, len(region_centers)), replace=False)
    sampled_centers = region_centers[sample_indices]
    sampled_curvature = region_curvature[sample_indices]
    
    scatter_3d = ax_3d.scatter(sampled_centers[:, 0], sampled_centers[:, 1], sampled_centers[:, 2],
                               c=sampled_curvature, s=2, cmap='RdBu', 
                               vmin=vmin, vmax=vmax, alpha=0.6)
    
    ax_3d.set_xlabel('X (μm)', fontsize=12, color='white')
    ax_3d.set_ylabel('Y (μm)', fontsize=12, color='white')
    ax_3d.set_zlabel('Z (μm)', fontsize=12, color='white')
    ax_3d.set_title('3D Region View', fontsize=18, color='white', fontweight='bold')
    ax_3d.view_init(elev=20, azim=45)
    ax_3d.grid(False)
    ax_3d.xaxis.pane.fill = False
    ax_3d.yaxis.pane.fill = False
    ax_3d.zaxis.pane.fill = False
    
    # 2. XY projection (top middle)
    ax_xy = fig.add_subplot(gs[0, 1], facecolor='black')
    
    # Plot all data in gray
    ax_xy.scatter(face_centers[:, 0], face_centers[:, 1], 
                  c='#333333', s=0.5, alpha=0.3)
    
    # Plot selected region with RdBu colors
    scatter_xy = ax_xy.scatter(region_centers[:, 0], region_centers[:, 1],
                               c=region_curvature, s=8, cmap='RdBu',
                               vmin=vmin, vmax=vmax, alpha=0.9, edgecolors='none')
    
    # Add selection rectangle
    if 'x' in region_bounds and 'y' in region_bounds:
        from matplotlib.patches import Rectangle
        rect = Rectangle((region_bounds['x'][0], region_bounds['y'][0]),
                        region_bounds['x'][1] - region_bounds['x'][0],
                        region_bounds['y'][1] - region_bounds['y'][0],
                        fill=False, edgecolor='yellow', linewidth=2)
        ax_xy.add_patch(rect)
    
    ax_xy.set_xlabel('X (μm)', fontsize=16, color='white', fontweight='bold')
    ax_xy.set_ylabel('Y (μm)', fontsize=16, color='white', fontweight='bold')
    ax_xy.set_title(f'XY Projection - {region_name}', fontsize=18, color='white', fontweight='bold')
    ax_xy.set_aspect('equal')
    ax_xy.tick_params(colors='white')
    for spine in ax_xy.spines.values():
        spine.set_color('white')
    
    # 3. Curvature histogram (top right)
    ax_hist = fig.add_subplot(gs[0, 2], facecolor='black')
    
    # Create histogram with presentation colors
    n, bins, patches = ax_hist.hist(region_curvature, bins=40, density=True,
                                    alpha=0.8, edgecolor='white')
    
    # Color bars like in your histogram
    for i, patch in enumerate(patches):
        if bins[i] < 0:
            patch.set_facecolor('#FF6B6B')  # Red for negative
        else:
            patch.set_facecolor('#4ECDC4')  # Cyan for positive
    
    ax_hist.axvline(0, color='white', linestyle='--', linewidth=2, alpha=0.8)
    ax_hist.axvline(np.median(region_curvature), color='orange', linestyle=':', linewidth=2,
                    label=f'Median: {np.median(region_curvature):.2f} μm⁻¹')
    
    ax_hist.set_xlabel('Curvature (μm⁻¹)', fontsize=16, color='white')
    ax_hist.set_ylabel('Density', fontsize=16, color='white')
    ax_hist.set_title('Distribution', fontsize=18, color='white', fontweight='bold')
    ax_hist.tick_params(colors='white')
    ax_hist.legend(loc='upper right', frameon=True, labelcolor='white',
                   facecolor='black', edgecolor='white', framealpha=0.8)
    for spine in ax_hist.spines.values():
        spine.set_color('white')
    
    # 4. XZ projection (bottom left)
    ax_xz = fig.add_subplot(gs[1, 0], facecolor='black')
    scatter_xz = ax_xz.scatter(region_centers[:, 0], region_centers[:, 2],
                               c=region_curvature, s=8, cmap='RdBu',
                               vmin=vmin, vmax=vmax, alpha=0.9, edgecolors='none')
    ax_xz.set_xlabel('X (μm)', fontsize=16, color='white', fontweight='bold')
    ax_xz.set_ylabel('Z (μm)', fontsize=16, color='white', fontweight='bold')
    ax_xz.set_title('XZ Cross-Section', fontsize=18, color='white', fontweight='bold')
    ax_xz.set_aspect('equal')
    ax_xz.tick_params(colors='white')
    for spine in ax_xz.spines.values():
        spine.set_color('white')
    
    # 5. YZ projection (bottom middle)
    ax_yz = fig.add_subplot(gs[1, 1], facecolor='black')
    scatter_yz = ax_yz.scatter(region_centers[:, 1], region_centers[:, 2],
                               c=region_curvature, s=8, cmap='RdBu',
                               vmin=vmin, vmax=vmax, alpha=0.9, edgecolors='none')
    ax_yz.set_xlabel('Y (μm)', fontsize=16, color='white', fontweight='bold')
    ax_yz.set_ylabel('Z (μm)', fontsize=16, color='white', fontweight='bold')
    ax_yz.set_title('YZ Cross-Section', fontsize=18, color='white', fontweight='bold')
    ax_yz.set_aspect('equal')
    ax_yz.tick_params(colors='white')
    for spine in ax_yz.spines.values():
        spine.set_color('white')
    
    # 6. Statistics box (bottom right)
    ax_stats = fig.add_subplot(gs[1, 2], facecolor='black')
    ax_stats.axis('off')
    
    # Calculate statistics
    mean_curv = np.mean(region_curvature)
    median_curv = np.median(region_curvature)
    std_curv = np.std(region_curvature)
    pct_negative = np.sum(region_curvature < 0) / len(region_curvature) * 100
    pct_positive = np.sum(region_curvature > 0) / len(region_curvature) * 100
    
    stats_text = f"""Region Statistics
    
Points analyzed: {len(region_curvature):,}
Mean: {mean_curv:.3f} μm⁻¹
Median: {median_curv:.3f} μm⁻¹
Std Dev: {std_curv:.3f} μm⁻¹

Cylindrical: {pct_negative:.1f}%
Protrusions: {pct_positive:.1f}%"""
    
    ax_stats.text(0.1, 0.5, stats_text, transform=ax_stats.transAxes,
                  va='center', ha='left', fontsize=14, color='white',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='#1a1a1a',
                           edgecolor='white', alpha=0.8))
    
    # Add colorbar to the figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(scatter_xy, cax=cbar_ax)
    cbar.set_label('Mean Curvature (μm⁻¹)', fontsize=14, color='white', labelpad=15)
    cbar.ax.tick_params(colors='white', labelsize=12)
    
    return fig, {
        'mean': mean_curv,
        'median': median_curv,
        'std': std_curv,
        'n_points': len(region_curvature),
        'pct_negative': pct_negative,
        'pct_positive': pct_positive
    }


def main():
    """Generate presentation-style spatial analysis"""
    
    print("="*60)
    print("NEURITE SPATIAL ANALYSIS - PRESENTATION STYLE")
    print("="*60)
    
    # Setup paths
    root = Path("/Users/philippkaintoch/Documents/Projects/01_Bleb3D/Datensatz/3D_Lightsheet/1_shortMovie_BAIAP2_OE/Composite/Output_threelevel")
    mesh_dir = Path("Morphology/Analysis/Mesh/ch1")
    output_dir = Path("/Users/philippkaintoch/Documents/Projects/01_Bleb3D/DanuserLab/Pipeline/Experimental/spatial_analysis_output")
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    surface_path = root / mesh_dir / "surface_1_1.mat"
    curvature_path = root / mesh_dir / "meanCurvature_1_1.mat"
    
    print(f"Loading mesh data...")
    analyzer = MeshAnalyzer(surface_path, curvature_path)
    analyzer.load_data(verbose=False)
    
    # Get mesh bounds for region selection
    vertices_um = analyzer.vertices.copy()
    vertices_um[:, 0] *= analyzer.pixel_size_xy
    vertices_um[:, 1] *= analyzer.pixel_size_xy
    vertices_um[:, 2] *= analyzer.pixel_size_z
    
    print(f"Mesh bounds:")
    print(f"  X: [{vertices_um[:, 0].min():.1f}, {vertices_um[:, 0].max():.1f}] μm")
    print(f"  Y: [{vertices_um[:, 1].min():.1f}, {vertices_um[:, 1].max():.1f}] μm")
    print(f"  Z: [{vertices_um[:, 2].min():.1f}, {vertices_um[:, 2].max():.1f}] μm")
    
    # Define regions to analyze based on actual mesh bounds
    # X: [0.3, 24.7], Y: [13.6, 70.8], Z: [8.9, 29.1]
    regions = {
        'soma_region': {
            'bounds': {'x': (8, 18), 'y': (30, 45), 'z': (12, 22)},
            'name': 'Soma Region'
        },
        'proximal_neurite': {
            'bounds': {'x': (5, 15), 'y': (45, 55), 'z': (10, 20)},
            'name': 'Proximal Neurite'
        },
        'distal_neurite': {
            'bounds': {'x': (10, 20), 'y': (55, 65), 'z': (15, 25)},
            'name': 'Distal Neurite'
        }
    }
    
    results = {}
    
    # Analyze each region
    for region_key, region_info in regions.items():
        print(f"\nAnalyzing {region_info['name']}...")
        
        fig, stats = analyze_neurite_region_presentation(
            analyzer,
            region_info['bounds'],
            region_info['name']
        )
        
        if fig is None:
            print(f"  Skipping {region_info['name']} - no data found")
            continue
        
        # Add main title
        fig.suptitle(f"Spatial Curvature Analysis: {region_info['name']}", 
                     fontsize=20, color='white', y=0.98)
        
        # Save figure
        output_path = output_dir / f'{region_key}_analysis.png'
        fig.savefig(output_path, dpi=500, facecolor='black')
        plt.close(fig)
        
        results[region_key] = stats
        
        print(f"  Mean curvature: {stats['mean']:.3f} μm⁻¹")
        print(f"  Points analyzed: {stats['n_points']:,}")
        print(f"  Saved to: {output_path}")
    
    # Create summary comparison if we have results
    if results:
        print("\nCreating summary comparison...")
        create_summary_figure(results, output_dir)
    else:
        print("\nNo valid regions found - skipping summary")
    
    print(f"\n✅ Analysis complete! All figures saved to: {output_dir}")
    
    return results


def create_summary_figure(results, output_dir):
    """Create summary comparison of all regions"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), facecolor='black')
    fig.suptitle('Regional Curvature Comparison', fontsize=20, color='white', y=0.98)
    
    regions = list(results.keys())
    # Map region keys to display names
    region_name_map = {
        'soma_region': 'Soma',
        'proximal_neurite': 'Proximal',
        'distal_neurite': 'Distal'
    }
    region_names = [region_name_map.get(r, r) for r in regions]
    
    # Extract data
    means = [results[r]['mean'] for r in regions]
    stds = [results[r]['std'] for r in regions]
    pct_neg = [results[r]['pct_negative'] for r in regions]
    
    # 1. Mean curvature
    ax1 = axes[0]
    bars1 = ax1.bar(region_names, means, color=['#FF6B6B' if m < 0 else '#4ECDC4' for m in means],
                     alpha=0.8, edgecolor='white', linewidth=2)
    ax1.axhline(0, color='white', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Mean Curvature (μm⁻¹)', fontsize=16, color='white')
    ax1.set_title('Average Curvature by Region', fontsize=18, color='white')
    ax1.tick_params(colors='white')
    
    # Add value labels
    for bar, val in zip(bars1, means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01 * np.sign(height),
                f'{val:.3f}', ha='center', va='bottom' if val > 0 else 'top', 
                color='white', fontweight='bold')
    
    # 2. Standard deviation
    ax2 = axes[1]
    bars2 = ax2.bar(region_names, stds, color='#888888', alpha=0.8, 
                     edgecolor='white', linewidth=2)
    ax2.set_ylabel('Std Dev (μm⁻¹)', fontsize=16, color='white')
    ax2.set_title('Curvature Variability', fontsize=18, color='white')
    ax2.tick_params(colors='white')
    
    # Add value labels
    for bar, val in zip(bars2, stds):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.3f}', ha='center', va='bottom', color='white', fontweight='bold')
    
    # 3. Percentage cylindrical
    ax3 = axes[2]
    bars3 = ax3.bar(region_names, pct_neg, color='#FF6B6B', alpha=0.8,
                     edgecolor='white', linewidth=2)
    ax3.set_ylabel('Percentage (%)', fontsize=16, color='white')
    ax3.set_title('Cylindrical Morphology', fontsize=18, color='white')
    ax3.set_ylim(0, 100)
    ax3.tick_params(colors='white')
    
    # Add value labels
    for bar, val in zip(bars3, pct_neg):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.1f}%', ha='center', va='bottom', color='white', fontweight='bold')
    
    # Style all axes
    for ax in axes:
        ax.set_facecolor('black')
        for spine in ax.spines.values():
            spine.set_color('white')
            spine.set_linewidth(1.5)
        ax.grid(True, alpha=0.2, color='white', axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'regional_comparison_summary.png', dpi=500, facecolor='black')
    plt.close()


if __name__ == "__main__":
    results = main()