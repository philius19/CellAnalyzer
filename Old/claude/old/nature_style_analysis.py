"""
Nature-Style Curvature Analysis using MeshAnalyzer Object
=========================================================

This script creates publication-ready visualizations using the existing
MeshAnalyzer object architecture. Based on visualization styles from:
"Blebs promote cell survival by assembling oncogenic signalling hubs"
Weems et al., Nature 2023

Author: Analysis Pipeline
Date: 2025-06-09
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy import stats as scipy_stats
from scipy.spatial import distance
import warnings
warnings.filterwarnings('ignore')

# Import the existing MeshAnalyzer object
import sys
sys.path.append('/Users/philippkaintoch/Documents/Projects/01_Bleb3D/DanuserLab/Pipeline')
from mesh_analysis import MeshAnalyzer
from mesh_analysis.visualization import plot_curvature_distribution
from mesh_analysis.utils import calculate_surface_roughness

# Set Nature-style plotting parameters
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'lines.linewidth': 1.0,
    'figure.dpi': 300
})

# Nature journal color palette
NATURE_COLORS = {
    'blue': '#2E86AB',
    'red': '#E63946', 
    'gray': '#6C757D',
    'light_gray': '#ADB5BD',
    'green': '#028A0F',
    'orange': '#F77F00'
}

def load_data_with_mesh_analyzer():
    """
    Load 2D and 3D data using the existing MeshAnalyzer object.
    
    Returns:
        tuple: (analyzer_2d, analyzer_3d) with loaded data
    """
    print("Loading data using MeshAnalyzer objects...")
    
    # Data paths (same as in test.py)
    surface_path_2d = "/Users/philippkaintoch/Documents/Projects/01_Bleb3D/Datensatz/02_2DPreprocessed/Results/Morphology/Analysis/Mesh/ch1/surface_1_1.mat"
    surface_path_3d = "/Users/philippkaintoch/Documents/Projects/01_Bleb3D/Datensatz/3DPreprocessed/Results/Morphology/Analysis/Mesh/ch1/surface_1_1.mat"
    curvature_path_2d = "/Users/philippkaintoch/Documents/Projects/01_Bleb3D/Datensatz/02_2DPreprocessed/Results/Morphology/Analysis/Mesh/ch1/meanCurvature_1_1.mat"
    curvature_path_3d = "/Users/philippkaintoch/Documents/Projects/01_Bleb3D/Datensatz/3DPreprocessed/Results/Morphology/Analysis/Mesh/ch1/meanCurvature_1_1.mat"
    
    # Create MeshAnalyzer objects
    analyzer_2d = MeshAnalyzer(surface_path_2d, curvature_path_2d)
    analyzer_3d = MeshAnalyzer(surface_path_3d, curvature_path_3d)
    
    # Load data
    analyzer_2d.load_data()
    analyzer_3d.load_data()
    
    return analyzer_2d, analyzer_3d

def calculate_comprehensive_statistics(analyzer_2d: MeshAnalyzer, analyzer_3d: MeshAnalyzer):
    """
    Calculate comprehensive statistics using MeshAnalyzer methods.
    
    Args:
        analyzer_2d: 2D cultivation analyzer
        analyzer_3d: 3D cultivation analyzer
        
    Returns:
        dict: Comprehensive statistics for both conditions
    """
    print("Calculating comprehensive statistics...")
    
    # Use existing calculate_statistics method
    stats_2d = analyzer_2d.calculate_statistics()
    stats_3d = analyzer_3d.calculate_statistics()
    
    # Extract curvature data for additional analysis
    curv_2d = remove_outliers(analyzer_2d.curvature)
    curv_3d = remove_outliers(analyzer_3d.curvature)
    
    # Add comparative statistics
    comparative_stats = calculate_comparative_statistics(curv_2d, curv_3d)
    
    # Add surface roughness (using existing utils function)
    stats_2d['curvature']['surface_roughness'] = calculate_surface_roughness(analyzer_2d.curvature)
    stats_3d['curvature']['surface_roughness'] = calculate_surface_roughness(analyzer_3d.curvature)
    
    return {
        '2D': stats_2d,
        '3D': stats_3d,
        'comparison': comparative_stats,
        'raw_data': {
            '2D': curv_2d,
            '3D': curv_3d
        }
    }

def remove_outliers(data: np.ndarray, n_std: float = 3) -> np.ndarray:
    """Remove outliers beyond n standard deviations."""
    mean = np.mean(data)
    std = np.std(data)
    return data[np.abs(data - mean) < n_std * std]

def calculate_comparative_statistics(data_2d: np.ndarray, data_3d: np.ndarray) -> dict:
    """
    Calculate statistical comparisons between 2D and 3D data.
    
    NOTE: Consider adding this function to utils.py for reusability
    """
    # Statistical tests
    t_stat, t_pval = scipy_stats.ttest_ind(data_2d, data_3d)
    mw_stat, mw_pval = scipy_stats.mannwhitneyu(data_2d, data_3d)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(data_2d)**2 + np.std(data_3d)**2) / 2)
    cohens_d = (np.mean(data_2d) - np.mean(data_3d)) / pooled_std
    
    return {
        't_test': {'statistic': t_stat, 'p_value': t_pval},
        'mann_whitney': {'statistic': mw_stat, 'p_value': mw_pval},
        'cohens_d': cohens_d,
        'mean_difference': np.mean(data_2d) - np.mean(data_3d),
        'median_difference': np.median(data_2d) - np.median(data_3d)
    }

def create_nature_main_figure(analyzer_2d: MeshAnalyzer, analyzer_3d: MeshAnalyzer, 
                             stats: dict, output_dir: Path):
    """
    Create main Nature-style figure using MeshAnalyzer data.
    
    NOTE: Consider adding 3D surface rendering to visualization.py
    """
    print("Creating Nature-style main figure...")
    
    fig = plt.figure(figsize=(7.2, 10))
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1, 0.8],
                          hspace=0.4, wspace=0.3)
    
    # Panel A: 3D surface renderings
    ax_a1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax_a2 = fig.add_subplot(gs[0, 1], projection='3d')
    
    plot_3d_surface_with_curvature(ax_a1, analyzer_2d, '2D cultivation')
    plot_3d_surface_with_curvature(ax_a2, analyzer_3d, '3D cultivation')
    
    # Panel B: Statistical comparison
    ax_b = fig.add_subplot(gs[1, :])
    plot_statistical_comparison(ax_b, stats)
    
    # Panel C: Curvature distributions  
    ax_c1 = fig.add_subplot(gs[2, 0])
    ax_c2 = fig.add_subplot(gs[2, 1])
    
    plot_curvature_distribution_nature_style(ax_c1, stats['raw_data']['2D'], '2D')
    plot_curvature_distribution_nature_style(ax_c2, stats['raw_data']['3D'], '3D')
    
    # Panel D: Physical dimensions comparison
    ax_d = fig.add_subplot(gs[3, :])
    plot_physical_dimensions_comparison(ax_d, analyzer_2d, analyzer_3d)
    
    # Add panel labels
    panels = [ax_a1, ax_b, ax_c1, ax_d]
    labels = ['a', 'b', 'c', 'd']
    
    for ax, label in zip(panels, labels):
        if hasattr(ax, 'text2D'):  # 3D axes
            ax.text2D(-0.1, 1.1, label, transform=ax.transAxes,
                     fontsize=12, fontweight='bold')
        else:  # 2D axes
            ax.text(-0.1, 1.1, label, transform=ax.transAxes,
                   fontsize=12, fontweight='bold')
    
    plt.savefig(output_dir / 'nature_main_figure.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_3d_surface_with_curvature(ax, analyzer: MeshAnalyzer, title: str):
    """
    Plot 3D surface with curvature coloring.
    
    NOTE: Consider adding this as enhanced function to visualization.py
    """
    mesh = analyzer.mesh
    curvature = analyzer.curvature
    
    # Sample for visualization performance
    n_faces = min(5000, len(mesh.faces))
    idx = np.random.choice(len(mesh.faces), n_faces, replace=False)
    
    triangles = mesh.vertices[mesh.faces[idx]]
    curv_sample = curvature[idx]
    
    # Nature-style colormap
    vmin, vmax = np.percentile(curv_sample, [10, 90])
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    colors = ['#2E86AB', '#87CEEB', '#FFFFFF', '#FFB6C1', '#E63946']
    cmap = LinearSegmentedColormap.from_list('nature', colors, N=256)
    face_colors = cmap(norm(curv_sample))
    
    # Plot
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    poly3d = Poly3DCollection(triangles, facecolors=face_colors,
                             edgecolors='none', alpha=0.8)
    ax.add_collection3d(poly3d)
    
    # Set limits and styling
    ax.set_xlim(mesh.bounds[0][0], mesh.bounds[1][0])
    ax.set_ylim(mesh.bounds[0][1], mesh.bounds[1][1])
    ax.set_zlim(mesh.bounds[0][2], mesh.bounds[1][2])
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_title(title, fontsize=10, pad=10)

def plot_statistical_comparison(ax, stats: dict):
    """
    Plot statistical comparison with Nature styling.
    
    NOTE: Consider adding this function to visualization.py
    """
    # Extract statistics
    mean_2d = stats['2D']['curvature']['mean']
    mean_3d = stats['3D']['curvature']['mean']
    sem_2d = stats['2D']['curvature']['sem']
    sem_3d = stats['3D']['curvature']['sem']
    
    means = [mean_2d, mean_3d]
    sems = [sem_2d, sem_3d]
    labels = ['2D cultivation', '3D cultivation']
    colors = [NATURE_COLORS['red'], NATURE_COLORS['blue']]
    
    # Bar plot
    bars = ax.bar(labels, means, yerr=sems, capsize=3,
                 color=colors, alpha=0.8, linewidth=0.5, edgecolor='black')
    
    # Styling
    ax.set_ylabel('Mean curvature (1/pixels)', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    # Add significance
    comp = stats['comparison']
    p_val = comp['mann_whitney']['p_value']
    
    if p_val < 0.001:
        sig_text = '***'
    elif p_val < 0.01:
        sig_text = '**'
    elif p_val < 0.05:
        sig_text = '*'
    else:
        sig_text = 'ns'
    
    y_max = max(means) + max(sems) + 0.01
    ax.plot([0, 1], [y_max, y_max], 'k-', linewidth=0.5)
    ax.text(0.5, y_max + 0.005, sig_text, ha='center', va='bottom', fontsize=8)
    
    # Add effect size
    ax.text(0.02, 0.98, f"Cohen's d = {comp['cohens_d']:.3f}",
            transform=ax.transAxes, va='top', fontsize=7)

def plot_curvature_distribution_nature_style(ax, curvature_data: np.ndarray, label: str):
    """Enhanced curvature distribution plot with Nature styling."""
    color = NATURE_COLORS['red'] if label == '2D' else NATURE_COLORS['blue']
    
    # Histogram
    ax.hist(curvature_data, bins=50, density=True, alpha=0.7, color=color,
            edgecolor='white', linewidth=0.5)
    
    # Add statistics
    mean_val = np.mean(curvature_data)
    ax.axvline(mean_val, color='black', linestyle='--', linewidth=1)
    
    # Styling
    ax.set_xlabel('Mean curvature (1/pixels)', fontsize=8)
    ax.set_ylabel('Density', fontsize=8)
    ax.set_title(f'{label} cultivation', fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=7)
    
    # Statistics text
    std_val = np.std(curvature_data)
    ax.text(0.7, 0.9, f'μ = {mean_val:.3f}\\nσ = {std_val:.3f}',
            transform=ax.transAxes, va='top', fontsize=7,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

def plot_physical_dimensions_comparison(ax, analyzer_2d: MeshAnalyzer, analyzer_3d: MeshAnalyzer):
    """
    Plot physical dimensions comparison using analyzer properties.
    
    NOTE: This uses the existing physical_dimensions property
    """
    # Get physical dimensions
    dims_2d = analyzer_2d.physical_dimensions
    dims_3d = analyzer_3d.physical_dimensions
    
    # Extract dimensions
    metrics = ['x_um', 'y_um', 'z_um']
    values_2d = [dims_2d.get(m, 0) for m in metrics]
    values_3d = [dims_3d.get(m, 0) for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax.bar(x - width/2, values_2d, width, label='2D', color=NATURE_COLORS['red'], alpha=0.7)
    ax.bar(x + width/2, values_3d, width, label='3D', color=NATURE_COLORS['blue'], alpha=0.7)
    
    ax.set_xlabel('Dimension', fontsize=8)
    ax.set_ylabel('Size (μm)', fontsize=8)
    ax.set_title('Physical dimensions comparison', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(['X', 'Y', 'Z'])
    ax.legend(fontsize=7, frameon=False)
    ax.tick_params(axis='both', which='major', labelsize=7)

def calculate_spatial_correlation(analyzer: MeshAnalyzer, max_dist: float = 50) -> dict:
    """
    Calculate spatial autocorrelation of curvature.
    
    NOTE: Consider adding this function to utils.py for advanced spatial analysis
    
    Args:
        analyzer: MeshAnalyzer object with loaded data
        max_dist: Maximum distance for correlation analysis
        
    Returns:
        dict: Spatial correlation data
    """
    mesh = analyzer.mesh
    curvature = analyzer.curvature
    face_centers = mesh.triangles_center
    
    # Sample for computational efficiency
    n_sample = min(3000, len(face_centers))
    idx = np.random.choice(len(face_centers), n_sample, replace=False)
    
    pos_sample = face_centers[idx]
    curv_sample = curvature[idx]
    
    # Calculate pairwise distances
    dists = distance.cdist(pos_sample, pos_sample)
    
    # Bin distances
    dist_bins = np.linspace(0, max_dist, 15)
    distances = []
    correlations = []
    
    for i in range(len(dist_bins)-1):
        mask = (dists > dist_bins[i]) & (dists <= dist_bins[i+1])
        np.fill_diagonal(mask, False)
        
        if np.sum(mask) > 50:
            pairs = np.where(mask)
            corr_vals = np.corrcoef(curv_sample[pairs[0]], curv_sample[pairs[1]])[0, 1]
            
            if not np.isnan(corr_vals):
                distances.append((dist_bins[i] + dist_bins[i+1]) / 2)
                correlations.append(corr_vals)
    
    return {'distances': distances, 'correlations': correlations}

def generate_nature_style_report(stats: dict, analyzer_2d: MeshAnalyzer, 
                                analyzer_3d: MeshAnalyzer, output_dir: Path):
    """Generate Nature-style analysis report."""
    print("Generating Nature-style report...")
    
    report = []
    report.append("MEMBRANE CURVATURE ANALYSIS: 2D vs 3D NEURONAL CULTIVATION")
    report.append("=" * 70)
    report.append("")
    
    report.append("METHODS")
    report.append("-" * 20)
    report.append("Surface meshes were analyzed using the MeshAnalyzer object with")
    report.append("integrated u-shape3D pipeline. Statistical comparisons used")
    report.append("non-parametric tests with effect size calculations.")
    report.append("")
    
    report.append("RESULTS")
    report.append("-" * 20)
    
    # Extract key statistics
    stats_2d = stats['2D']['curvature']
    stats_3d = stats['3D']['curvature']
    comp = stats['comparison']
    
    report.append(f"2D cultivation: {stats_2d['mean']:.4f} ± {stats_2d['sem']:.4f} (SEM)")
    report.append(f"3D cultivation: {stats_3d['mean']:.4f} ± {stats_3d['sem']:.4f} (SEM)")
    report.append("")
    
    report.append(f"Statistical significance: p = {comp['mann_whitney']['p_value']:.2e}")
    report.append(f"Effect size (Cohen's d): {comp['cohens_d']:.3f}")
    report.append("")
    
    report.append("MESH PROPERTIES")
    report.append("-" * 20)
    mesh_2d = stats['2D']['mesh']
    mesh_3d = stats['3D']['mesh']
    
    report.append(f"2D mesh: {mesh_2d['n_vertices']} vertices, {mesh_2d['n_faces']} faces")
    report.append(f"3D mesh: {mesh_3d['n_vertices']} vertices, {mesh_3d['n_faces']} faces")
    report.append(f"2D volume: {mesh_2d['volume_um3']:.1f} μm³")
    report.append(f"3D volume: {mesh_3d['volume_um3']:.1f} μm³")
    
    # Save report
    report_text = '\n'.join(report)
    with open(output_dir / 'nature_analysis_report.txt', 'w') as f:
        f.write(report_text)
    
    print(report_text)
    return report_text

def main():
    """
    Main analysis function using existing MeshAnalyzer architecture.
    """
    print("=" * 60)
    print("NATURE-STYLE CURVATURE ANALYSIS")
    print("Using MeshAnalyzer Object Architecture")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("/Users/philippkaintoch/Documents/Projects/01_Bleb3D/DanuserLab/Pipeline/claude")
    output_dir.mkdir(exist_ok=True)
    
    # Load data using existing MeshAnalyzer
    analyzer_2d, analyzer_3d = load_data_with_mesh_analyzer()
    
    # Calculate comprehensive statistics
    stats = calculate_comprehensive_statistics(analyzer_2d, analyzer_3d)
    
    # Create Nature-style visualizations
    create_nature_main_figure(analyzer_2d, analyzer_3d, stats, output_dir)
    
    # Generate report
    generate_nature_style_report(stats, analyzer_2d, analyzer_3d, output_dir)
    
    print("\n✓ Nature-style analysis complete!")
    print(f"Results saved to: {output_dir}")
    
    # Summary of findings
    comp = stats['comparison']
    print(f"\nKey Finding: Cohen's d = {comp['cohens_d']:.3f}, p = {comp['mann_whitney']['p_value']:.2e}")

if __name__ == "__main__":
    main()