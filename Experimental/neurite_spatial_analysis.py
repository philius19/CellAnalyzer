#!/usr/bin/env python3
"""
Neurite Spatial Analysis for Presentation
Using your spatial analysis style to showcase regional curvature patterns
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import trimesh
from pathlib import Path
from matplotlib.gridspec import GridSpec

# Use your dark background style
plt.style.use('dark_background')
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 11,
    'axes.linewidth': 0.5,
    'figure.facecolor': 'black',
    'axes.facecolor': 'black'
})

def load_mesh_data(surface_path, curvature_path):
    """Load mesh and curvature data"""
    with h5py.File(surface_path, 'r') as f:
        vertices = f['surface']['vertices'][:].T
        faces = f['surface']['faces'][:].T - 1
        
    with h5py.File(curvature_path, 'r') as f:
        face_curvature = f['meanCurvature'][0]
        
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh, face_curvature

def analyze_neurite_curvature_presentation(mesh, curvature, title_suffix="", 
                                          y_range=(660, 730), x_range=(230, 250), z_range=None):
    """
    Enhanced version of your neurite analysis for presentation
    """
    face_centers = mesh.triangles_center
    
    # Create spatial filter mask
    mask = (face_centers[:, 1] > y_range[0]) & (face_centers[:, 1] < y_range[1])
    
    if x_range is not None:
        mask &= (face_centers[:, 0] > x_range[0]) & (face_centers[:, 0] < x_range[1])
    
    if z_range is not None:
        mask &= (face_centers[:, 2] > z_range[0]) & (face_centers[:, 2] < z_range[1])
    
    neurite_curvature = curvature[mask]
    neurite_centers = face_centers[mask]
    
    # Create figure with your style
    fig = plt.figure(figsize=(14, 6), facecolor='black')
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.8], wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # 1. Spatial view (XZ projection)
    vmax = np.percentile(np.abs(neurite_curvature), 95)
    vmin = -vmax
    
    scatter = ax1.scatter(neurite_centers[:, 0]/10, neurite_centers[:, 2]/10,
                          c=neurite_curvature*10, s=8, cmap='RdBu_r',
                          vmin=vmin*10, vmax=vmax*10, alpha=0.8)
    ax1.set_xlabel('X position (μm)', color='white')
    ax1.set_ylabel('Z position (μm)', color='white')
    ax1.set_title(f'Neurite Cross-Section {title_suffix}', color='white')
    ax1.set_aspect('equal')
    ax1.tick_params(colors='white')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Mean Curvature (1/μm)', color='white')
    cbar.ax.tick_params(colors='white')
    
    # 2. Full mesh context view (XY projection)
    all_centers = mesh.triangles_center
    ax2.scatter(all_centers[:, 0]/10, all_centers[:, 1]/10, 
               c='gray', s=0.5, alpha=0.3)
    ax2.scatter(neurite_centers[:, 0]/10, neurite_centers[:, 1]/10,
               c=neurite_curvature*10, s=8, cmap='RdBu_r',
               vmin=vmin*10, vmax=vmax*10)
    ax2.set_xlabel('X position (μm)', color='white')
    ax2.set_ylabel('Y position (μm)', color='white')
    ax2.set_title(f'Selection Context {title_suffix}', color='white')
    ax2.set_aspect('equal')
    ax2.tick_params(colors='white')
    
    # Add selection box
    if x_range and y_range:
        from matplotlib.patches import Rectangle
        rect = Rectangle((x_range[0]/10, y_range[0]/10), 
                        (x_range[1]-x_range[0])/10, (y_range[1]-y_range[0])/10,
                        fill=False, edgecolor='yellow', linewidth=2)
        ax2.add_patch(rect)
    
    # 3. Curvature distribution
    ax3.hist(neurite_curvature*10, bins=30, alpha=0.7, edgecolor='white', 
             color='skyblue')
    ax3.axvline(0, color='white', linestyle='--', linewidth=2, alpha=0.8)
    ax3.set_xlabel('Curvature (1/μm)', color='white')
    ax3.set_ylabel('Count', color='white')
    ax3.set_title(f'Distribution {title_suffix}', color='white')
    ax3.tick_params(colors='white')
    
    # Add statistics text
    mean_curv = np.mean(neurite_curvature*10)
    std_curv = np.std(neurite_curvature*10)
    stats_text = f'μ = {mean_curv:.3f}\nσ = {std_curv:.3f}\nn = {len(neurite_curvature)}'
    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, 
             va='top', ha='left', color='white',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    # Style all axes
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('black')
        for spine in ax.spines.values():
            spine.set_color('white')
    
    plt.tight_layout()
    return fig, (mean_curv, std_curv, len(neurite_curvature))

def create_comprehensive_spatial_analysis():
    """Create comprehensive spatial analysis for presentation"""
    
    base_path = Path("/Users/philippkaintoch/Documents/Projects/01_Bleb3D/DanuserLab/Pipeline/Claude_workspace")
    output_path = base_path / "presentation_figures"
    
    print("🔬 Creating comprehensive spatial analysis...")
    
    # Load 2D data
    print("Loading 2D neuron data...")
    try:
        mesh_2d, curvature_2d = load_mesh_data(
            base_path / "4_BAIAP2_OE/Morphology/Analysis/Mesh/ch1/surface_1_1.mat",
            base_path / "4_BAIAP2_OE/Morphology/Analysis/Mesh/ch1/meanCurvature_1_1.mat"
        )
        has_2d = True
    except:
        print("  ⚠️ 2D data not found")
        has_2d = False
    
    # Load 3D data
    print("Loading 3D neuron data...")
    try:
        mesh_3d, curvature_3d = load_mesh_data(
            base_path / "1_shortMovie_BAIAP2_OE_T0_Mesh/Morphology/Analysis/Mesh/ch1/surface_1_1.mat",
            base_path / "1_shortMovie_BAIAP2_OE_T0_Mesh/Morphology/Analysis/Mesh/ch1/meanCurvature_1_1.mat"
        )
        has_3d = True
    except:
        print("  ⚠️ 3D data not found")
        has_3d = False
    
    results = {}
    
    # Analyze different regions for each dataset
    if has_2d:
        print("Analyzing 2D neurite regions...")
        
        # Soma region (center)
        fig_2d_soma, stats_2d_soma = analyze_neurite_curvature_presentation(
            mesh_2d, curvature_2d, 
            title_suffix="(2D Soma)",
            y_range=(400, 500), x_range=(300, 400)
        )
        fig_2d_soma.suptitle('2D Culture: Soma Region Analysis', 
                            fontsize=16, color='#FF6B6B', y=0.98)
        fig_2d_soma.savefig(output_path / '2D_soma_analysis.png', 
                           dpi=300, bbox_inches='tight', facecolor='black')
        plt.close(fig_2d_soma)
        
        # Neurite region
        fig_2d_neurite, stats_2d_neurite = analyze_neurite_curvature_presentation(
            mesh_2d, curvature_2d,
            title_suffix="(2D Neurite)",
            y_range=(200, 300), x_range=(200, 300)
        )
        fig_2d_neurite.suptitle('2D Culture: Neurite Region Analysis',
                               fontsize=16, color='#FF6B6B', y=0.98)
        fig_2d_neurite.savefig(output_path / '2D_neurite_analysis.png',
                              dpi=300, bbox_inches='tight', facecolor='black')
        plt.close(fig_2d_neurite)
        
        results['2D'] = {
            'soma': stats_2d_soma,
            'neurite': stats_2d_neurite
        }
    
    if has_3d:
        print("Analyzing 3D neurite regions...")
        
        # Adjust ranges for 3D data (different coordinate system)
        fig_3d_soma, stats_3d_soma = analyze_neurite_curvature_presentation(
            mesh_3d, curvature_3d,
            title_suffix="(3D Soma)", 
            y_range=(450, 550), x_range=(100, 200)
        )
        fig_3d_soma.suptitle('3D Culture: Soma Region Analysis',
                            fontsize=16, color='#4ECDC4', y=0.98)
        fig_3d_soma.savefig(output_path / '3D_soma_analysis.png',
                           dpi=300, bbox_inches='tight', facecolor='black')
        plt.close(fig_3d_soma)
        
        # 3D neurite
        fig_3d_neurite, stats_3d_neurite = analyze_neurite_curvature_presentation(
            mesh_3d, curvature_3d,
            title_suffix="(3D Neurite)",
            y_range=(600, 700), x_range=(150, 250)
        )
        fig_3d_neurite.suptitle('3D Culture: Neurite Region Analysis',
                               fontsize=16, color='#4ECDC4', y=0.98)
        fig_3d_neurite.savefig(output_path / '3D_neurite_analysis.png',
                              dpi=300, bbox_inches='tight', facecolor='black')
        plt.close(fig_3d_neurite)
        
        results['3D'] = {
            'soma': stats_3d_soma,
            'neurite': stats_3d_neurite
        }
    
    # Create comparative summary
    if has_2d and has_3d:
        create_comparative_summary(results, output_path)
    
    print(f"✅ Spatial analysis complete! Saved to: {output_path}")
    return results

def create_comparative_summary(results, output_path):
    """Create summary comparison figure"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor='black')
    fig.suptitle('Regional Curvature Comparison: 2D vs 3D', 
                 fontsize=18, color='white', y=0.98)
    
    regions = ['soma', 'neurite']
    colors = ['#FF6B6B', '#4ECDC4']  # Red for 2D, Teal for 3D
    
    for i, region in enumerate(regions):
        ax_mean = axes[i, 0]
        ax_std = axes[i, 1]
        
        # Extract stats
        mean_2d = results['2D'][region][0]
        std_2d = results['2D'][region][1]
        mean_3d = results['3D'][region][0]
        std_3d = results['3D'][region][1]
        
        # Mean curvature comparison
        bars_mean = ax_mean.bar(['2D', '3D'], [mean_2d, mean_3d], 
                               color=colors, alpha=0.7)
        ax_mean.set_title(f'{region.title()} - Mean Curvature', color='white')
        ax_mean.set_ylabel('Curvature (1/μm)', color='white')
        ax_mean.tick_params(colors='white')
        
        # Add value labels
        for bar, val in zip(bars_mean, [mean_2d, mean_3d]):
            height = bar.get_height()
            ax_mean.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}', ha='center', va='bottom', color='white')
        
        # Standard deviation comparison
        bars_std = ax_std.bar(['2D', '3D'], [std_2d, std_3d], 
                             color=colors, alpha=0.7)
        ax_std.set_title(f'{region.title()} - Curvature Variability', color='white')
        ax_std.set_ylabel('Standard Deviation', color='white')
        ax_std.tick_params(colors='white')
        
        # Add value labels
        for bar, val in zip(bars_std, [std_2d, std_3d]):
            height = bar.get_height()
            ax_std.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', color='white')
        
        # Style axes
        for ax in [ax_mean, ax_std]:
            ax.set_facecolor('black')
            for spine in ax.spines.values():
                spine.set_color('white')
    
    plt.tight_layout()
    plt.savefig(output_path / 'regional_comparison_summary.png',
               dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()

if __name__ == "__main__":
    results = create_comprehensive_spatial_analysis()
    print("\n📊 Analysis Results Summary:")
    for culture, regions in results.items():
        print(f"\n{culture} Culture:")
        for region, (mean, std, n) in regions.items():
            print(f"  {region.title()}: μ={mean:.3f}, σ={std:.3f}, n={n}")