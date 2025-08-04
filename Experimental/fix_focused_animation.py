#!/usr/bin/env python3
"""
Fix Focused Animation
Fix Y-axis mirroring in the focused animation only
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
import tifffile
from mat73 import loadmat
from skimage import measure

# Minimal dark theme
plt.style.use('dark_background')
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.linewidth': 0,
    'figure.facecolor': '#000000',
    'axes.facecolor': '#000000',
})

def load_data():
    """Load mask and mesh data"""
    image_path = Path("/Users/philippkaintoch/Documents/Projects/01_Bleb3D/DanuserLab/Pipeline/Claude_workspace/final_segmentation_mask.tif")
    mesh_dir = Path("/Users/philippkaintoch/Documents/Projects/01_Bleb3D/Analyze/2D/4_BAIAP2_OE/Morphology/Analysis/Mesh/ch1")
    
    mask_3d = tifffile.imread(image_path)
    
    surface_file = mesh_dir / "surface_1_1.mat"
    surface_data = loadmat(str(surface_file))
    vertices = np.array(surface_data['surface']['vertices'], dtype=np.float32) - 1.0
    
    return mask_3d, vertices

def create_fixed_focused_animation():
    """Create focused animation with proper Y-axis alignment"""
    print("🎬 Creating fixed focused animation...")
    
    mask_3d, vertices = load_data()
    
    # Find best region
    x_center, y_center = 350, 450
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black')
    ax.set_facecolor('black')
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Focus on region
    window = 150
    ax.set_xlim(x_center - window, x_center + window)
    ax.set_ylim(y_center - window, y_center + window)
    
    # Animation parameters
    z_with_vertices = np.unique(np.round(vertices[:, 2]).astype(int))
    z_with_vertices = z_with_vertices[(z_with_vertices >= 0) & (z_with_vertices < mask_3d.shape[0])]
    
    z_start = z_with_vertices.min()
    z_end = z_with_vertices.max()
    z_slices = np.arange(z_start, z_end + 1, 2)
    n_frames = len(z_slices)
    
    # Initialize elements - FIX: Remove extent parameter that was causing Y-flip
    mask_img = ax.imshow(np.zeros((mask_3d.shape[1], mask_3d.shape[2])), 
                        cmap='gray', alpha=0.4, vmin=0, vmax=255)
    
    boundary_line, = ax.plot([], [], 'white', linewidth=1.5, alpha=0.9)
    
    vertex_scatter = ax.scatter([], [], c='#ff4444', s=8, alpha=0.9)
    
    # Minimal slice indicator
    slice_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        fontsize=8, color='white', alpha=0.6)
    
    def init():
        return mask_img, boundary_line, vertex_scatter, slice_text
    
    def animate(frame):
        z_idx = z_slices[frame]
        
        # Update mask
        mask_slice = mask_3d[z_idx]
        mask_img.set_data(mask_slice)
        
        # Boundary
        if mask_slice.max() > 0:
            contours = measure.find_contours(mask_slice, 127)
            if contours:
                all_x, all_y = [], []
                for contour in contours:
                    if len(contour) > 30:
                        all_x.extend(contour[:, 1])
                        all_y.extend(contour[:, 0])
                        all_x.append(np.nan)
                        all_y.append(np.nan)
                boundary_line.set_data(all_x, all_y)
        else:
            boundary_line.set_data([], [])
        
        # Vertices in view
        z_tolerance = 1.0
        near_mask = np.abs(vertices[:, 2] - z_idx) < z_tolerance
        slice_vertices = vertices[near_mask]
        
        # Filter to visible region
        if len(slice_vertices) > 0:
            in_view = ((slice_vertices[:, 0] >= x_center - window) & 
                       (slice_vertices[:, 0] <= x_center + window) &
                       (slice_vertices[:, 1] >= y_center - window) & 
                       (slice_vertices[:, 1] <= y_center + window))
            
            visible_vertices = slice_vertices[in_view]
            
            if len(visible_vertices) > 0:
                vertex_scatter.set_offsets(visible_vertices[:, :2])
                vertex_scatter.set_sizes(np.ones(len(visible_vertices)) * 8)
            else:
                vertex_scatter.set_offsets(np.empty((0, 2)))
        else:
            vertex_scatter.set_offsets(np.empty((0, 2)))
        
        # Update text
        slice_text.set_text(f'z: {z_idx}')
        
        return mask_img, boundary_line, vertex_scatter, slice_text
    
    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                        interval=100, blit=True)
    
    # Save
    output_path = Path("/Users/philippkaintoch/Documents/Projects/01_Bleb3D/DanuserLab/Pipeline/Claude_workspace/presentation_figures/fixed_focused_animation.gif")
    writer = PillowWriter(fps=10)
    anim.save(output_path, writer=writer, dpi=150)
    plt.close(fig)
    
    print(f"✅ Fixed focused animation saved: {output_path}")
    return output_path

def main():
    """Fix the focused animation only"""
    print("🔧 Fixing focused animation Y-axis alignment...")
    print("=" * 50)
    
    output_path = create_fixed_focused_animation()
    
    print(f"\n✅ Fixed animation created: {output_path}")
    print("💡 Removed extent parameter that was causing Y-axis flip")

if __name__ == "__main__":
    main()