#!/usr/bin/env python3
"""
Mesh Generation Movie
Creates a cinematic animation showing how vertices trace the neuron boundary
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
from pathlib import Path
import tifffile
from mat73 import loadmat
from skimage import measure
from scipy.ndimage import gaussian_filter

# Cinematic dark theme
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

def create_cinematic_mesh_movie():
    """Create cinematic mesh generation animation"""
    print("🎬 Creating cinematic mesh generation movie...")
    
    mask_3d, vertices = load_data()
    
    # Setup figure with black background
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#000000')
    ax.set_facecolor('#000000')
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Focus on a specific region with interesting structure
    x_center, y_center = 350, 450
    window_size = 150
    
    ax.set_xlim(x_center - window_size, x_center + window_size)
    ax.set_ylim(y_center - window_size, y_center + window_size)
    
    # Animation parameters
    z_start, z_end = 190, 210
    z_slices = np.linspace(z_start, z_end, 40)
    n_frames = len(z_slices)
    
    # Create gradient mask effect
    gradient = np.zeros((mask_3d.shape[1], mask_3d.shape[2], 3))
    
    # Initialize plot elements
    mask_img = ax.imshow(gradient, alpha=1.0, extent=[0, mask_3d.shape[2], 0, mask_3d.shape[1]])
    
    # Boundary line with glow effect
    boundary_lines = []
    
    # Vertex elements
    vertex_dots = []
    vertex_trails = []
    
    # Title with fade effect
    title = ax.text(0.5, 0.95, '', transform=ax.transAxes, ha='center',
                   fontsize=16, color='white', weight='light', alpha=0)
    
    # Progress bar
    progress_bar = ax.add_patch(plt.Rectangle((0.1, 0.02), 0, 0.01, 
                                            transform=ax.transAxes,
                                            facecolor='cyan', alpha=0.5))
    
    def init():
        return [mask_img, title, progress_bar]
    
    def animate(frame):
        # Clear previous elements
        for line in boundary_lines:
            line.remove()
        boundary_lines.clear()
        
        for dot in vertex_dots:
            dot.remove()
        vertex_dots.clear()
        
        # Calculate Z position with easing
        t = frame / (n_frames - 1)
        z_idx = int(z_start + (z_end - z_start) * (3*t**2 - 2*t**3))  # Smooth interpolation
        
        # Get mask slice
        mask_slice = mask_3d[z_idx].astype(float)
        
        # Create gradient effect
        gradient_slice = np.zeros((mask_slice.shape[0], mask_slice.shape[1], 3))
        
        # Blue channel for mask
        gradient_slice[:, :, 2] = gaussian_filter(mask_slice / 255.0, sigma=2) * 0.3
        
        # Find and draw boundary
        if mask_slice.max() > 0:
            contours = measure.find_contours(mask_slice, 127)
            if contours:
                for contour in contours:
                    if len(contour) > 50:  # Only significant contours
                        # Main boundary line
                        line = ax.plot(contour[:, 1], contour[:, 0], 
                                     color='white', linewidth=2, alpha=0.8)[0]
                        boundary_lines.append(line)
                        
                        # Glow effect
                        glow = ax.plot(contour[:, 1], contour[:, 0], 
                                     color='cyan', linewidth=4, alpha=0.3)[0]
                        boundary_lines.append(glow)
        
        # Find vertices near this slice
        z_tolerance = 1.0
        near_mask = np.abs(vertices[:, 2] - z_idx) < z_tolerance
        slice_vertices = vertices[near_mask]
        
        # Draw vertices with animation
        if len(slice_vertices) > 0:
            # Limit vertices shown for performance
            if len(slice_vertices) > 200:
                indices = np.random.choice(len(slice_vertices), 200, replace=False)
                slice_vertices = slice_vertices[indices]
            
            for i, vertex in enumerate(slice_vertices):
                # Animate appearance
                alpha = min(1.0, (frame - 10) / 20) if frame > 10 else 0
                
                # Pulsing effect
                size = 30 + 10 * np.sin(frame * 0.1 + i * 0.1)
                
                # Main vertex dot
                dot = ax.scatter(vertex[0], vertex[1], c='red', s=size, 
                               alpha=alpha * 0.9, edgecolors='yellow', linewidth=1)
                vertex_dots.append(dot)
                
                # Add glow
                glow = ax.scatter(vertex[0], vertex[1], c='yellow', s=size*2, 
                                alpha=alpha * 0.2)
                vertex_dots.append(glow)
        
        # Update gradient background
        gradient_slice[:, :, 0] = gaussian_filter(mask_slice / 255.0, sigma=5) * 0.1  # Red channel
        mask_img.set_data(gradient_slice)
        
        # Update title with fade
        title_alpha = min(1.0, frame / 10)
        title.set_alpha(title_alpha)
        title.set_text(f'Mesh Generation • Z = {z_idx} • {len(slice_vertices)} vertices')
        
        # Update progress bar
        progress = (frame + 1) / n_frames
        progress_bar.set_width(0.8 * progress)
        
        return [mask_img, title, progress_bar] + boundary_lines + vertex_dots
    
    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                        interval=100, blit=True)
    
    # Save with higher quality
    output_path = Path("/Users/philippkaintoch/Documents/Projects/01_Bleb3D/DanuserLab/Pipeline/Claude_workspace/presentation_figures/mesh_generation_movie.gif")
    writer = PillowWriter(fps=10)
    anim.save(output_path, writer=writer, dpi=200)
    plt.close(fig)
    
    print(f"✅ Cinematic movie saved: {output_path}")
    return output_path

def create_vertex_flow_animation():
    """Create animation showing vertex flow along the surface"""
    print("🎬 Creating vertex flow animation...")
    
    _, vertices = load_data()
    
    # Create figure
    fig = plt.figure(figsize=(10, 10), facecolor='#000000')
    ax = fig.add_subplot(111, projection='3d', facecolor='#000000')
    
    # Clean 3D view
    ax.set_axis_off()
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    # Subsample vertices
    step = max(1, len(vertices) // 20000)
    vertices_sub = vertices[::step]
    
    # Set view limits
    x_center = (vertices_sub[:, 0].min() + vertices_sub[:, 0].max()) / 2
    y_center = (vertices_sub[:, 1].min() + vertices_sub[:, 1].max()) / 2
    z_center = (vertices_sub[:, 2].min() + vertices_sub[:, 2].max()) / 2
    
    max_range = max(vertices_sub[:, 0].max() - vertices_sub[:, 0].min(),
                   vertices_sub[:, 1].max() - vertices_sub[:, 1].min(),
                   vertices_sub[:, 2].max() - vertices_sub[:, 2].min()) / 2
    
    ax.set_xlim(x_center - max_range, x_center + max_range)
    ax.set_ylim(y_center - max_range, y_center + max_range)
    ax.set_zlim(z_center - max_range, z_center + max_range)
    
    # Color by height with gradient
    z_norm = (vertices_sub[:, 2] - vertices_sub[:, 2].min()) / (vertices_sub[:, 2].max() - vertices_sub[:, 2].min())
    
    # Title
    title = fig.text(0.5, 0.95, 'Vertex Distribution Flow', ha='center',
                    fontsize=18, color='white', weight='light')
    
    def animate(frame):
        ax.clear()
        ax.set_axis_off()
        
        # Rotate view
        ax.view_init(elev=20 + 5*np.sin(frame*0.05), azim=frame)
        
        # Create flowing effect by varying alpha based on Z and time
        flow_alpha = 0.3 + 0.5 * np.sin(z_norm * np.pi + frame * 0.1)
        flow_alpha = np.clip(flow_alpha, 0.1, 0.9)  # Ensure alpha is in valid range
        
        # Vary size based on position and time
        sizes = 0.5 + 1.5 * np.sin(z_norm * np.pi + frame * 0.05)**2
        
        # Color gradient
        colors = plt.cm.plasma(z_norm + 0.1 * np.sin(frame * 0.05))
        
        # Plot with flow effect
        ax.scatter(vertices_sub[:, 0], vertices_sub[:, 1], vertices_sub[:, 2],
                  c=colors, s=sizes, alpha=flow_alpha)
        
        # Reset limits
        ax.set_xlim(x_center - max_range, x_center + max_range)
        ax.set_ylim(y_center - max_range, y_center + max_range)
        ax.set_zlim(z_center - max_range, z_center + max_range)
        
        return [title]
    
    # Create animation
    anim = FuncAnimation(fig, animate, frames=120, interval=50, blit=False)
    
    # Save
    output_path = Path("/Users/philippkaintoch/Documents/Projects/01_Bleb3D/DanuserLab/Pipeline/Claude_workspace/presentation_figures/vertex_flow.gif")
    writer = PillowWriter(fps=20)
    anim.save(output_path, writer=writer, dpi=150)
    plt.close(fig)
    
    print(f"✅ Vertex flow saved: {output_path}")
    return output_path

def main():
    """Create cinematic animations"""
    print("🎬 Creating cinematic mesh animations...")
    print("=" * 50)
    
    # Create animations
    movie_path = create_cinematic_mesh_movie()
    flow_path = create_vertex_flow_animation()
    
    print("\n✅ Cinematic animations complete!")
    print(f"\nOutputs:")
    print(f"  • Mesh generation movie: {movie_path}")
    print(f"  • Vertex flow animation: {flow_path}")

if __name__ == "__main__":
    main()