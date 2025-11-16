"""
Publication Figure Generator for Vessel Meshes
Creates consistent 2D views of all meshes for publication
Oriented with large component facing downward
Author: Assistant
Date: 2025-08-31
"""

import sys
import os
import numpy as np
from pathlib import Path
from vedo import Plotter
import matplotlib.pyplot as plt

# Add parent directory to path for MeshAnalyzer import
sys.path.append(str(Path(__file__).parent.parent))
from MeshAnalyzer import MeshAnalyzer


class PublicationFigureGenerator:
    """Generate publication-quality 2D figure views of meshes."""
    
    def __init__(self, surface_path, curvature_path):
        """
        Initialize the figure generator.
        
        Args:
            surface_path: Path to surface .mat file
            curvature_path: Path to curvature .mat file
        """
        self.surface_path = Path(surface_path)
        self.curvature_path = Path(curvature_path)
        
        # Load mesh data
        self.analyzer = MeshAnalyzer(str(self.surface_path), str(self.curvature_path))
        self.analyzer.load_data()
    
    def create_publication_view(self, output_path, 
                               view_angle='front',
                               zoom_factor=0.75,
                               image_size=(2400, 3200),
                               dpi=300):
        """
        Create a high-quality 2D publication figure of the mesh.
        
        Args:
            output_path: Path for output image
            view_angle: 'front' or 'side' view
            zoom_factor: Camera zoom level
            image_size: Image dimensions (width, height)
            dpi: Resolution for publication
        """
        # Clone the mesh
        mesh = self.analyzer.mesh.clone()
        
        # Create styled mesh components
        # Semi-transparent solid mesh
        solid_mesh = mesh.clone()
        solid_mesh.color('lightgrey')
        solid_mesh.alpha(0.3)  # Semi-transparent
        solid_mesh.lighting('default')
        
        # Wireframe overlay
        wireframe = mesh.clone()
        wireframe = wireframe.wireframe()
        wireframe.color('darkgrey')
        wireframe.linewidth(0.8)
        wireframe.alpha(0.8)
        
        # Rotate mesh to orient with large part down (180° around Y)
        solid_mesh.rotate_y(180)
        wireframe.rotate_y(180)
        
        # Setup plotter for high-quality rendering
        plotter = Plotter(
            bg='white',  # White background for publication
            size=image_size,
            offscreen=True
        )
        
        # Add meshes without axes or text
        plotter.show(
            solid_mesh, wireframe,
            axes=0,  # No axes for publication figure
            viewup='y',
            resetcam=True,
            interactive=False
        )
        
        # Center the mesh properly
        bounds = solid_mesh.bounds()
        center = [(bounds[0] + bounds[1])/2, 
                  (bounds[2] + bounds[3])/2,
                  (bounds[4] + bounds[5])/2]
        
        # Set camera position based on view angle
        if view_angle == 'front':
            # Front view - looking along Z axis
            plotter.camera.SetPosition(center[0], center[1], center[2] + 1500)
            plotter.camera.SetFocalPoint(center[0], center[1], center[2])
            plotter.camera.SetViewUp(0, 1, 0)
        elif view_angle == 'side':
            # Side view - looking along X axis
            plotter.camera.SetPosition(center[0] + 1500, center[1], center[2])
            plotter.camera.SetFocalPoint(center[0], center[1], center[2])
            plotter.camera.SetViewUp(0, 1, 0)
        
        # Apply zoom
        plotter.camera.Zoom(zoom_factor)
        
        # Render and save
        plotter.render()
        screenshot = plotter.screenshot(filename=str(output_path), scale=2)
        plotter.close()
        
        print(f"  Saved: {output_path}")
        
        return str(output_path)
    
    def create_comparison_figure(self, output_path,
                                zoom_factor=0.65,
                                image_size=(2400, 3200)):
        """
        Create a comparison figure showing both front and side views.
        
        Args:
            output_path: Path for output image
            zoom_factor: Camera zoom level
            image_size: Image dimensions per view
        """
        # Create temporary images for both views
        temp_dir = Path(output_path).parent / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        front_path = temp_dir / "front_view.png"
        side_path = temp_dir / "side_view.png"
        
        # Generate both views
        self.create_publication_view(front_path, 'front', zoom_factor, image_size)
        self.create_publication_view(side_path, 'side', zoom_factor, image_size)
        
        # Combine using matplotlib
        fig, axes = plt.subplots(1, 2, figsize=(16, 12))
        
        # Load and display images
        import imageio
        front_img = imageio.imread(str(front_path))
        side_img = imageio.imread(str(side_path))
        
        axes[0].imshow(front_img)
        axes[0].axis('off')
        axes[0].set_title('Front View', fontsize=16, pad=20)
        
        axes[1].imshow(side_img)
        axes[1].axis('off')
        axes[1].set_title('Side View', fontsize=16, pad=20)
        
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Clean up temp files
        front_path.unlink()
        side_path.unlink()
        
        print(f"  Saved comparison: {output_path}")
        return str(output_path)


def generate_all_publication_figures(output_base_dir="publication_figures"):
    """
    Generate publication figures for all available meshes.
    
    Args:
        output_base_dir: Base directory for output figures
    """
    # Define all available meshes
    meshes_48h = {
        "Control": "Control_Batch4",
        "KO_1": "KO_Batch1", 
        "KO_2": "KO_Batch2",
        "Rescue_1": "Rescue_Batch1",
        "Rescue_2": "Rescue_Batch2"
    }
    
    meshes_24h = {
        "Control_1": "Control_Batch1",
        "Control_2": "Control_Batch2",
        "Control_3": "Control_Batch3",
        "KO": "KO_Batch1",
        "Rescue_1": "Rescue_Batch1",
        "Rescue_2": "Rescue_Batch2",
        "Rescue_3": "Rescue_Batch3"
    }
    
    # Base path for mesh data
    base_path = Path("/Users/philippkaintoch/Documents/Projects/09_Milos_Revision/Mesh")
    
    # Create output directory structure
    output_dir = Path(__file__).parent / output_base_dir
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("GENERATING PUBLICATION FIGURES")
    print("="*60)
    
    # Process 48h meshes
    print("\n48-HOUR TIMEPOINT:")
    print("-"*40)
    
    output_48h = output_dir / "48h"
    output_48h.mkdir(exist_ok=True)
    
    for label, batch_name in meshes_48h.items():
        mesh_path = base_path / "48h" / batch_name / "Morphology/Analysis/Mesh/ch1"
        surface_path = mesh_path / "surface_1_1.mat"
        curvature_path = mesh_path / "meanCurvature_1_1.mat"
        
        if not surface_path.exists() or not curvature_path.exists():
            print(f"  Skipping {label} - files not found")
            continue
        
        print(f"\nProcessing 48h {label}...")
        
        # Initialize generator
        generator = PublicationFigureGenerator(surface_path, curvature_path)
        
        # Generate front view (main publication figure)
        generator.create_publication_view(
            output_path=output_48h / f"{label}_front.png",
            view_angle='front',
            zoom_factor=0.75
        )
        
        # Generate side view (to show back vessel issue)
        generator.create_publication_view(
            output_path=output_48h / f"{label}_side.png",
            view_angle='side',
            zoom_factor=0.75
        )
    
    # Process 24h meshes
    print("\n24-HOUR TIMEPOINT:")
    print("-"*40)
    
    output_24h = output_dir / "24h"
    output_24h.mkdir(exist_ok=True)
    
    for label, batch_name in meshes_24h.items():
        mesh_path = base_path / "24h" / batch_name / "Morphology/Analysis/Mesh/ch1"
        surface_path = mesh_path / "surface_1_1.mat"
        curvature_path = mesh_path / "meanCurvature_1_1.mat"
        
        if not surface_path.exists() or not curvature_path.exists():
            print(f"  Skipping {label} - files not found")
            continue
        
        print(f"\nProcessing 24h {label}...")
        
        # Initialize generator
        generator = PublicationFigureGenerator(surface_path, curvature_path)
        
        # Generate front view
        generator.create_publication_view(
            output_path=output_24h / f"{label}_front.png",
            view_angle='front',
            zoom_factor=0.75
        )
    
    print("\n" + "="*60)
    print("PUBLICATION FIGURES COMPLETE")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    # Create summary figure comparing Control, KO, and Rescue at 48h
    print("\nGenerating summary comparison figure...")
    create_summary_figure_48h(output_dir, meshes_48h, base_path)


def create_summary_figure_48h(output_dir, meshes_48h, base_path):
    """
    Create a summary figure showing Control, KO, and Rescue side by side.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Rectangle
    import imageio
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.05)
    
    # Define which samples to show
    samples_to_show = [
        ("Control", "Control_Batch4"),
        ("KO", "KO_Batch2"),
        ("Rescue", "Rescue_Batch2")
    ]
    
    for idx, (label, batch_name) in enumerate(samples_to_show):
        mesh_path = base_path / "48h" / batch_name / "Morphology/Analysis/Mesh/ch1"
        surface_path = mesh_path / "surface_1_1.mat"
        curvature_path = mesh_path / "meanCurvature_1_1.mat"
        
        if not surface_path.exists() or not curvature_path.exists():
            continue
        
        # Generate view
        generator = PublicationFigureGenerator(surface_path, curvature_path)
        
        # Create temporary image
        temp_path = output_dir / f"temp_{label}.png"
        generator.create_publication_view(
            output_path=temp_path,
            view_angle='front',
            zoom_factor=0.75
        )
        
        # Load and display
        ax = fig.add_subplot(gs[0, idx])
        img = imageio.imread(str(temp_path))
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(label, fontsize=20, fontweight='bold', pad=15)
        
        # Add annotation for KO showing Y-shape
        if label == "KO":
            # Add subtle arrow or box to highlight Y-shape region
            # This is where the back vessel disconnection creates the Y pattern
            pass
        
        # Clean up temp file
        temp_path.unlink()
    
    # Add main title
    fig.suptitle('48-Hour Vascular Mesh Comparison', fontsize=24, fontweight='bold', y=0.98)
    
    # Add subtle note about back vessel
    fig.text(0.5, 0.02, 
             'Note: Front view shown. Back vessel disconnection in KO/Rescue creates characteristic Y-shape pattern.',
             ha='center', fontsize=10, style='italic', color='gray')
    
    # Save
    summary_path = output_dir / "48h_comparison_summary.png"
    plt.savefig(str(summary_path), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  Saved summary: {summary_path}")


def generate_single_mesh_figure(batch_name, timepoint="48h", output_name=None):
    """
    Generate publication figure for a single mesh.
    
    Args:
        batch_name: Name of the batch (e.g., "Control_Batch4")
        timepoint: "24h" or "48h"
        output_name: Custom output name (optional)
    """
    # Paths
    base_path = Path("/Users/philippkaintoch/Documents/Projects/09_Milos_Revision/Mesh")
    mesh_path = base_path / timepoint / batch_name / "Morphology/Analysis/Mesh/ch1"
    surface_path = mesh_path / "surface_1_1.mat"
    curvature_path = mesh_path / "meanCurvature_1_1.mat"
    
    if not surface_path.exists() or not curvature_path.exists():
        print(f"Error: Mesh files not found for {batch_name}")
        return
    
    # Output setup
    output_dir = Path(__file__).parent / "publication_figures" / timepoint
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if output_name is None:
        output_name = batch_name
    
    print(f"\nGenerating publication figure for {timepoint}/{batch_name}...")
    
    # Initialize generator
    generator = PublicationFigureGenerator(surface_path, curvature_path)
    
    # Generate front view
    front_path = generator.create_publication_view(
        output_path=output_dir / f"{output_name}_front.png",
        view_angle='front',
        zoom_factor=0.65
    )
    
    # Generate side view
    side_path = generator.create_publication_view(
        output_path=output_dir / f"{output_name}_side.png",
        view_angle='side',
        zoom_factor=0.65
    )
    
    print(f"Figures saved to: {output_dir}")
    return front_path, side_path


if __name__ == "__main__":
    # Generate all publication figures
    generate_all_publication_figures()
    
    # Or generate for a single mesh:
    # generate_single_mesh_figure("Control_Batch4", timepoint="48h")