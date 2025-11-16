"""
Rotating Mesh Movie Generator
Creates smooth rotating animations of 3D meshes from u-shape3D output
Author: Philipp Kaintoch
Date: 2025-08-31
"""

import sys
import os
import numpy as np
from pathlib import Path
from vedo import Plotter, Video
import imageio

# Add parent directory to path for MeshAnalyzer import
sys.path.append(str(Path(__file__).parent.parent))
from MeshAnalyzer import MeshAnalyzer


class RotatingMeshMovie:
    """Create rotating movie animations from mesh data."""
    
    def __init__(self, surface_path, curvature_path, output_dir="outputs"):
        """
        Initialize the movie generator.
        
        Args:
            surface_path: Path to surface .mat file
            curvature_path: Path to curvature .mat file
            output_dir: Directory for output movies
        """
        self.surface_path = Path(surface_path)
        self.curvature_path = Path(curvature_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load mesh data
        self.analyzer = MeshAnalyzer(str(self.surface_path), str(self.curvature_path))
        self.analyzer.load_data()
        
    def create_styled_mesh(self):
        """Create styled mesh with semi-transparent surface and wireframe."""
        # Clone the mesh from analyzer
        mesh = self.analyzer.mesh.clone()
        
        # Create semi-transparent solid mesh
        solid_mesh = mesh.clone()
        solid_mesh.color('lightgrey')
        solid_mesh.alpha(0.3)  # 30% opacity
        solid_mesh.lighting('default')
        
        # Create wireframe overlay
        wireframe = mesh.clone()
        wireframe = wireframe.wireframe()
        wireframe.color('darkgrey')
        wireframe.linewidth(0.8)
        wireframe.alpha(0.8)
        
        return solid_mesh, wireframe
    
    def create_rotation_movie(self, output_name="mesh_rotation", 
                            n_frames=180, fps=30, 
                            initial_rotation=(0, 180, 0), zoom_factor=0.6):
        """
        Create a smooth leftward rotating movie of the mesh.
        
        Args:
            output_name: Base name for output file
            n_frames: Number of frames in the rotation (default 180 for 6 seconds)
            fps: Frames per second
            initial_rotation: Tuple of (x, y, z) rotation angles to orient mesh
            zoom_factor: Camera zoom level (default 0.75 for good framing)
        """
        print(f"Creating rotation movie with {n_frames} frames...")
        
        # Get styled meshes
        solid_mesh, wireframe = self.create_styled_mesh()
        
        # Apply initial rotation if specified
        if initial_rotation:
            # Rotate both meshes to orient them properly
            solid_mesh.rotate_x(initial_rotation[0])
            solid_mesh.rotate_y(initial_rotation[1])
            solid_mesh.rotate_z(initial_rotation[2])
            
            wireframe.rotate_x(initial_rotation[0])
            wireframe.rotate_y(initial_rotation[1])
            wireframe.rotate_z(initial_rotation[2])
        
        # Setup plotter with white background
        plotter = Plotter(
            bg='white',
            size=(1200, 800),
            offscreen=True  # Render offscreen for movie creation
        )
        
        # Add meshes to plotter with clean publication style
        plotter.show(
            solid_mesh, wireframe,
            axes=0,  # No axes for clean rotation
            viewup='y',
            resetcam=True,
            interactive=False
        )
        
        # Center the mesh properly using center of mass
        center = solid_mesh.center_of_mass()
        
        # Set camera to front view, looking at center of mass
        plotter.camera.SetPosition(center[0], center[1], center[2] + 1500)
        plotter.camera.SetFocalPoint(center[0], center[1], center[2])
        plotter.camera.SetViewUp(0, 1, 0)
        
        # Apply zoom factor
        plotter.camera.Zoom(zoom_factor)
        
        # Create video writer
        video_path = self.output_dir / f"{output_name}.mp4"
        video = Video(str(video_path), fps=fps, backend='imageio')
        
        # Generate rotation frames
        angles = np.linspace(0, 360, n_frames, endpoint=False)
        
        # Get the center of mass once for rotation
        rotation_center = solid_mesh.center_of_mass()
        
        for i, angle in enumerate(angles):
            # Rotate around the mesh's center of mass (Y-axis, leftward)
            # Shift to origin, rotate, shift back to maintain position
            solid_mesh.shift(-rotation_center).rotate_y(-360/n_frames).shift(rotation_center)
            wireframe.shift(-rotation_center).rotate_y(-360/n_frames).shift(rotation_center)
            
            # Capture frame
            plotter.render()
            video.add_frame()
            
            # Progress indicator
            if (i + 1) % 20 == 0:
                print(f"  Frame {i+1}/{n_frames} rendered")
        
        # Close video and plotter
        video.close()
        plotter.close()
        
        print(f"Movie saved to: {video_path}")
        return str(video_path)
    
    def create_multi_angle_movie(self, output_name="mesh_multi_angle",
                                n_frames=180, fps=30):
        """
        Create a movie with multiple viewing angles.
        
        Args:
            output_name: Base name for output file
            n_frames: Total number of frames
            fps: Frames per second
        """
        print(f"Creating multi-angle movie with {n_frames} frames...")
        
        # Get styled meshes
        solid_mesh, wireframe = self.create_styled_mesh()
        
        # Setup plotter
        plotter = Plotter(
            bg='white',
            size=(1200, 800),
            offscreen=True
        )
        
        # Add meshes
        plotter.show(
            solid_mesh, wireframe,
            axes=dict(
                xyGrid=True, yzGrid=True, zxGrid=True,
                gridLineWidth=1,
                xTitleSize=0, yTitleSize=0, zTitleSize=0,
                numberOfDivisions=10,
                axesLineWidth=2,
                tipSize=0.01
            ),
            viewup='z',
            resetcam=True,
            interactive=False
        )
        
        # Create video writer
        video_path = self.output_dir / f"{output_name}.mp4"
        video = Video(str(video_path), fps=fps, backend='imageio')
        
        # Create animation sequence
        # Phase 1: Rotate around Z axis (60 frames)
        phase1_frames = n_frames // 3
        for i in range(phase1_frames):
            plotter.camera.Azimuth(360/phase1_frames)
            plotter.render()
            video.add_frame()
            if (i + 1) % 20 == 0:
                print(f"  Phase 1 - Frame {i+1}/{phase1_frames}")
        
        # Phase 2: Elevate camera (60 frames)
        phase2_frames = n_frames // 3
        for i in range(phase2_frames):
            plotter.camera.Elevation(90/phase2_frames)
            plotter.camera.Azimuth(180/phase2_frames)
            plotter.render()
            video.add_frame()
            if (i + 1) % 20 == 0:
                print(f"  Phase 2 - Frame {i+1}/{phase2_frames}")
        
        # Phase 3: Return to original view with rotation
        phase3_frames = n_frames - phase1_frames - phase2_frames
        for i in range(phase3_frames):
            plotter.camera.Elevation(-90/phase3_frames)
            plotter.camera.Azimuth(180/phase3_frames)
            plotter.render()
            video.add_frame()
            if (i + 1) % 20 == 0:
                print(f"  Phase 3 - Frame {i+1}/{phase3_frames}")
        
        # Close video and plotter
        video.close()
        plotter.close()
        
        print(f"Multi-angle movie saved to: {video_path}")
        return str(video_path)
    
    def create_gif(self, output_name="mesh_rotation", n_frames=60, fps=15):
        """
        Create an animated GIF of the rotating mesh.
        
        Args:
            output_name: Base name for output file
            n_frames: Number of frames (fewer for smaller file size)
            fps: Frames per second
        """
        print(f"Creating animated GIF with {n_frames} frames...")
        
        # Get styled meshes
        solid_mesh, wireframe = self.create_styled_mesh()
        
        # Setup plotter
        plotter = Plotter(
            bg='white',
            size=(800, 600),  # Smaller size for GIF
            offscreen=True
        )
        
        # Add meshes
        plotter.show(
            solid_mesh, wireframe,
            axes=dict(
                xyGrid=True, yzGrid=True, zxGrid=True,
                gridLineWidth=1,
                xTitleSize=0, yTitleSize=0, zTitleSize=0,
                numberOfDivisions=10,
                axesLineWidth=2,
                tipSize=0.01
            ),
            viewup='z',
            resetcam=True,
            interactive=False
        )
        
        # Collect frames
        frames = []
        angles = np.linspace(0, 360, n_frames, endpoint=False)
        
        for i, angle in enumerate(angles):
            plotter.camera.Azimuth(360/n_frames)
            plotter.render()
            
            # Get screenshot as numpy array
            screenshot = plotter.screenshot(asarray=True)
            frames.append(screenshot)
            
            if (i + 1) % 20 == 0:
                print(f"  Frame {i+1}/{n_frames} captured")
        
        # Save as GIF
        gif_path = self.output_dir / f"{output_name}.gif"
        imageio.mimsave(str(gif_path), frames, fps=fps, loop=0)
        
        plotter.close()
        
        print(f"GIF saved to: {gif_path}")
        return str(gif_path)


def process_batch(batch_name, timepoint="48h", create_gif=True):
    """
    Process a single batch to create movies.
    
    Args:
        batch_name: Name of the batch (e.g., "Control_Batch4")
        timepoint: Timepoint folder ("24h" or "48h")
        create_gif: Whether to also create an animated GIF
    """
    # Define paths
    base_path = Path("/Users/philippkaintoch/Documents/Projects/09_Milos_Revision/Mesh")
    mesh_path = base_path / timepoint / batch_name / "Morphology/Analysis/Mesh/ch1"
    
    surface_path = mesh_path / "surface_1_1.mat"
    curvature_path = mesh_path / "meanCurvature_1_1.mat"
    
    # Check if files exist
    if not surface_path.exists() or not curvature_path.exists():
        print(f"Error: Mesh files not found for {batch_name}")
        return
    
    # Create output directory
    script_dir = Path(__file__).parent
    output_dir = script_dir / "outputs" / timepoint / batch_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize movie maker
    movie_maker = RotatingMeshMovie(
        surface_path=surface_path,
        curvature_path=curvature_path,
        output_dir=output_dir
    )
    
    print(f"\nProcessing {timepoint}/{batch_name}...")
    print("="*50)
    
    # Create leftward rotation movie
    movie_maker.create_rotation_movie(
        output_name=f"{batch_name}_rotation",
        n_frames=180,  # 6 seconds at 30fps
        fps=30,
        initial_rotation=(0, 180, 0),  # Orient mesh with large part down
        zoom_factor=0.6  # Zoom out more for better framing
    )
    
    # Create GIF if requested
    if create_gif:
        movie_maker.create_gif(
            output_name=f"{batch_name}_rotation",
            n_frames=60,
            fps=15
        )
    
    print(f"\nCompleted processing {batch_name}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    # Process Control_Batch4 from 48h timepoint
    process_batch("KO_Batch2", timepoint="48h", create_gif=True)
    
    # Optionally process other batches
    # Uncomment to process additional meshes:
    
    # # Process all 48h batches
    # for batch in ["Control_Batch4", "KO_Batch2", "Rescue_Batch2"]:
    #     process_batch(batch, timepoint="48h", create_gif=True)
    
    # # Process all 24h batches
    # for batch in ["Control_Batch3", "KO_Batch1", "Rescue_Batch3"]:
    #     process_batch(batch, timepoint="24h", create_gif=True)
    
    print("\n" + "="*50)
    print("All movies created successfully!")
    print("Check the outputs folder for results")