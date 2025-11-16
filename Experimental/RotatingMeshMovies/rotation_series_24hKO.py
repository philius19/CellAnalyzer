"""
Rotation Series Generator for 24h KO Mesh
Creates a series of static images with incremental leftwise rotation
to find the optimal angle where sprouts overlap
Author: Assistant
Date: 2025-08-31
"""

import sys
import os
import numpy as np
from pathlib import Path
from vedo import Plotter

# Add parent directory to path for MeshAnalyzer import
sys.path.append(str(Path(__file__).parent.parent))
from MeshAnalyzer import MeshAnalyzer


def generate_rotation_series(batch_name="KO_Batch1", 
                           timepoint="24h",
                           rotation_increment=20,
                           total_rotations=18,  # 18 * 20 = 360 degrees
                           output_dir="rotation_series"):
    """
    Generate a series of images with incremental rotation.
    
    Args:
        batch_name: Name of the batch to process
        timepoint: "24h" or "48h"
        rotation_increment: Degrees to rotate for each image
        total_rotations: Number of images to generate
        output_dir: Output directory for images
    """
    
    # Setup paths
    base_path = Path("/Users/philippkaintoch/Documents/Projects/09_Milos_Revision/Mesh")
    mesh_path = base_path / timepoint / batch_name / "Morphology/Analysis/Mesh/ch1"
    surface_path = mesh_path / "surface_1_1.mat"
    curvature_path = mesh_path / "meanCurvature_1_1.mat"
    
    if not surface_path.exists() or not curvature_path.exists():
        print(f"Error: Mesh files not found for {batch_name}")
        return
    
    # Create output directory
    script_dir = Path(__file__).parent
    output_path = script_dir / output_dir / f"{timepoint}_{batch_name}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating rotation series for {timepoint}/{batch_name}")
    print(f"Rotation increment: {rotation_increment}°")
    print(f"Total images: {total_rotations}")
    print("="*60)
    
    # Load mesh data
    analyzer = MeshAnalyzer(str(surface_path), str(curvature_path))
    analyzer.load_data()
    
    # Generate images at different rotation angles
    for i in range(total_rotations):
        rotation_angle = i * rotation_increment
        
        print(f"Generating image {i+1}/{total_rotations}: {rotation_angle}° rotation...")
        
        # Clone the mesh for this iteration
        mesh = analyzer.mesh.clone()
        
        # Create styled mesh components
        solid_mesh = mesh.clone()
        solid_mesh.color('lightgrey')
        solid_mesh.alpha(0.3)
        solid_mesh.lighting('default')
        
        wireframe = mesh.clone()
        wireframe = wireframe.wireframe()
        wireframe.color('darkgrey')
        wireframe.linewidth(0.8)
        wireframe.alpha(0.8)
        
        # Apply rotations:
        # 1. First rotate 180° around Y to orient with large part down
        solid_mesh.rotate_y(180)
        wireframe.rotate_y(180)
        
        # 2. Then apply the incremental leftwise rotation (negative Y rotation)
        solid_mesh.rotate_y(-rotation_angle)
        wireframe.rotate_y(-rotation_angle)
        
        # Setup plotter
        plotter = Plotter(
            bg='white',
            size=(2400, 3200),
            offscreen=True
        )
        
        # Add meshes
        plotter.show(
            solid_mesh, wireframe,
            axes=0,  # No axes for clean figures
            viewup='y',
            resetcam=True,
            interactive=False
        )
        
        # Center the mesh and set camera
        bounds = solid_mesh.bounds()
        center = [(bounds[0] + bounds[1])/2, 
                  (bounds[2] + bounds[3])/2,
                  (bounds[4] + bounds[5])/2]
        
        # Front view camera position
        plotter.camera.SetPosition(center[0], center[1], center[2] + 1500)
        plotter.camera.SetFocalPoint(center[0], center[1], center[2])
        plotter.camera.SetViewUp(0, 1, 0)
        
        # Apply zoom
        plotter.camera.Zoom(0.75)
        
        # Render and save
        output_file = output_path / f"rotation_{rotation_angle:03d}_deg.png"
        plotter.render()
        plotter.screenshot(filename=str(output_file), scale=2)
        plotter.close()
        
        print(f"  Saved: {output_file.name}")
    
    print("\n" + "="*60)
    print(f"Rotation series complete!")
    print(f"Output directory: {output_path}")
    print("\nTo find overlapping sprouts:")
    print("1. Review images in sequence")
    print("2. Note the rotation angle where sprouts best overlap")
    print("3. Use that angle for final publication figure")
    
    # Create an HTML viewer for easy browsing
    create_html_viewer(output_path, rotation_increment, total_rotations)


def create_html_viewer(output_path, rotation_increment, total_rotations):
    """
    Create an HTML file for easy viewing of the rotation series.
    """
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Rotation Series Viewer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            background-color: #f0f0f0;
            margin: 20px;
        }
        #image-container {
            background-color: white;
            display: inline-block;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        #controls {
            margin: 20px;
        }
        button {
            font-size: 16px;
            padding: 10px 20px;
            margin: 0 5px;
            cursor: pointer;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
        }
        button:hover {
            background-color: #45a049;
        }
        #angle-display {
            font-size: 24px;
            font-weight: bold;
            margin: 20px;
            color: #333;
        }
        #image {
            max-width: 90%;
            height: auto;
            border: 1px solid #ddd;
        }
        input[type="range"] {
            width: 400px;
            margin: 20px;
        }
        .info {
            background-color: #e7f3ff;
            padding: 15px;
            border-radius: 5px;
            margin: 20px auto;
            max-width: 600px;
        }
    </style>
</head>
<body>
    <h1>24h KO Mesh - Rotation Series Viewer</h1>
    
    <div class="info">
        <p><strong>Instructions:</strong> Use the controls below to find the angle where sprouts overlap best.</p>
        <p>Rotation is leftwise (counterclockwise when viewed from above)</p>
    </div>
    
    <div id="angle-display">Rotation: 0°</div>
    
    <div id="controls">
        <button onclick="previousImage()">← Previous</button>
        <button onclick="playAnimation()">▶ Play Animation</button>
        <button onclick="stopAnimation()">⏸ Stop</button>
        <button onclick="nextImage()">Next →</button>
    </div>
    
    <input type="range" id="slider" min="0" max="{{MAX_INDEX}}" value="0" 
           oninput="updateImage(this.value)">
    
    <div id="image-container">
        <img id="image" src="rotation_000_deg.png" alt="Mesh rotation">
    </div>
    
    <script>
        const increment = {{INCREMENT}};
        const totalImages = {{TOTAL}};
        let currentIndex = 0;
        let animationId = null;
        
        function updateImage(index) {
            currentIndex = parseInt(index);
            const angle = currentIndex * increment;
            const filename = `rotation_${String(angle).padStart(3, '0')}_deg.png`;
            
            document.getElementById('image').src = filename;
            document.getElementById('angle-display').textContent = `Rotation: ${angle}°`;
            document.getElementById('slider').value = currentIndex;
        }
        
        function nextImage() {
            if (currentIndex < totalImages - 1) {
                updateImage(currentIndex + 1);
            }
        }
        
        function previousImage() {
            if (currentIndex > 0) {
                updateImage(currentIndex - 1);
            }
        }
        
        function playAnimation() {
            if (animationId) return;
            
            animationId = setInterval(() => {
                currentIndex = (currentIndex + 1) % totalImages;
                updateImage(currentIndex);
            }, 500); // Change image every 500ms
        }
        
        function stopAnimation() {
            if (animationId) {
                clearInterval(animationId);
                animationId = null;
            }
        }
        
        // Keyboard controls
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') previousImage();
            if (e.key === 'ArrowRight') nextImage();
            if (e.key === ' ') {
                e.preventDefault();
                if (animationId) stopAnimation();
                else playAnimation();
            }
        });
    </script>
</body>
</html>
    """
    
    # Replace placeholders
    html_content = html_content.replace("{{INCREMENT}}", str(rotation_increment))
    html_content = html_content.replace("{{TOTAL}}", str(total_rotations))
    html_content = html_content.replace("{{MAX_INDEX}}", str(total_rotations - 1))
    
    # Save HTML file
    html_path = output_path / "viewer.html"
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"\nHTML viewer created: {html_path}")
    print("Open this file in a web browser to easily browse through rotations")


def generate_specific_angle(batch_name="KO_Batch1", 
                          timepoint="24h",
                          rotation_angle=60,
                          output_name=None):
    """
    Generate a single image at a specific rotation angle.
    
    Args:
        batch_name: Name of the batch
        timepoint: "24h" or "48h"
        rotation_angle: Specific angle in degrees
        output_name: Custom output filename
    """
    # Setup paths
    base_path = Path("/Users/philippkaintoch/Documents/Projects/09_Milos_Revision/Mesh")
    mesh_path = base_path / timepoint / batch_name / "Morphology/Analysis/Mesh/ch1"
    surface_path = mesh_path / "surface_1_1.mat"
    curvature_path = mesh_path / "meanCurvature_1_1.mat"
    
    # Load mesh
    analyzer = MeshAnalyzer(str(surface_path), str(curvature_path))
    analyzer.load_data()
    
    # Create mesh
    mesh = analyzer.mesh.clone()
    
    solid_mesh = mesh.clone()
    solid_mesh.color('lightgrey')
    solid_mesh.alpha(0.3)
    solid_mesh.lighting('default')
    
    wireframe = mesh.clone()
    wireframe = wireframe.wireframe()
    wireframe.color('darkgrey')
    wireframe.linewidth(0.8)
    wireframe.alpha(0.8)
    
    # Apply rotations
    solid_mesh.rotate_y(180)  # Orient with large part down
    wireframe.rotate_y(180)
    solid_mesh.rotate_y(-rotation_angle)  # Apply specific rotation
    wireframe.rotate_y(-rotation_angle)
    
    # Setup plotter
    plotter = Plotter(bg='white', size=(2400, 3200), offscreen=True)
    plotter.show(solid_mesh, wireframe, axes=0, viewup='y', resetcam=True, interactive=False)
    
    # Center and set camera
    bounds = solid_mesh.bounds()
    center = [(bounds[0] + bounds[1])/2, (bounds[2] + bounds[3])/2, (bounds[4] + bounds[5])/2]
    plotter.camera.SetPosition(center[0], center[1], center[2] + 1500)
    plotter.camera.SetFocalPoint(center[0], center[1], center[2])
    plotter.camera.SetViewUp(0, 1, 0)
    plotter.camera.Zoom(0.75)
    
    # Save
    if output_name is None:
        output_name = f"{timepoint}_{batch_name}_rotation_{rotation_angle}deg.png"
    
    output_path = Path(__file__).parent / "publication_figures" / output_name
    plotter.render()
    plotter.screenshot(filename=str(output_path), scale=2)
    plotter.close()
    
    print(f"Saved: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    # Generate rotation series for 24h KO
    generate_rotation_series(
        batch_name="KO_Batch1",
        timepoint="24h",
        rotation_increment=20,  # 20-degree increments
        total_rotations=18       # Full 360-degree rotation
    )
    
    # After finding the best angle, you can generate a specific one:
    # generate_specific_angle("KO_Batch1", "24h", rotation_angle=60)