#!/usr/bin/env python3
import numpy as np
import h5py
from pathlib import Path
import json
import datetime
from typing import Dict, List, Tuple, Optional
import vedo
from vedo import Plotter, Mesh, Sphere, Text2D, Line
import trimesh
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree


class SomaAnnotationFixed:
    """
    annotation tool 
    """
    
    def __init__(self, surface_path: str, curvature_path: str, dataset_name: str = "unnamed"):
        self.surface_path = Path(surface_path)
        self.curvature_path = Path(curvature_path)
        self.dataset_name = dataset_name
        
        # Data containers
        self.vertices = None
        self.faces = None
        self.vertex_curvature = None
        self.mesh = None
        self.trimesh_mesh = None
        
        # Annotation data
        self.annotations = []
        # CRITICAL: Mesh coordinates are in 0.1 μm units!
        # So 150 mesh units = 15 μm real size
        self.current_soma_radius = 150.0  # Starting radius (15 μm real)
        
        # Measurement mode
        self.measurement_mode = False
        self.measurement_points = []
        self.measurement_lines = []
        
        # UI elements
        self.plotter = None
        self.mesh_actor = None
        self.soma_spheres = []
        self.info_text = None
        self.scale_bar = None
        
    def load_data(self):
        """Load mesh and curvature data"""
        print("Loading mesh data...")
        
        # Load surface
        with h5py.File(self.surface_path, 'r') as f:
            self.vertices = f['surface']['vertices'][:].T
            self.faces = f['surface']['faces'][:].T - 1
            
        # Load curvature
        with h5py.File(self.curvature_path, 'r') as f:
            face_curvature = f['meanCurvature'][0]
            
        # Create trimesh object
        self.trimesh_mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)
        
        # Convert to vertex curvature
        self.vertex_curvature = self._convert_curvature(face_curvature)
        
        # Calculate mesh statistics
        self._calculate_mesh_stats()
        
    def _calculate_mesh_stats(self):
        """Calculate useful mesh statistics with fixed algorithms"""
        # Edge lengths (corrected calculation)
        edges = self.trimesh_mesh.edges_unique
        edge_lengths = np.linalg.norm(
            self.vertices[edges[:, 0]] - self.vertices[edges[:, 1]], 
            axis=1
        )
        
        # Fixed vertex density calculation
        mesh_volume = self.trimesh_mesh.volume if self.trimesh_mesh.is_volume else self.trimesh_mesh.area * 1.0  # Assume 1μm thickness
        vertex_density = len(self.vertices) / mesh_volume if mesh_volume > 0 else 0
        
        self.mesh_stats = {
            'mean_edge_length': np.mean(edge_lengths),
            'median_edge_length': np.median(edge_lengths),
            'vertex_density_per_cubic_um': vertex_density,
            'bbox': self.trimesh_mesh.bounds,
            'centroid': self.trimesh_mesh.centroid
        }
        
        print(f"\nMesh Statistics (CORRECTED - mesh units are 0.1 μm):")
        print(f"  Vertices: {len(self.vertices):,}")
        print(f"  Mean edge length: {self.mesh_stats['mean_edge_length']/10:.2f} μm")
        print(f"  Mesh resolution: ~{self.mesh_stats['median_edge_length']/10:.2f} μm")
        print(f"  Vertex density: {vertex_density*1000:.1f} vertices/μm³")  # Convert from 0.1μm³ to μm³
        bbox_size = self.mesh_stats['bbox'][1] - self.mesh_stats['bbox'][0]
        real_bbox = bbox_size / 10  # Convert to real μm
        print(f"  Neuron size: {real_bbox} μm")
        
    def _convert_curvature(self, face_curvature):
        """Convert face to vertex curvature"""
        vertex_faces = self.trimesh_mesh.vertex_faces
        
        vertex_curvature = np.zeros(len(self.vertices))
        for i in range(len(self.vertices)):
            adjacent_faces = vertex_faces[i]
            valid_faces = adjacent_faces[adjacent_faces >= 0]
            if len(valid_faces) > 0:
                vertex_curvature[i] = np.mean(face_curvature[valid_faces])
                
        return vertex_curvature
        
    def create_visualization(self):
        """Create interactive visualization with fixed scale references"""
        print("\nCreating enhanced visualization...")
        print("=" * 60)
        print("🎯 SOMA ANNOTATION MODE (Default)")
        print("=" * 60)
        print("✅ ONE CLICK = ONE SOMA")
        print("- Left click anywhere on the mesh to mark soma center")
        print("- Green sphere = Valid soma | Orange sphere = Warning")
        print("- Current radius: {:.1f} μm".format(self.current_soma_radius/10))
        print("")
        print("⚙️  CONTROLS:")
        print("- '+/-' keys: Adjust radius")
        print("- '1-9' keys: Quick radius (1=5μm, 2=10μm, etc.)")
        print("- 'u' key: Undo last annotation")
        print("- 's' key: Save annotations") 
        print("- 'm' key: Toggle measurement mode")
        print("=" * 60)
        print("📏 MEASUREMENT MODE:")
        print("- Press 'm' to toggle measurement mode")
        print("- Click two points to measure distance")
        print("- Press 'c' to clear measurements")
        print("=" * 60)
        
        # Create vedo mesh with curvature-based coloring (following test.py approach)
        self.mesh = Mesh([self.vertices, self.faces])
        
        # Convert face curvature to vertex curvature first
        with h5py.File(self.curvature_path, 'r') as f:
            face_curvature = f['meanCurvature'][0]
        
        # Apply curvature coloring with symmetric scale (like test.py)
        percentile = 98
        vmax = np.percentile(np.abs(face_curvature), percentile)
        
        # Use face data directly like test.py
        self.mesh.celldata["curvature"] = face_curvature
        self.mesh.cmap("RdBu", "curvature", on='cells', vmin=-vmax, vmax=vmax)
        
        # Lock the mesh properties to prevent color changes
        self.mesh.lighting('ambient')
        
        print(f"Curvature range: -{vmax:.3f} to {vmax:.3f}")
        print("Color scheme: Red=negative curvature, Blue=positive curvature (like test.py)")
        
        # Create plotter  
        self.plotter = Plotter(title=f"Soma Annotation - {self.dataset_name}")
        self.plotter.add(self.mesh)
        
        # Add scale bar (fixed implementation)
        self._add_scale_bar()
        
        # Update info text
        self._update_info_text_only()
        
        # Set up callbacks with proper order to override defaults
        self.plotter.add_callback('left mouse click', self._on_click)
        self.plotter.add_callback('KeyPress', self._on_keypress)
        
        # Show with interactive=True to ensure our callbacks work
        self.plotter.show(interactive=True)
        
    def _add_scale_bar(self):
        """Add a scale bar to the scene (fixed implementation)"""
        bbox = self.mesh_stats['bbox']
        bbox_size = bbox[1] - bbox[0]
        
        # Dynamic scale bar length based on mesh size (convert to real μm)
        max_dimension_real = np.max(bbox_size) / 10  # Convert to real μm
        if max_dimension_real > 20:
            bar_length_real = 5.0   # 5 μm
        elif max_dimension_real > 10:
            bar_length_real = 2.0   # 2 μm
        else:
            bar_length_real = 1.0   # 1 μm
            
        bar_length = bar_length_real * 10  # Convert back to mesh units
            
        # Position at bottom of scene with validation
        max_dimension = np.max(bbox_size)
        margin = max_dimension * 0.05  # 5% margin
        start_point = [bbox[0][0] + margin, bbox[0][1] + margin, bbox[0][2]]
        end_point = [start_point[0] + bar_length, start_point[1], start_point[2]]
        
        # Create scale bar using Line (since Ruler doesn't exist)
        scale_line = Line(start_point, end_point, lw=5, c='white')
        self.plotter.add(scale_line)
        
        # Add text label (Text2D needs 2D position, show REAL size)
        scale_label = Text2D(f"{bar_length_real:.0f} μm", pos='bottom-left', s=1.2, c='white')
        self.plotter.add(scale_label)
        
        self.scale_bar = [scale_line, scale_label]
        
            
    def _on_click(self, evt):
        """Handle mouse click events"""
        if not evt.actor:
            return
            
        # Get clicked point
        point = evt.picked3d
        if point is None:
            return
            
        if self.measurement_mode:
            self._handle_measurement_click(point)
        else:
            self._add_soma(point)
                
    def _handle_measurement_click(self, point):
        """Handle clicks in measurement mode"""
        self.measurement_points.append(point)
        
        # Add visual marker - smaller but visible
        marker = Sphere(point, r=5, c='cyan', alpha=1.0)  # Smaller cyan marker
        self.measurement_lines.append(marker)  # Store in lines for easy removal
        self.plotter.add(marker)
        print(f"🎯 Measurement point {len(self.measurement_points)} placed")
        
        if len(self.measurement_points) >= 2:
            # Calculate and show distance
            p1 = self.measurement_points[-2]
            p2 = self.measurement_points[-1]
            distance = np.linalg.norm(p2 - p1)
            
            # Create line
            line = Line(p1, p2, c='yellow', lw=5)  # Thicker, bright yellow line
            self.measurement_lines.append(line)
            self.plotter.add(line)
            
            # Add distance label (convert to real μm) - positioned ABOVE the line
            midpoint = (p1 + p2) / 2
            distance_real = distance / 10  # Convert to real μm
            from vedo import Text3D
            # Position label well above the measurement line
            label_pos = [midpoint[0], midpoint[1], midpoint[2] + 30]  # 3μm above
            label = Text3D(f"{distance_real:.1f} μm", pos=label_pos, s=5, c='white')
            self.measurement_lines.append(label)  # Store for removal
            self.plotter.add(label)
            
            print(f"📏 Measured distance: {distance_real:.1f} μm (real)")
            
    def _validate_soma_region(self, center, radius):
        """Validate if a region is soma-like (biological constraints)"""
        # Find vertices within radius
        distances = np.linalg.norm(self.vertices - center, axis=1)
        mask = distances < radius
        n_vertices = np.sum(mask)
        
        # Basic validation criteria (adjusted for high-resolution mesh)
        # With ~0.09 μm resolution, a 20 μm radius soma has ~150k vertices
        min_vertices = 1000      # Minimum vertices for a valid soma
        max_vertices = 500000    # Maximum vertices (allows for high-res somas)
        
        if n_vertices < min_vertices:
            return False, f"Too few vertices ({n_vertices}), region too small for soma"
        if n_vertices > max_vertices:
            return False, f"Too many vertices ({n_vertices}), likely selected entire neuron"
            
        # Check local curvature (somas should have relatively low curvature)
        if len(self.vertex_curvature) > 0:
            local_curvature = self.vertex_curvature[mask]
            mean_curvature = np.mean(np.abs(local_curvature))
            if mean_curvature > 0.5:  # High curvature suggests thin processes, not soma
                return False, f"High curvature ({mean_curvature:.3f}), likely dendrite/axon"
                
        return True, f"Valid soma: {n_vertices} vertices, curvature: {mean_curvature:.3f}"
            
    def _on_keypress(self, evt):
        """Handle keyboard events"""
        key = evt.keypress
        
        # Only handle specific keys to prevent unintended effects
        valid_keys = ['m', 'c', 'u', 's', '+', '-', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        if key not in valid_keys:
            return
            
        # Store original colormap settings for restoration
        if not hasattr(self, '_original_vmax'):
            self._original_vmax = None
            with h5py.File(self.curvature_path, 'r') as f:
                face_curvature = f['meanCurvature'][0]
                self._original_vmax = np.percentile(np.abs(face_curvature), 98)
        
        if key == 'm':  # Toggle measurement mode
            # Clear measurements when switching modes
            if len(self.measurement_lines) > 0:
                print("🧹 Auto-clearing measurements when switching modes...")
                self.measurement_points = []
                for item in self.measurement_lines:
                    try:
                        self.plotter.remove(item)
                    except:
                        pass
                self.measurement_lines = []
            
            self.measurement_mode = not self.measurement_mode
            mode = "MEASUREMENT" if self.measurement_mode else "ANNOTATION"
            print(f"\n🔄 Switched to {mode} mode")
            if self.measurement_mode:
                print("📏 Click two points to measure distance")
            else:
                print("🎯 Click to place soma")
            
            # Update info without affecting mesh
            self._update_info_text_only()
            
                
        elif key == 'c':  # Clear measurements
            print("🧹 Clearing all measurements...")
            self.measurement_points = []
            # Remove all measurement visuals
            for item in self.measurement_lines:
                try:
                    self.plotter.remove(item)
                except:
                    pass  # In case item was already removed
            self.measurement_lines = []
            print("✅ All measurements cleared")
            
        elif key == 'u':  # Undo
            self._undo_last_annotation()
            
        elif key == 's':  # Save
            self._save_annotations()
            
        elif key == '+':  # Increase radius
            self.current_soma_radius = min(500.0, self.current_soma_radius + 20.0)  # 2μm steps
            self._update_info_text()
            print(f"Radius increased to {self.current_soma_radius/10:.1f} μm")
            
        elif key == '-':  # Decrease radius
            self.current_soma_radius = max(50.0, self.current_soma_radius - 20.0)  # 2μm steps
            self._update_info_text()
            print(f"Radius decreased to {self.current_soma_radius/10:.1f} μm")
            
        elif key.isdigit():  # Number keys for quick radius
            if not self.measurement_mode:  # Only in annotation mode
                radius_real = int(key) * 5  # 1=5μm, 2=10μm, etc.
                radius_mesh = radius_real * 10  # Convert to mesh units
                if 50 <= radius_mesh <= 450:  # 5-45 μm real
                    self.current_soma_radius = radius_mesh
                    self._update_info_text()
                    print(f"Radius set to {self.current_soma_radius/10:.1f} μm")
                    
                    # RESTORE COLORMAP after number key (vedo may have changed it)
                    if hasattr(self, '_original_vmax') and self._original_vmax:
                        self.mesh.cmap("RdBu", "curvature", on='cells', 
                                     vmin=-self._original_vmax, vmax=self._original_vmax)
            else:
                print("📏 Number keys disabled in measurement mode")
                
    def _add_soma(self, center):
        """Add a soma annotation with validation - ONE CLICK = ONE SOMA"""
        # Validate the region first
        is_valid, message = self._validate_soma_region(center, self.current_soma_radius)
        
        status_icon = "✅" if is_valid else "⚠️"
        print(f"{status_icon} {message}")
        
        soma = {
            'id': len(self.annotations) + 1,
            'center': center.tolist(),
            'radius': self.current_soma_radius,
            'validation': {'valid': is_valid, 'message': message},
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        self.annotations.append(soma)
        
        # Find closest mesh point to position sphere properly
        distances = np.linalg.norm(self.vertices - center, axis=1)
        closest_vertex_idx = np.argmin(distances)
        closest_point = self.vertices[closest_vertex_idx]
        
        # Position sphere slightly away from mesh surface to avoid penetration
        direction = center - closest_point
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        else:
            direction = np.array([0, 0, 1])  # Default upward
            
        # Offset sphere by a small amount from surface
        offset_distance = self.current_soma_radius * 0.1  # 10% of radius
        sphere_center = center + direction * offset_distance
        
        # Add visual sphere - always grey for consistency
        sphere = Sphere(sphere_center, r=self.current_soma_radius, c='lightgrey', alpha=0.7)
        self.soma_spheres.append(sphere)
        self.plotter.add(sphere)
        
        # Add label well above sphere (show REAL size)
        from vedo import Text3D
        label_pos = [sphere_center[0], sphere_center[1], sphere_center[2] + self.current_soma_radius * 1.5]
        real_radius = self.current_soma_radius / 10  # Convert to real μm
        label = Text3D(f"SOMA {soma['id']}\n{real_radius:.0f}μm", 
                      pos=label_pos, c='white', s=3)
        self.soma_spheres.append(label)
        self.plotter.add(label)
        
        # Update info
        self._update_info_text()
        print(f"🎯 Added SOMA {soma['id']} at {center} with radius {real_radius:.1f} μm")
        print(f"   Status: {'VALID' if is_valid else 'WARNING'}")
        
    def _update_info_text_only(self):
        """Update only the info text without affecting mesh"""
        if self.info_text:
            try:
                self.plotter.remove(self.info_text)
            except:
                pass
            
        mode = "📏 MEASUREMENT" if self.measurement_mode else "🎯 ANNOTATION"
        
        info = f"MODE: {mode}\n"
        info += f"Somas marked: {len(self.annotations)}\n"
        if not self.measurement_mode:
            info += f"Current radius: {self.current_soma_radius/10:.1f} μm\n"
        info += f"Mesh resolution: ~{self.mesh_stats['median_edge_length']/10:.1f} μm\n"
        info += f"Measurements: {len(self.measurement_points)} points\n"
        info += "\nControls:\n"
        info += "m: Toggle mode\n"
        if not self.measurement_mode:
            info += "+/-: Adjust radius | 1-9: Quick size\n"
            info += "u: Undo soma\n"
        else:
            info += "c: Clear measurements\n"
        info += "s: Save annotations"
        
        self.info_text = Text2D(info, pos='top-right', c='black', bg='white', alpha=0.8)
        self.plotter.add(self.info_text)

    def _update_info_text(self):
        """Update information display (legacy wrapper)"""
        self._update_info_text_only()
        
    def _undo_last_annotation(self):
        """Remove the last annotation"""
        if len(self.annotations) > 0:
            removed = self.annotations.pop()
            print(f"🗑️  Removing SOMA {removed['id']}...")
            
            # Remove the last TWO items (sphere + label)
            if len(self.soma_spheres) >= 2:
                # Remove label first (last added)
                label = self.soma_spheres.pop()
                try:
                    self.plotter.remove(label)
                except:
                    pass
                    
                # Remove sphere
                sphere = self.soma_spheres.pop()
                try:
                    self.plotter.remove(sphere)
                except:
                    pass
                    
            elif len(self.soma_spheres) > 0:
                # Fallback - remove whatever is there
                item = self.soma_spheres.pop()
                try:
                    self.plotter.remove(item)
                except:
                    pass
                
            self._update_info_text()
            print(f"✅ Removed SOMA {removed['id']}")
            
    def _save_annotations(self):
        """Save annotations with improved metadata"""
        output_dir = Path("annotations") / self.dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"soma_annotations_fixed_{timestamp}.json"
        
        # Calculate summary statistics
        valid_annotations = [a for a in self.annotations if a['validation']['valid']]
        
        # Convert radii to real micrometers for summary
        radii_real = [a['radius']/10 for a in self.annotations]
        
        data = {
            'dataset': self.dataset_name,
            'file_path': str(self.surface_path),
            'timestamp': datetime.datetime.now().isoformat(),
            'coordinate_system': {
                'mesh_units': '0.1_micrometers',
                'conversion_factor': 10,
                'note': 'Mesh coordinates are in 0.1 μm units. Divide by 10 for real micrometers.'
            },
            'mesh_stats': self.mesh_stats,
            'summary': {
                'total_annotations': len(self.annotations),
                'valid_annotations': len(valid_annotations),
                'mean_radius_mesh_units': np.mean([a['radius'] for a in self.annotations]),
                'mean_radius_micrometers': np.mean(radii_real),
                'radius_range_micrometers': [np.min(radii_real), np.max(radii_real)]
            },
            'somas': self.annotations,
            'method': 'interactive_3D_with_validation_corrected_scale'
        }
        
        # Convert numpy types for JSON serialization
        data['mesh_stats']['bbox'] = data['mesh_stats']['bbox'].tolist()
        data['mesh_stats']['centroid'] = data['mesh_stats']['centroid'].tolist()
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"\nSaved {len(self.annotations)} annotations to {filename}")
        print(f"Valid annotations: {len(valid_annotations)}/{len(self.annotations)}")
        
        # Print summary with corrected units
        print("\nAnnotation Summary (corrected units):")
        for soma in self.annotations:
            status = "✓" if soma['validation']['valid'] else "⚠"
            real_radius = soma['radius'] / 10
            print(f"  {status} Soma {soma['id']}: radius={real_radius:.1f} μm (real)")


def main():
    """Main function"""
    workspace = Path("/Users/philippkaintoch/Documents/Projects/01_Bleb3D/Analyze")
    
    print("=" * 60)
    print("Fixed Soma Annotation Tool")
    print("=" * 60)
    print("\n1. Annotate 3D Culture")
    print("2. Annotate 2D Culture")
    
    choice = input("\nEnter choice (1-2): ")
    
    if choice == '1':
        surface_path = workspace / "3D/1_shortMovie_BAIAP2_OE_T0_Mesh/Morphology/Analysis/Mesh/ch1/surface_1_1.mat"
        curvature_path = workspace / "3D/1_shortMovie_BAIAP2_OE_T0_Mesh/Morphology/Analysis/Mesh/ch1/meanCurvature_1_1.mat"
        
        tool = SomaAnnotationFixed(surface_path, curvature_path, "3D_culture")
        tool.load_data()
        tool.create_visualization()
        
    elif choice == '2':
        surface_path = workspace / "2D/4_BAIAP2_OE/Morphology/Analysis/Mesh/ch1/surface_1_1.mat"
        curvature_path = workspace / "2D/4_BAIAP2_OE/Morphology/Analysis/Mesh/ch1/meanCurvature_1_1.mat"
        
        tool = SomaAnnotationFixed(surface_path, curvature_path, "2D_culture")
        tool.load_data()
        tool.create_visualization()
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()