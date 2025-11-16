from MeshAnalyzer import MeshAnalyzer
from vedo import Plotter

# Load data
surface_path3D = "/Users/philippkaintoch/Documents/Projects/09_Milos_Revision/Mesh/48h/Control_Batch4/Morphology/Analysis/Mesh/ch1/surface_1_1.mat"
curvature_path3D = "/Users/philippkaintoch/Documents/Projects/09_Milos_Revision/Mesh/48h/Control_Batch4/Morphology/Analysis/Mesh/ch1/meanCurvature_1_1.mat"

analyzer_3d = MeshAnalyzer(surface_path3D, curvature_path3D)
analyzer_3d.load_data()

# Get mesh from analyzer (already a vedo mesh)
mesh = analyzer_3d.mesh.clone()

# Style 1: Semi-transparent mesh with wireframe
# Create a semi-transparent solid mesh
solid_mesh = mesh.clone()
solid_mesh.color('lightgrey')  # Light grey surface
solid_mesh.alpha(0.3)  # Semi-transparent (30% opacity)
solid_mesh.lighting('default')  # Default lighting for better depth perception

# Create wireframe overlay
wireframe = mesh.clone()
wireframe = wireframe.wireframe()  # Convert to wireframe
wireframe.color('darkgrey')  # Darker grey for edges
wireframe.linewidth(0.8)  # Thinner lines for cleaner look
wireframe.alpha(0.8)  # Slightly transparent wireframe

# Show both together
plotter = Plotter(bg='white', size=(1200, 800))
plotter.show(solid_mesh, wireframe, 
             axes=dict(xyGrid=True, yzGrid=True, zxGrid=True,  # Show grid on all planes
                      gridLineWidth=1,
                      xTitleSize=0, yTitleSize=0, zTitleSize=0,  # Hide axis titles
                      numberOfDivisions=10,
                      axesLineWidth=2,
                      tipSize=0.01),
             viewup='z',
             interactive=True)

# Alternative Style 2: Pure wireframe grid (uncomment to use)
# wireframe_only = mesh.clone()
# wireframe_only = wireframe_only.wireframe()
# wireframe_only.color('grey')
# wireframe_only.linewidth(1.5)
# 
# vedo.show(wireframe_only, 
#           axes=dict(xyGrid=True, yzGrid=True, zxGrid=True,
#                    gridLineWidth=1,
#                    xTitleSize=0, yTitleSize=0, zTitleSize=0),
#           bg='white',
#           viewup='z',
#           interactive=True)