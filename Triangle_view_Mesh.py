from MeshAnalyzer import MeshAnalyzer
from vedo import Plotter
import numpy as np
import vedo

# Load data
surface_path3D = "/Users/philippkaintoch/Documents/Projects/02_Bleb3D/Analyze/2D/4_BAIAP2_OE/Morphology/Analysis/Mesh/ch1/surface_1_1.mat"
curvature_path3D = "/Users/philippkaintoch/Documents/Projects/02_Bleb3D/Analyze/2D/4_BAIAP2_OE/Morphology/Analysis/Mesh/ch1/meanCurvature_1_1.mat"

analyzer_3d = MeshAnalyzer(surface_path3D, curvature_path3D)
analyzer_3d.load_data()

# Get mesh from analyzer (already a vedo mesh)
mesh = analyzer_3d.mesh.clone()


# Apply curvature coloring with symmetric scale
percentile = 98
vmax = np.percentile(np.abs(analyzer_3d.curvature), percentile)

print(analyzer_3d.faces)

#mesh.celldata["curvature"] = analyzer_3d.curvature
#mesh.cmap("RdBu", "curvature", on='cells', vmin=-vmax, vmax=vmax)
#mesh.add_scalarbar(title="Mean Curvature\n(1/pixels)")

# Show
#vedo.show(mesh, axes=7, bg='white')

#surf = mesh.normalize().wireframe().color('white')

#vol = surf.binarize()
#vol.alpha([0,0.75]).cmap('blue5')

#iso = vol.isosurface().color("blue5")

#plt = Plotter(N=2, bg='black')
#plt.at(0).show(vol, surf, __doc__)
#plt.at(1).show("..the volume is isosurfaced:", iso)
#plt.interactive().close()