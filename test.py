from mesh_analysis import MeshAnalyzer
from mesh_analysis.visualization import basic_spatial_plot, plot_curvature_distribution
import plotly.graph_objects as go
import numpy as np


surface_path3D = "/Users/philippkaintoch/Documents/Projects/01_Bleb3D/Datensatz/3DPreprocessed/Results/Morphology/Analysis/Mesh/ch1/surface_1_1.mat"
surface_path2D = "/Users/philippkaintoch/Documents/Projects/01_Bleb3D/Datensatz/5_BAIAP2_OE/250415_Galic__B2_BAR_2D__04__GPUdecon/Result/Morphology/Analysis/Mesh/ch1/surface_1_1.mat"
curvature_path3D = "/Users/philippkaintoch/Documents/Projects/01_Bleb3D/Datensatz/3DPreprocessed/Results/Morphology/Analysis/Mesh/ch1/meanCurvature_1_1.mat"
curvature_path2D = "/Users/philippkaintoch/Documents/Projects/01_Bleb3D/Datensatz/5_BAIAP2_OE/250415_Galic__B2_BAR_2D__04__GPUdecon/Result/Morphology/Analysis/Mesh/ch1/meanCurvature_1_1.mat"


analyzer3D = MeshAnalyzer(surface_path3D, curvature_path3D)
analyzer2D = MeshAnalyzer(surface_path2D, curvature_path2D)


analyzer3D.load_data()
analyzer2D.load_data()

stats3D = analyzer3D.calculate_statistics()
stats2D = analyzer2D.calculate_statistics()

curv3D = analyzer3D.curvature
curv2D = analyzer2D.curvature

mesh = analyzer2D.mesh

print(len(analyzer3D.curvature), len(analyzer3D.faces))





#plot_curvature_distribution(curv3D, "/Users/philippkaintoch/Desktop/3D.png")
#plot_curvature_distribution(curv2D, "/Users/philippkaintoch/Desktop/2D.png")

basic_spatial_plot(analyzer3D.mesh, analyzer3D.curvature, "/Users/philippkaintoch/Desktop/3D_Pl.png")
basic_spatial_plot(analyzer2D.mesh, analyzer2D.curvature, "/Users/philippkaintoch/Desktop/2D_Pl.png")