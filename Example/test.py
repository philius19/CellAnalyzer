from MeshAnalyzer import MeshAnalyzer

surface_path = "/Volumes/T7/Analysis_Neutros/Batch1/Morphology/Analysis/Mesh/ch1/surface_1_1.mat"
curvature_path = "/Volumes/T7/Analysis_Neutros/Batch1/Morphology/Analysis/Mesh/ch1/meanCurvature_1_1.mat"

analyzer = MeshAnalyzer(surface_path, curvature_path)
analyzer.load_data()