%first try to start bleb3d from script 

%% Set directories
imageDirectory = '/Users/philippkaintoch/Documents/Projects/01_Bleb3D/Datensatz/3D_Lightsheet/1_shortMovie_BAIAP2_OE/ch1t0';
saveDirectory = '/Users/philippkaintoch/Documents/Projects/01_Bleb3D/Output/Batch8';

%% Set Image Metadata 
pixelSizeXY = 103; 
pixelSizeZ = 205.8;
timeInterval = 1; 

%% Turn processes on and off
p.control.resetMD = 0; 
p.control.deconvolution = 0;         p.control.deconvolutionReset = 0;
p.control.computeMIP = 0;            p.control.computeMIPReset = 0;
p.control.mesh = 1;                  p.control.meshReset = 0;
p.control.meshThres = 0;             p.control.meshThresReset = 0;
p.control.surfaceSegment = 0;        p.control.surfaceSegmentReset = 0;
p.control.patchDescribeForMerge = 0; p.control.patchDescribeForMergeReset = 0;
p.control.patchMerge = 0;            p.control.patchMergeReset = 0;
p.control.patchDescribe = 0;         p.control.patchDescribeReset = 0;
p.control.motifDetect = 0;           p.control.motifDetectReset = 0;
p.control.meshMotion = 0;            p.control.meshMotionReset = 0;
p.control.intensity = 0;             p.control.intensityReset = 0;
p.control.intensityBlebCompare = 0; p.control.intensityBlebCompareReset = 0;

%% Override Default Parameters
p.mesh.meshMode = 'twoLevelSurface';
p.mesh.useUndeconvolved = 1;
p.mesh.imageGamma = 0.8;
p.mesh.smoothMeshMode = 'none';

%% Run the analysis

% load the movie
if ~isfolder(saveDirectory), mkdir(saveDirectory); end
MD = makeMovieDataOneChannel(imageDirectory, saveDirectory, pixelSizeXY, pixelSizeZ, timeInterval);

% analyze the cell
morphology3D(MD, p)

% make figures
plotMeshMD(MD, 'surfaceMode', 'curvature'); title('Curvature');