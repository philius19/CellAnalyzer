function VesselMeshGeneration()
% VESSELMESHGENERATION Generate mesh from pre-computed vessel masks
% This script uses u-shape3D's loadMask mode to bypass segmentation
% and work directly with pre-computed vessel masks
%
% Based on the working MeshGenerationScript.m architecture with minimal
% changes to support loadMask mode for vascular data.
%
% Key changes for loadMask mode:
%   1. meshMode = 'loadMask' (instead of 'threeLevelSurface')
%   2. Added maskDir and maskName parameters
%   3. Removed threeLevelSurface-specific parameters
%   4. Adjusted scaleOtsu for mask intensity distribution
%   5. Set smoothImageSize to minimal (masks are pre-smoothed)
%
% Author: Generated for vascular mesh processing
% Date: 2025-08-12

%% Set directories
imageDirectory = '/Users/philippkaintoch/Documents/Projects/09_Milos_Revision/focused_improved_masks';
saveDirectory = '/Users/philippkaintoch/Desktop/VesselMeshResults';

%% Set Image Metadata 
pixelSizeXY = 207.556; % nm
pixelSizeZ = 1500;    % nm
timeInterval = 1; 

%% Turn processes on and off
p.control.resetMD = 0; 
p.control.deconvolution = 0;         p.control.deconvolutionReset = 0;
p.control.computeMIP = 1;            p.control.computeMIPReset = 0;
p.control.mesh = 1;                  p.control.meshReset = 1;  % Force regeneration
p.control.meshThres = 0;             p.control.meshThresReset = 0;  % Not needed for loadMask
p.control.surfaceSegment = 0;        p.control.surfaceSegmentReset = 0;
p.control.patchDescribeForMerge = 0; p.control.patchDescribeForMergeReset = 0;
p.control.patchMerge = 0;            p.control.patchMergeReset = 0;
p.control.patchDescribe = 0;         p.control.patchDescribeReset = 0;
p.control.motifDetect = 0;           p.control.motifDetectReset = 0;
p.control.meshMotion = 0;            p.control.meshMotionReset = 0;
p.control.intensity = 0;             p.control.intensityReset = 0;
p.control.intensityBlebCompare = 0; p.control.intensityBlebCompareReset = 0;

cellSegChannel = 1; 
collagenChannel = 1; 
p = setChannels(p, cellSegChannel, collagenChannel);

%% Override Default Parameters for LoadMask Mode

% CHANGE 1: Switch to loadMask mode (instead of threeLevelSurface)
p.mesh.meshMode = 'loadMask';
p.mesh.useUndeconvolved = 1;

% CHANGE 2: Add mask location parameters
p.mesh.maskDir = '/Users/philippkaintoch/Documents/Projects/09_Milos_Revision/focused_improved_masks';
p.mesh.maskName = 'vessel_mask_focused_improved';  % Base name for mask files

% CHANGE 3: Remove threeLevelSurface-specific parameters
% (These are simply not included as they're not needed for loadMask mode)

% CHANGE 4 & 5: Adjust parameters for pre-computed masks
p.mesh.imageGamma = 1;              % No gamma correction needed on masks
p.mesh.scaleOtsu = 0.4;             % Adjusted for mask intensity distribution
p.mesh.smoothImageSize = 1.0;       % Minimal smoothing (masks are pre-smoothed)

% Mesh smoothing parameters (same as original)
p.mesh.smoothMeshMode = 'none';
p.mesh.smoothMeshIterations = 10;

% Additional mesh parameters
p.mesh.removeSmallComponents = 1;
p.mesh.curvatureMedianFilterRadius = 3;

%% Run the analysis

fprintf('\n=== VESSEL MESH GENERATION ===\n');
fprintf('Input directory: %s\n', imageDirectory);
fprintf('Output directory: %s\n', saveDirectory);
fprintf('Mesh mode: %s\n', p.mesh.meshMode);
fprintf('Mask base name: %s\n', p.mesh.maskName);
fprintf('Scale Otsu: %.2f\n', p.mesh.scaleOtsu);
fprintf('Smooth image size: %.1f\n', p.mesh.smoothImageSize);

% Create save directory
if ~isfolder(saveDirectory), mkdir(saveDirectory); end

% Load the movie
fprintf('\nCreating MovieData object...\n');
% Create MovieData directly
MD = makeMovieDataOneChannel(imageDirectory, saveDirectory, pixelSizeXY, pixelSizeZ, timeInterval);

% analyze the cell
morphology3D(MD, p)

plotMeshMD(MD, 'surfaceMode', 'intensity')


