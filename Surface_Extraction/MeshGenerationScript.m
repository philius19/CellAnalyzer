function MeshGenerationScript_BioFormats()

% Mesh Generation Script - BioFormats Integration
%
% This script uses u-shape3D's object-oriented architecture with support for
% both simple TIFF series and BioFormats import.
%
% KEY FEATURES:
%   - Toggle between TIFF and BioFormats import (useBioFormats flag)
%   - Uses Process framework correctly
%   - Validates parameters through u-shape3D's system
%   - Compatible with Lattice Light Sheet microscopy
%
% CONTROL PARAMETERS:
%   - p.control.computeMIP: Generate Maximum Intensity Projections
%   - p.control.mesh: Generate 3D surface mesh using threeLevelSegmentation
%   - p.control.meshThres: Create threshold-based segmentation mask
%   - p.control.meshMotion: Calculate surface motion between frames
%   - p.control.render: Export publication figures, movies, and 3D files
%
% If there are Problems with Tiff-Reader, its caused due to hidden files.
% Fix: find /Volumes/T7/Images_Tim/Segment -name "._*" -delete
%
% Author: Philipp Kaintoch
% Date: 2025-12-14, Version 2.2 (added meshMotion + rendering parameters)

%% Set directories
% imageDirectory = '/Volumes/T7/Spinning Disc/2025_11_11_Neutro/Decon/01';
saveDirectory = '/Volumes/T9/LLSM/2025-12-08_09-05-39/251104_BMMC_1_raws_decon/06_Surface_Rendering/02/Batch_3';

%% Set Image Metadata
pixelSizeXY = 103;   % nm
pixelSizeZ = 216;    % nm
timeInterval = 2;

%% Import Configuration
useBioFormats = true;  % Toggle: false = TIFF directory, true = BioFormats file

% For BioFormats mode
imageFile = '/Volumes/T9/LLSM/2025-12-08_09-05-39/251104_BMMC_1_raws_decon/03_Decon/251104_BMMC_mTdt_2_2s_interval__CamA_ch0_decon.tif';
importMetadata = true;

%% Turn processes on and off
p.control.resetMD = 0;
p.control.deconvolution = 0;         p.control.deconvolutionReset = 0;
p.control.computeMIP = 1;            p.control.computeMIPReset = 0;
p.control.mesh = 1;                  p.control.meshReset = 0;
p.control.meshThres = 0;             p.control.meshThresReset = 0;
p.control.surfaceSegment = 0;        p.control.surfaceSegmentReset = 0;
p.control.patchDescribeForMerge = 0; p.control.patchDescribeForMergeReset = 0;
p.control.patchMerge = 0;            p.control.patchMergeReset = 0;
p.control.patchDescribe = 0;         p.control.patchDescribeReset = 0;
p.control.motifDetect = 0;           p.control.motifDetectReset = 0;
p.control.meshMotion = 0;            p.control.meshMotionReset = 0;           % Mesh Motion works now 
p.control.intensity = 0;             p.control.intensityReset = 0;
p.control.intensityBlebCompare = 0;  p.control.intensityBlebCompareReset = 0;
p.control.render = 0;                % Generate publication figures and movies

cellSegChannel = 1;
collagenChannel = 1;
p = setChannels(p, cellSegChannel, collagenChannel);

addpath('/Users/philippkaintoch/Documents/Projects/02_Codebase/Pipeline/Scripts/Surface_Extraction') % Experimental for integrating Silicon-Compatiblity Smoothing  

%% Override Default Parameters
% ALL 18 PARAMETERS with defaults from Mesh3DProcess.getDefaultParams()

% Basic mesh generation mode
p.mesh.meshMode = 'threeLevelSurface';              % Default: 'otsu'
p.mesh.useUndeconvolved = 1;                        % Default: 0

% Three-level segmentation parameters
p.mesh.insideGamma = 0.7;                           % Default: 0.6
p.mesh.insideBlur = 2;                              % Default: 2
p.mesh.filterScales = [1.5, 2, 3];                  % Default: [1.5, 2, 4]
p.mesh.filterNumStdSurface = 2;                     % Default: 2
p.mesh.insideDilateRadius = 5;                      % Default: 5
p.mesh.insideErodeRadius = 6.5;                     % Default: 6.5
p.mesh.steerableType = 2;                           % Default: 1 | 1=line filter, 2=surface filter

% Image preprocessing
p.mesh.imageGamma = 1;                              % Default: 1
p.mesh.scaleOtsu = 1;                               % Default: 1
p.mesh.smoothImageSize = 0;                         % Default: 0

% Mesh smoothing
p.mesh.smoothMeshMode = 'none';                     % Default: 'curvature' | Options: 'curvature', 'none', 'appleSilicon'
p.mesh.smoothMeshIterations = 6;                    % Default: 6

% Curvature computation parameters
p.mesh.curvatureMedianFilterRadius = 2;             % Default: 2
p.mesh.curvatureSmoothOnMeshIterations = 20;        % Default: 20

% Mesh cleanup
p.mesh.removeSmallComponents = 1;                   % Default: 1 | Remove disconnected mesh parts

% Image registration (not used in typical workflow)
p.mesh.registerImages = 0;                          % Default: 0
p.mesh.saveRawImages = 0;                           % Default: 0
p.mesh.registrationMode = 'translation';            % Default: 'translation' | Options: 'translation', 'rigid', 'affine'

%% Mesh Motion Parameters
% Motion analysis measures surface displacement between consecutive frames
p.meshMotion.motionMode = 'backwardsForwards';      % Default: 'backwards' | Options: 'backwards', 'forwards', 'backwardsForwards'
p.meshMotion.numNearestNeighbors = 5;               % Default: 1 | K neighbors for median distance calculation
p.meshMotion.registerImages = 0;                    % Default: 0 | Align frames before motion calculation

%% Rendering Parameters
% Publication-quality figure and movie export settings
p.render.surfaceMode = 'curvature';                 % Options: 'blank', 'curvature', 'motion', 'intensity'
p.render.representativeFrame = 1;                   % Frame for static figures (0 = middle frame)
p.render.setView = [45, 30];                        % Camera angle [azimuth, elevation]
p.render.useBlackBkg = 1;                           % Black background for publication
p.render.makeTimelapse = 1;                         % Generate time-lapse movie
p.render.makeRotation = 1;                          % Generate 360 rotation movie
p.render.makeDAE = 0;                               % Export 3D Collada file
p.render.movieFrameRate = 10;                       % Frames per second for time-lapse
p.render.rotationFrameRate = 30;                    % Frames per second for rotation

%% Run the analysis

% load the movie
if ~isfolder(saveDirectory), mkdir(saveDirectory); end

if useBioFormats
    MD = MovieData(imageFile, importMetadata, 'outputDirectory', saveDirectory);
    if MD.pixelSize_ == 1000
        MD.pixelSize_ = pixelSizeXY;
        MD.pixelSizeZ_ = pixelSizeZ;
        MD.timeInterval_ = timeInterval;
        MD.save;
    end
else
    MD = makeMovieDataOneChannel(imageDirectory, saveDirectory, pixelSizeXY, pixelSizeZ, timeInterval);
end

% analyze the cell
morphology3D(MD, p)

% make figures
plotMeshMD(MD, 'surfaceMode', 'curvature'); title('Curvature');

% Export comprehensive metadata for Python analysis
try
    % Prepare metadata export options with script-level information
    metadataOpts = struct();
    metadataOpts.sourceImageFile = imageFile;
    metadataOpts.pixelSizeXY = pixelSizeXY;
    metadataOpts.pixelSizeZ = pixelSizeZ;
    metadataOpts.timeInterval = timeInterval;
    metadataOpts.meshParams = p.mesh;
    metadataOpts.meshMotionParams = p.meshMotion;
    metadataOpts.verbose = true;

    exportSuccess = exportMetadata(MD, saveDirectory, metadataOpts);
    if exportSuccess
        fprintf('\n=== Comprehensive Metadata Export Complete ===\n');
        fprintf('Python-readable metadata saved to:\n');
        fprintf('  %s/metadata_export.mat\n', saveDirectory);
        fprintf('Contains: acquisition params, processing params, provenance\n\n');
    end
catch ME
    warning('Metadata export failed: %s', ME.message);
    fprintf('Continuing with existing workflow...\n');
end

%% Publication Rendering
if p.control.render
    fprintf('\n Initialise Rendering \n');

    renderPath = fullfile(saveDirectory, 'Renders');
    if ~isfolder(renderPath), mkdir(renderPath); end

    % Determine representative frame
    if p.render.representativeFrame == 0
        repFrame = ceil(MD.nFrames_ / 2);
    else
        repFrame = min(p.render.representativeFrame, MD.nFrames_);
    end

    % Static figures - multiple views
    fprintf('Generating static figures...\n');
    views = {[0, 90], 'XY'; [0, 0], 'XZ'; [90, 0], 'YZ'; p.render.setView, 'iso'};
    for i = 1:size(views, 1)
        try
            [~, fig] = plotMeshMD(MD, ...
                'surfaceMode', p.render.surfaceMode, ...
                'frame', repFrame, ...
                'useBlackBkg', p.render.useBlackBkg, ...
                'setView', views{i,1});
            set(fig, 'Position', [100 100 1200 900]);
            saveas(fig, fullfile(renderPath, sprintf('%s_frame%03d_%s.png', ...
                p.render.surfaceMode, repFrame, views{i,2})));
            close(fig);
        catch
        end
    end

    % Time-lapse movie
    if p.render.makeTimelapse && MD.nFrames_ > 1
        fprintf('Generating time-lapse frames...\n');
        tlPath = fullfile(renderPath, 'TimeLapse');
        if ~isfolder(tlPath), mkdir(tlPath); end

        for t = 1:MD.nFrames_
            try
                [~, fig] = plotMeshMD(MD, ...
                    'surfaceMode', p.render.surfaceMode, ...
                    'frame', t, ...
                    'useBlackBkg', p.render.useBlackBkg, ...
                    'setView', p.render.setView);
                set(fig, 'Position', [100 100 800 600]);
                saveas(fig, fullfile(tlPath, sprintf('frame%03d.tif', t)));
                close(fig);
            catch
            end
        end

        % Compile to AVI
        files = dir(fullfile(tlPath, 'frame*.tif'));
        if ~isempty(files)
            [~, idx] = sort({files.name});
            files = files(idx);
            aviFile = fullfile(renderPath, sprintf('timelapse_%s.avi', p.render.surfaceMode));
            v = VideoWriter(aviFile);
            v.FrameRate = p.render.movieFrameRate;
            open(v);
            for i = 1:length(files)
                writeVideo(v, imread(fullfile(tlPath, files(i).name)));
            end
            close(v);
            fprintf('  Created: %s\n', aviFile);
        end
    end

    % Rotation movie
    if p.render.makeRotation
        fprintf('Generating rotation frames...\n');
        rotPath = fullfile(renderPath, 'Rotation');
        if ~isfolder(rotPath), mkdir(rotPath); end

        try
            plotMeshMD(MD, ...
                'surfaceMode', p.render.surfaceMode, ...
                'frame', repFrame, ...
                'useBlackBkg', p.render.useBlackBkg, ...
                'makeRotation', 1, ...
                'rotSavePath', rotPath);
        catch
        end

        % Compile to AVI
        files = dir(fullfile(rotPath, 'rotate*.tif'));
        if ~isempty(files)
            [~, idx] = sort({files.name});
            files = files(idx);
            aviFile = fullfile(renderPath, sprintf('rotation_%s.avi', p.render.surfaceMode));
            v = VideoWriter(aviFile);
            v.FrameRate = p.render.rotationFrameRate;
            open(v);
            for i = 1:length(files)
                writeVideo(v, imread(fullfile(rotPath, files(i).name)));
            end
            close(v);
            fprintf('  Created: %s\n', aviFile);
        end
    end

    % DAE export
    if p.render.makeDAE
        fprintf('Exporting 3D model...\n');
        daePath = fullfile(renderPath, 'Collada');
        if ~isfolder(daePath), mkdir(daePath); end
        try
            plotMeshMD(MD, ...
                'surfaceMode', p.render.surfaceMode, ...
                'frame', repFrame, ...
                'makeColladaDae', 1, ...
                'daeSavePathMain', daePath);
            fprintf('  Created DAE in: %s\n', daePath);
        catch
        end
    end

    fprintf('Renders saved to: %s\n', renderPath);
end
