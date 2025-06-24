function ThreeLevelSegmentation_Preprocessing_Script()
%% ========================================================================
%  COMPREHENSIVE 3D IMAGE PREPROCESSING SCRIPT FOR BLEB3D SOFTWARE
%  ========================================================================
%
%  PURPOSE:
%  This script performs comprehensive preprocessing of 3D fluorescence 
%  microscopy images using the three-level segmentation approach. It is 
%  designed for beginners to understand each step of the image processing
%  pipeline without performing final mesh generation.
%
%  WHAT THIS SCRIPT DOES:
%  1. Loads 3D microscopy images and creates proper folder structure
%  2. Applies three-level segmentation preprocessing:
%     - Level 1: Inside processing (interior enhancement)
%     - Level 2: Cell processing (overall normalization)
%     - Level 3: Surface processing (membrane detection)
%  3. Saves all intermediate results as TIFF files
%  4. Generates visualization figures for each step
%  5. Creates a final mask ready for bleb3d software
%  6. Logs all parameters and saves MovieData object
%
%  INPUTS REQUIRED FROM USER:
%  - Input image directory path (containing TIFF stack)
%  - Output directory path (where results will be saved)
%  - Pixel size information (XY and Z dimensions)
%
%  OUTPUTS GENERATED:
%  - Organized folder structure with all intermediate images
%  - Processing visualization figures (PNG format)
%  - Final segmentation mask (TIFF format)
%  - Parameter log file (TXT format)
%  - MovieData object (MAT format)
%
%  FOR BEGINNERS:
%  This script includes detailed comments explaining each step. You can
%  run the entire script or execute sections individually to understand
%  how each processing step affects your images.
%
%  AUTHOR: Created for 3D Cell Morphology Analysis
%  DATE: 2025
%
%% ========================================================================

%% ========================================================================
%  USER INPUT SECTION - MODIFY THESE PATHS FOR YOUR DATA
%% ========================================================================

% *** MODIFY THESE PATHS FOR YOUR SPECIFIC DATA ***
% Input directory containing your 3D TIFF images
imageDirectory = '/Users/philippkaintoch/Documents/Projects/01_Bleb3D/Datensatz/3D_Lightsheet/1_shortMovie_BAIAP2_OE/ch1t0';

% Output directory where all results will be saved
outputBaseDirectory = '/Users/philippkaintoch/Desktop/Results';

% Imaging parameters - MODIFY THESE FOR YOUR MICROSCOPE SETTINGS
pixelSizeXY = 103;          % Pixel size in XY plane (nanometers)
pixelSizeZ = 205.8;         % Pixel size in Z direction (nanometers)
timeInterval = 1;           % Time interval (seconds) - use 1 for single timepoint

% Processing parameters - ADVANCED USERS CAN MODIFY THESE
scales = [1.5, 2, 4];       % Surface filter scales (pixel units)
nSTDsurface = 2;            % Surface threshold multiplier (higher = more selective)
insideGamma = 0.6;          % Gamma correction (0.6 enhances dim regions)
insideBlur = 2;             % Gaussian blur sigma for noise reduction
insideDilateRadius = 5;     % Morphological dilation radius (pixels)
insideErodeRadius = 6.5;    % Morphological erosion radius (pixels)

%% ========================================================================
%  AUTOMATIC FOLDER STRUCTURE CREATION
%% ========================================================================

fprintf('\n=== STARTING 3D IMAGE PREPROCESSING ===\n');
fprintf('Creating organized folder structure...\n');

% Extract image name from input directory for main folder naming
[~, imageName, ~] = fileparts(imageDirectory);
mainOutputDir = fullfile(outputBaseDirectory, imageName);

% Create main output directory
if ~exist(mainOutputDir, 'dir')
    mkdir(mainOutputDir);
    fprintf('Created main directory: %s\n', mainOutputDir);
end

% Create level-specific directories
level1Dir = fullfile(mainOutputDir, 'Level1_Inside_Processing');
level2Dir = fullfile(mainOutputDir, 'Level2_Cell_Processing');
level3Dir = fullfile(mainOutputDir, 'Level3_Surface_Processing');
finalDir = fullfile(mainOutputDir, 'Final_Combination');

% Create directories for each level
mkdir(level1Dir); mkdir(level2Dir); mkdir(level3Dir); mkdir(finalDir);

% Create section subdirectories within each level
% Level 1: Sections 1-11 (Setup through Inside processing)
for i = 1:11
    mkdir(fullfile(level1Dir, sprintf('Section_%02d', i)));
end

% Level 2: Sections 12-13 (Cell processing)
for i = 12:13
    mkdir(fullfile(level2Dir, sprintf('Section_%02d', i)));
end

% Level 3: Sections 14-15 (Surface processing)
for i = 14:15
    mkdir(fullfile(level3Dir, sprintf('Section_%02d', i)));
end

% Final: Sections 16-18 (Combination and finalization)
for i = 16:18
    mkdir(fullfile(finalDir, sprintf('Section_%02d', i)));
end

fprintf('Folder structure created successfully!\n\n');

%% ========================================================================
%  SECTION 1-2: SETUP AND DATA LOADING
%% ========================================================================

fprintf('=== SECTION 1-2: SETUP AND DATA LOADING ===\n');

% Create MovieData save directory (required by bleb3d software)
saveDirectory = fullfile(mainOutputDir, 'MovieData');
if ~exist(saveDirectory, 'dir')
    mkdir(saveDirectory);
end

% Create MovieData object (handles metadata for bleb3d compatibility)
fprintf('Creating MovieData object...\n');
MD = makeMovieDataOneChannel(imageDirectory, saveDirectory, pixelSizeXY, pixelSizeZ, timeInterval);

% Load the 3D image stack
fprintf('Loading 3D image stack...\n');
image3D_original = im2double(MD.getChannel(1).loadStack(1));

% Display basic information about the loaded image
fprintf('Image loaded successfully!\n');
fprintf('  Original size: %d x %d x %d voxels\n', size(image3D_original));
fprintf('  Pixel size XY: %.1f nm\n', pixelSizeXY);
fprintf('  Pixel size Z: %.1f nm\n', pixelSizeZ);
fprintf('  Intensity range: %.4f to %.4f\n', min(image3D_original(:)), max(image3D_original(:)));

% Save original image and create visualization
section01Dir = fullfile(level1Dir, 'Section_01');
save3DImage(uint16(image3D_original * 65535), fullfile(section01Dir, 'original_image.tif'));

% Create and save visualization figure
fig1 = figure('Name', 'Section 1: Original Image', 'Visible', 'off');
imagesc(image3D_original(:,:,round(end/2))); 
colormap(gray); axis equal; axis off; title('Original 3D Image (Middle Slice)');
colorbar;
saveas(fig1, fullfile(section01Dir, 'original_image_visualization.png'));
close(fig1);

fprintf('Section 1-2 completed. Results saved to Level1/Section_01/\n\n');

%% ========================================================================
%  SECTION 3: MAKE VOXELS ISOTROPIC
%% ========================================================================

fprintf('=== SECTION 3: MAKING VOXELS ISOTROPIC ===\n');

% WHAT THIS DOES:
% Makes all voxels cubic by adjusting for different XY vs Z pixel sizes.
% This ensures that morphological operations (like dilation/erosion) work
% correctly regardless of your microscope's acquisition settings.

fprintf('Converting to isotropic voxels...\n');
image3D = make3DImageVoxelsSymmetric(image3D_original, MD.pixelSize_, MD.pixelSizeZ_);

scalingFactor = pixelSizeZ / pixelSizeXY;
fprintf('  Isotropic size: %d x %d x %d voxels\n', size(image3D));
fprintf('  Z scaling factor: %.2f\n', scalingFactor);

% Save isotropic image
section03Dir = fullfile(level1Dir, 'Section_03');
save3DImage(uint16(image3D * 65535), fullfile(section03Dir, 'isotropic_image.tif'));

% Create comparison visualization
fig3 = figure('Name', 'Section 3: Isotropic Conversion', 'Visible', 'off');
subplot(1,2,1); imagesc(image3D_original(:,:,round(end/2))); 
colormap(gray); axis equal; title('Original'); colorbar;
subplot(1,2,2); imagesc(image3D(:,:,round(end/2))); 
colormap(gray); axis equal; title('Isotropic'); colorbar;
saveas(fig3, fullfile(section03Dir, 'isotropic_comparison.png'));
close(fig3);

fprintf('Section 3 completed. Isotropic image saved to Level1/Section_03/\n\n');

%% ========================================================================
%  SECTION 4: ADD BLACK BORDER
%% ========================================================================

fprintf('=== SECTION 4: ADDING BLACK BORDER ===\n');

% WHAT THIS DOES:
% Adds a 1-pixel border around the image to prevent edge artifacts during
% filtering and morphological operations. The border is filled with the
% median intensity value to avoid creating artificial boundaries.

fprintf('Adding protective border...\n');
image3D_bordered = addBlackBorder(image3D, 1);

fprintf('  Size after border: %d x %d x %d voxels\n', size(image3D_bordered));
fprintf('  Border filled with median value: %.4f\n', median(image3D(:)));

% Save bordered image
section04Dir = fullfile(level1Dir, 'Section_04');
save3DImage(uint16(image3D_bordered * 65535), fullfile(section04Dir, 'bordered_image.tif'));

% Create border visualization (showing corner region)
fig4 = figure('Name', 'Section 4: Border Addition', 'Visible', 'off');
subplot(1,2,1); imagesc(image3D(1:50,1:50,round(end/2))); 
colormap(gray); axis equal; title('No Border'); colorbar;
subplot(1,2,2); imagesc(image3D_bordered(1:50,1:50,round(end/2))); 
colormap(gray); axis equal; title('With Border'); colorbar;
saveas(fig4, fullfile(section04Dir, 'border_comparison.png'));
close(fig4);

fprintf('Section 4 completed. Bordered image saved to Level1/Section_04/\n\n');

%% ========================================================================
%  SECTIONS 5-11: LEVEL 1 - INSIDE PROCESSING
%% ========================================================================

fprintf('=== LEVEL 1: INSIDE PROCESSING (SECTIONS 5-11) ===\n');
fprintf('This level enhances the interior regions of the cell.\n\n');

% SECTION 5: GAMMA CORRECTION
fprintf('--- Section 5: Gamma Correction ---\n');
% WHAT THIS DOES:
% Applies gamma correction to enhance dim interior regions. Gamma < 1 
% brightens dark areas more than bright areas, making cell interiors
% more visible for better thresholding.

image3D_gamma = image3D_bordered.^insideGamma;
fprintf('  Gamma correction applied: %.2f\n', insideGamma);
fprintf('  Value range after gamma: %.4f to %.4f\n', min(image3D_gamma(:)), max(image3D_gamma(:)));

section05Dir = fullfile(level1Dir, 'Section_05');
save3DImage(uint16(image3D_gamma * 65535), fullfile(section05Dir, 'gamma_corrected.tif'));

fig5 = figure('Name', 'Section 5: Gamma Correction', 'Visible', 'off');
subplot(2,2,1); histogram(image3D_bordered(:), 100); title('Original Histogram');
subplot(2,2,2); histogram(image3D_gamma(:), 100); title('After Gamma Correction');
subplot(2,2,3); imagesc(image3D_bordered(:,:,round(end/2))); 
colormap(gray); axis equal; title('Original'); colorbar;
subplot(2,2,4); imagesc(image3D_gamma(:,:,round(end/2))); 
colormap(gray); axis equal; title('Gamma Corrected'); colorbar;
saveas(fig5, fullfile(section05Dir, 'gamma_correction_comparison.png'));
close(fig5);

% SECTION 6: GAUSSIAN BLUR
fprintf('--- Section 6: Gaussian Blur ---\n');
% WHAT THIS DOES:
% Applies Gaussian blur to reduce noise and create smoother regions.
% This helps create more reliable thresholding results.

image3D_blurred = filterGauss3D(image3D_gamma, insideBlur);
fprintf('  Gaussian blur applied with sigma = %.1f\n', insideBlur);

section06Dir = fullfile(level1Dir, 'Section_06');
save3DImage(uint16(image3D_blurred * 65535), fullfile(section06Dir, 'blurred_image.tif'));

fig6 = figure('Name', 'Section 6: Gaussian Blur', 'Visible', 'off');
subplot(1,2,1); imagesc(image3D_gamma(:,:,round(end/2))); 
colormap(gray); axis equal; title('After Gamma'); colorbar;
subplot(1,2,2); imagesc(image3D_blurred(:,:,round(end/2))); 
colormap(gray); axis equal; title('After Blur'); colorbar;
saveas(fig6, fullfile(section06Dir, 'blur_comparison.png'));
close(fig6);

% SECTION 7: OTSU THRESHOLDING
fprintf('--- Section 7: Otsu Thresholding ---\n');
% WHAT THIS DOES:
% Automatically determines the optimal threshold to separate foreground
% (cell) from background using Otsu's method, which minimizes the
% variance within each class.

insideThreshValue = thresholdOtsu(image3D_blurred(:));
image3D_thresh_binary = image3D_blurred > insideThreshValue;

fprintf('  Otsu threshold value: %.4f\n', insideThreshValue);
fprintf('  Pixels above threshold: %.1f%%\n', 100*sum(image3D_thresh_binary(:))/numel(image3D_thresh_binary));

section07Dir = fullfile(level1Dir, 'Section_07');
save3DImage(uint8(image3D_thresh_binary * 255), fullfile(section07Dir, 'binary_mask.tif'));

fig7 = figure('Name', 'Section 7: Otsu Thresholding', 'Visible', 'off');
subplot(1,3,1); imagesc(image3D_blurred(:,:,round(end/2))); 
colormap(gray); axis equal; title('Blurred'); colorbar;
subplot(1,3,2); imagesc(image3D_thresh_binary(:,:,round(end/2))); 
colormap(gray); axis equal; title('Binary Mask'); colorbar;
subplot(1,3,3); histogram(image3D_blurred(:), 100); hold on;
line([insideThreshValue insideThreshValue], ylim, 'Color', 'r', 'LineWidth', 2);
title('Histogram with Threshold'); legend('Histogram', 'Otsu Threshold');
saveas(fig7, fullfile(section07Dir, 'thresholding_result.png'));
close(fig7);

% SECTION 8: MORPHOLOGICAL DILATION
fprintf('--- Section 8: Morphological Dilation ---\n');
% WHAT THIS DOES:
% Expands the detected regions to fill small gaps and connect nearby
% objects. Uses a spherical structuring element for isotropic expansion.

sphereDilate = makeSphere3D(insideDilateRadius);
image3D_dilated = imdilate(image3D_thresh_binary, sphereDilate);

fprintf('  Dilation applied with radius = %.1f pixels\n', insideDilateRadius);
fprintf('  Pixels after dilation: %.1f%% (was %.1f%%)\n', ...
    100*sum(image3D_dilated(:))/numel(image3D_dilated), ...
    100*sum(image3D_thresh_binary(:))/numel(image3D_thresh_binary));

section08Dir = fullfile(level1Dir, 'Section_08');
save3DImage(uint8(image3D_dilated * 255), fullfile(section08Dir, 'dilated_mask.tif'));

fig8 = figure('Name', 'Section 8: Morphological Dilation', 'Visible', 'off');
subplot(1,3,1); imagesc(image3D_thresh_binary(:,:,round(end/2))); 
colormap(gray); axis equal; title('Binary'); colorbar;
subplot(1,3,2); imagesc(image3D_dilated(:,:,round(end/2))); 
colormap(gray); axis equal; title('Dilated'); colorbar;
subplot(1,3,3); imagesc(image3D_dilated(:,:,round(end/2)) - image3D_thresh_binary(:,:,round(end/2))); 
colormap(gray); axis equal; title('Added Pixels'); colorbar;
saveas(fig8, fullfile(section08Dir, 'dilation_comparison.png'));
close(fig8);

% SECTION 9: HOLE FILLING
fprintf('--- Section 9: Hole Filling ---\n');
% WHAT THIS DOES:
% Fills interior holes within detected regions using morphological
% reconstruction. Done slice-by-slice for computational efficiency.

image3D_filled = image3D_dilated;
for h = 1:size(image3D_filled, 3)
    image3D_filled(:,:,h) = imfill(image3D_filled(:,:,h), 'holes');
end

fprintf('  Hole filling completed on %d slices\n', size(image3D_filled, 3));

section09Dir = fullfile(level1Dir, 'Section_09');
save3DImage(uint8(image3D_filled * 255), fullfile(section09Dir, 'filled_mask.tif'));

fig9 = figure('Name', 'Section 9: Hole Filling', 'Visible', 'off');
subplot(1,3,1); imagesc(image3D_dilated(:,:,round(end/2))); 
colormap(gray); axis equal; title('Dilated'); colorbar;
subplot(1,3,2); imagesc(image3D_filled(:,:,round(end/2))); 
colormap(gray); axis equal; title('Holes Filled'); colorbar;
subplot(1,3,3); imagesc(image3D_filled(:,:,round(end/2)) - image3D_dilated(:,:,round(end/2))); 
colormap(gray); axis equal; title('Filled Holes'); colorbar;
saveas(fig9, fullfile(section09Dir, 'hole_filling_comparison.png'));
close(fig9);

% SECTION 10: MORPHOLOGICAL EROSION
fprintf('--- Section 10: Morphological Erosion ---\n');
% WHAT THIS DOES:
% Contracts regions back toward original size and smooths boundaries.
% The erosion radius is typically larger than dilation to ensure
% proper shape refinement.

sphereErode = makeSphere3D(insideErodeRadius);
image3D_eroded = double(imerode(image3D_filled, sphereErode));

fprintf('  Erosion applied with radius = %.1f pixels\n', insideErodeRadius);
fprintf('  Pixels after erosion: %.1f%% (was %.1f%%)\n', ...
    100*sum(image3D_eroded(:))/numel(image3D_eroded), ...
    100*sum(image3D_filled(:))/numel(image3D_filled));

section10Dir = fullfile(level1Dir, 'Section_10');
save3DImage(uint8(image3D_eroded * 255), fullfile(section10Dir, 'eroded_mask.tif'));

fig10 = figure('Name', 'Section 10: Morphological Erosion', 'Visible', 'off');
subplot(1,3,1); imagesc(image3D_filled(:,:,round(end/2))); 
colormap(gray); axis equal; title('Filled'); colorbar;
subplot(1,3,2); imagesc(image3D_eroded(:,:,round(end/2))); 
colormap(gray); axis equal; title('Eroded'); colorbar;
subplot(1,3,3); imagesc(image3D_filled(:,:,round(end/2)) - image3D_eroded(:,:,round(end/2))); 
colormap(gray); axis equal; title('Removed Pixels'); colorbar;
saveas(fig10, fullfile(section10Dir, 'erosion_comparison.png'));
close(fig10);

% SECTION 11: FINAL SMOOTHING
fprintf('--- Section 11: Final Smoothing ---\n');
% WHAT THIS DOES:
% Applies light Gaussian smoothing to create smooth transitions for
% better mesh generation. This creates the final "inside" level image.

image3D_inside_final = filterGauss3D(image3D_eroded, 1);

fprintf('  Final smoothing applied (sigma = 1)\n');
fprintf('  Inside level processing complete\n');
fprintf('  Final range: %.4f to %.4f\n', min(image3D_inside_final(:)), max(image3D_inside_final(:)));

section11Dir = fullfile(level1Dir, 'Section_11');
save3DImage(uint16((image3D_inside_final - min(image3D_inside_final(:))) / ...
    (max(image3D_inside_final(:)) - min(image3D_inside_final(:))) * 65535), ...
    fullfile(section11Dir, 'inside_level_final.tif'));

fig11 = figure('Name', 'Section 11: Final Inside Level', 'Visible', 'off');
subplot(1,2,1); imagesc(image3D_eroded(:,:,round(end/2))); 
colormap(gray); axis equal; title('Eroded'); colorbar;
subplot(1,2,2); imagesc(image3D_inside_final(:,:,round(end/2))); 
colormap(gray); axis equal; title('Inside Final'); colorbar;
saveas(fig11, fullfile(section11Dir, 'inside_final_comparison.png'));
close(fig11);

fprintf('Level 1 (Inside Processing) completed!\n\n');

%% ========================================================================
%  SECTIONS 12-13: LEVEL 2 - CELL PROCESSING
%% ========================================================================

fprintf('=== LEVEL 2: CELL PROCESSING (SECTIONS 12-13) ===\n');
fprintf('This level creates normalized intensity image for overall cell contrast.\n\n');

% SECTION 12: CELL IMAGE NORMALIZATION
fprintf('--- Section 12: Cell Image Normalization ---\n');
% WHAT THIS DOES:
% Creates a normalized intensity image that provides overall cell contrast
% information. This complements the inside processing by preserving
% intensity variations across the cell.

% Calculate Otsu threshold for normalization baseline
cellThreshValue = thresholdOtsu(image3D_bordered(:));
fprintf('  Cell Otsu threshold: %.4f\n', cellThreshValue);

% Subtract threshold and normalize by standard deviation
image3D_cell_subtracted = image3D_bordered - cellThreshValue;
image3D_cell_normalized = image3D_cell_subtracted / std(image3D_cell_subtracted(:));

fprintf('  Cell image normalized\n');
fprintf('  Range after normalization: %.3f to %.3f\n', min(image3D_cell_normalized(:)), max(image3D_cell_normalized(:)));

section12Dir = fullfile(level2Dir, 'Section_12');
save3DImage(uint16((image3D_cell_normalized - min(image3D_cell_normalized(:))) / ...
    (max(image3D_cell_normalized(:)) - min(image3D_cell_normalized(:))) * 65535), ...
    fullfile(section12Dir, 'cell_level_normalized.tif'));

% Create normalization visualization
fig12 = figure('Name', 'Section 12: Cell Normalization', 'Visible', 'off');
subplot(2,2,1); imagesc(image3D_bordered(:,:,round(end/2))); 
colormap(gray); axis equal; title('Original'); colorbar;
subplot(2,2,2); imagesc(image3D_cell_subtracted(:,:,round(end/2))); 
colormap(gray); axis equal; title('Threshold Subtracted'); colorbar;
subplot(2,2,3); imagesc(image3D_cell_normalized(:,:,round(end/2))); 
colormap(gray); axis equal; title('Normalized'); colorbar;
subplot(2,2,4); histogram(image3D_cell_normalized(:), 100); 
title('Normalized Histogram'); xlabel('Intensity'); ylabel('Count');
saveas(fig12, fullfile(section12Dir, 'cell_normalization_process.png'));
close(fig12);

fprintf('Level 2 (Cell Processing) completed!\n\n');

%% ========================================================================
%  SECTIONS 14-15: LEVEL 3 - SURFACE PROCESSING
%% ========================================================================

fprintf('=== LEVEL 3: SURFACE PROCESSING (SECTIONS 14-15) ===\n');
fprintf('This level detects membrane/surface structures at multiple scales.\n\n');

% SECTION 14: MULTISCALE SURFACE FILTER
fprintf('--- Section 14: Multiscale Surface Filter ---\n');
% WHAT THIS DOES:
% Applies surface detection filtering at multiple scales using second
% derivatives. This specifically detects membrane-like structures that
% appear as ridges or valleys in the intensity landscape.

fprintf('  Applying multiscale surface filter...\n');
fprintf('  Scales: [%.1f, %.1f, %.1f]\n', scales);

q.SigmasXY = scales; 
q.SigmasZ = scales; 
q.WeightZ = 1;
[maxResp, ~, ~, ~, maxRespScale] = multiscaleSurfaceFilter3D(image3D_cell_normalized, q);

fprintf('  Surface filter response computed\n');
fprintf('  Response range: %.4f to %.4f\n', min(maxResp(:)), max(maxResp(:)));

section14Dir = fullfile(level3Dir, 'Section_14');
save3DImage(uint16((maxResp - min(maxResp(:))) / (max(maxResp(:)) - min(maxResp(:))) * 65535), ...
    fullfile(section14Dir, 'surface_filter_response.tif'));

% Save scale map
save3DImage(uint8(maxRespScale), fullfile(section14Dir, 'surface_scale_map.tif'));

% Create surface filter visualization
fig14 = figure('Name', 'Section 14: Surface Filter', 'Visible', 'off');
subplot(1,3,1); imagesc(image3D_cell_normalized(:,:,round(end/2))); 
colormap(gray); axis equal; title('Normalized Cell'); colorbar;
subplot(1,3,2); imagesc(maxResp(:,:,round(end/2))); 
colormap(gray); axis equal; title('Surface Response'); colorbar;
subplot(1,3,3); imagesc(maxRespScale(:,:,round(end/2))); 
colormap(jet); axis equal; title('Scale Map'); colorbar;
saveas(fig14, fullfile(section14Dir, 'surface_filter_results.png'));
close(fig14);

% SECTION 15: SURFACE THRESHOLDING AND NORMALIZATION
fprintf('--- Section 15: Surface Thresholding and Normalization ---\n');
% WHAT THIS DOES:
% Thresholds the surface response using statistical criteria (mean + N*std)
% and normalizes the result for combination with other levels.

% Calculate surface threshold
surfBackMean = mean(maxResp(:));
surfBackSTD = std(maxResp(:));
surfThresh = surfBackMean + (nSTDsurface * surfBackSTD);

fprintf('  Surface statistics:\n');
fprintf('    Mean: %.4f, Std: %.4f\n', surfBackMean, surfBackSTD);
fprintf('    Threshold: %.4f (mean + %.1f*std)\n', surfThresh, nSTDsurface);

% Apply threshold and normalize
image3D_surface_thresholded = maxResp - surfThresh;
image3D_surface_final = image3D_surface_thresholded / std(image3D_surface_thresholded(:));

fprintf('  Surface processing complete\n');
fprintf('  Final range: %.3f to %.3f\n', min(image3D_surface_final(:)), max(image3D_surface_final(:)));

section15Dir = fullfile(level3Dir, 'Section_15');
save3DImage(uint16((image3D_surface_final - min(image3D_surface_final(:))) / ...
    (max(image3D_surface_final(:)) - min(image3D_surface_final(:))) * 65535), ...
    fullfile(section15Dir, 'surface_level_final.tif'));

% Create surface processing visualization
fig15 = figure('Name', 'Section 15: Surface Processing', 'Visible', 'off');
subplot(2,2,1); imagesc(maxResp(:,:,round(end/2))); 
colormap(gray); axis equal; title('Surface Response'); colorbar;
subplot(2,2,2); imagesc(maxResp(:,:,round(end/2)) > surfThresh); 
colormap(gray); axis equal; title('Above Threshold'); colorbar;
subplot(2,2,3); imagesc(image3D_surface_thresholded(:,:,round(end/2))); 
colormap(gray); axis equal; title('Thresholded'); colorbar;
subplot(2,2,4); imagesc(image3D_surface_final(:,:,round(end/2))); 
colormap(gray); axis equal; title('Surface Final'); colorbar;
saveas(fig15, fullfile(section15Dir, 'surface_processing_steps.png'));
close(fig15);

fprintf('Level 3 (Surface Processing) completed!\n\n');

%% ========================================================================
%  SECTIONS 16-18: FINAL COMBINATION
%% ========================================================================

fprintf('=== FINAL COMBINATION (SECTIONS 16-18) ===\n');
fprintf('Combining all three levels and creating final segmentation mask.\n\n');

% SECTION 16: COMBINE ALL THREE LEVELS
fprintf('--- Section 16: Combining All Three Levels ---\n');
% WHAT THIS DOES:
% Merges information from all three processing streams using the MAX
% operation. This preserves the best features from each level while
% maintaining the overall structure.

combinedImage = max(max(image3D_inside_final, image3D_cell_normalized), image3D_surface_final);

fprintf('  Three levels combined using MAX operation\n');
fprintf('  Combined range: %.3f to %.3f\n', min(combinedImage(:)), max(combinedImage(:)));

section16Dir = fullfile(finalDir, 'Section_16');
save3DImage(uint16((combinedImage - min(combinedImage(:))) / ...
    (max(combinedImage(:)) - min(combinedImage(:))) * 65535), ...
    fullfile(section16Dir, 'combined_levels.tif'));

% Create combination visualization
fig16 = figure('Name', 'Section 16: Level Combination', 'Visible', 'off');
subplot(2,3,1); imagesc(image3D_inside_final(:,:,round(end/2))); 
colormap(gray); axis equal; title('Level 1: Inside'); colorbar;
subplot(2,3,2); imagesc(image3D_cell_normalized(:,:,round(end/2))); 
colormap(gray); axis equal; title('Level 2: Cell'); colorbar;
subplot(2,3,3); imagesc(image3D_surface_final(:,:,round(end/2))); 
colormap(gray); axis equal; title('Level 3: Surface'); colorbar;
subplot(2,3,4); imagesc(max(image3D_inside_final(:,:,round(end/2)), image3D_cell_normalized(:,:,round(end/2)))); 
colormap(gray); axis equal; title('Level 1 + 2'); colorbar;
subplot(2,3,5); imagesc(max(image3D_cell_normalized(:,:,round(end/2)), image3D_surface_final(:,:,round(end/2)))); 
colormap(gray); axis equal; title('Level 2 + 3'); colorbar;
subplot(2,3,6); imagesc(combinedImage(:,:,round(end/2))); 
colormap(gray); axis equal; title('All Combined'); colorbar;
saveas(fig16, fullfile(section16Dir, 'level_combination_process.png'));
close(fig16);

% SECTION 17: POST-PROCESSING
fprintf('--- Section 17: Post-processing ---\n');
% WHAT THIS DOES:
% Cleans up the combined image by filling holes, removing negative values,
% and eliminating small disconnected components that could cause mesh
% generation problems.

% Fill holes in the combined image
combinedImage_filled = imfill(combinedImage);
fprintf('  Holes filled in combined image\n');

% Remove negative values (set to zero)
combinedImage_positive = combinedImage_filled;
combinedImage_positive(combinedImage_positive < 0) = 0;
negativePixels = sum(combinedImage_filled(:) < 0);
fprintf('  Removed %d negative pixels\n', negativePixels);

% Remove disconnected components (keep only largest)
level = 0.999;
combinedImage_final = removeDisconectedComponents(combinedImage_positive, level);
fprintf('  Disconnected components removed (level = %.3f)\n', level);

section17Dir = fullfile(finalDir, 'Section_17');
save3DImage(uint16((combinedImage_final - min(combinedImage_final(:))) / ...
    (max(combinedImage_final(:)) - min(combinedImage_final(:))) * 65535), ...
    fullfile(section17Dir, 'processed_final.tif'));

% Create post-processing visualization
fig17 = figure('Name', 'Section 17: Post-processing', 'Visible', 'off');
subplot(2,2,1); imagesc(combinedImage(:,:,round(end/2))); 
colormap(gray); axis equal; title('Combined'); colorbar;
subplot(2,2,2); imagesc(combinedImage_filled(:,:,round(end/2))); 
colormap(gray); axis equal; title('Holes Filled'); colorbar;
subplot(2,2,3); imagesc(combinedImage_positive(:,:,round(end/2))); 
colormap(gray); axis equal; title('Negatives Removed'); colorbar;
subplot(2,2,4); imagesc(combinedImage_final(:,:,round(end/2))); 
colormap(gray); axis equal; title('Final Processed'); colorbar;
saveas(fig17, fullfile(section17Dir, 'postprocessing_steps.png'));
close(fig17);

% SECTION 18: CREATE FINAL MASK FOR BLEB3D SOFTWARE
fprintf('--- Section 18: Creating Final Mask ---\n');
% WHAT THIS DOES:
% Creates the final segmentation mask that can be used directly with
% the bleb3d software for mesh generation and morphological analysis.

% Create binary mask at the specified level
finalMask = combinedImage_final > level;
fprintf('  Final binary mask created\n');
fprintf('  Mask coverage: %.1f%% of volume\n', 100*sum(finalMask(:))/numel(finalMask));

section18Dir = fullfile(finalDir, 'Section_18');

% Save final mask in the main directory (ready for bleb3d software)
save3DImage(uint8(finalMask * 255), fullfile(mainOutputDir, 'final_segmentation_mask.tif'));
save3DImage(uint8(finalMask * 255), fullfile(section18Dir, 'final_mask.tif'));

% Also save the intensity version
save3DImage(uint16((combinedImage_final - min(combinedImage_final(:))) / ...
    (max(combinedImage_final(:)) - min(combinedImage_final(:))) * 65535), ...
    fullfile(mainOutputDir, 'final_segmentation_intensity.tif'));

% Create final summary visualization
fig18 = figure('Name', 'Section 18: Final Results', 'Visible', 'off');
subplot(1,4,1); imagesc(image3D_original(:,:,round(end/2))); 
colormap(gray); axis equal; axis off; title('Original');
subplot(1,4,2); imagesc(image3D_inside_final(:,:,round(end/2))); 
colormap(gray); axis equal; axis off; title('Inside Level');
subplot(1,4,3); imagesc(image3D_surface_final(:,:,round(end/2))); 
colormap(gray); axis equal; axis off; title('Surface Level');
subplot(1,4,4); imagesc(finalMask(:,:,round(end/2))); 
colormap(gray); axis equal; axis off; title('Final Mask');
saveas(fig18, fullfile(section18Dir, 'final_results_summary.png'));
saveas(fig18, fullfile(mainOutputDir, 'processing_summary.png'));
close(fig18);

fprintf('Final mask created and saved to main directory!\n\n');

%% ========================================================================
%  SAVE MOVIEDATA OBJECT AND PARAMETER LOG
%% ========================================================================

fprintf('=== SAVING MOVIEDATA AND PARAMETERS ===\n');

% Save MovieData object for bleb3d compatibility
save(fullfile(mainOutputDir, 'MovieData.mat'), 'MD');
fprintf('MovieData object saved to: %s\n', fullfile(mainOutputDir, 'MovieData.mat'));

% Create comprehensive parameter log
parameterFile = fullfile(mainOutputDir, 'processing_parameters.txt');
fid = fopen(parameterFile, 'w');
fprintf(fid, 'ThreeLevelSegmentation3D Processing Parameters\n');
fprintf(fid, '=============================================\n\n');
fprintf(fid, 'Input Information:\n');
fprintf(fid, '  Image Directory: %s\n', imageDirectory);
fprintf(fid, '  Output Directory: %s\n', mainOutputDir);
fprintf(fid, '  Original Image Size: %d x %d x %d\n', size(image3D_original));
fprintf(fid, '  Final Image Size: %d x %d x %d\n', size(combinedImage_final));
fprintf(fid, '\nImaging Parameters:\n');
fprintf(fid, '  Pixel Size XY: %.1f nm\n', pixelSizeXY);
fprintf(fid, '  Pixel Size Z: %.1f nm\n', pixelSizeZ);
fprintf(fid, '  Time Interval: %.1f seconds\n', timeInterval);
fprintf(fid, '  Z Scaling Factor: %.2f\n', pixelSizeZ/pixelSizeXY);
fprintf(fid, '\nProcessing Parameters:\n');
fprintf(fid, '  Surface Filter Scales: [%.1f, %.1f, %.1f]\n', scales);
fprintf(fid, '  Surface Threshold Multiplier: %.1f\n', nSTDsurface);
fprintf(fid, '  Inside Gamma Correction: %.2f\n', insideGamma);
fprintf(fid, '  Inside Blur Sigma: %.1f\n', insideBlur);
fprintf(fid, '  Inside Dilation Radius: %.1f\n', insideDilateRadius);
fprintf(fid, '  Inside Erosion Radius: %.1f\n', insideErodeRadius);
fprintf(fid, '\nComputed Thresholds:\n');
fprintf(fid, '  Inside Otsu Threshold: %.4f\n', insideThreshValue);
fprintf(fid, '  Cell Otsu Threshold: %.4f\n', cellThreshValue);
fprintf(fid, '  Surface Threshold: %.4f\n', surfThresh);
fprintf(fid, '  Final Isosurface Level: %.3f\n', level);
fprintf(fid, '\nFinal Results:\n');
fprintf(fid, '  Processing Date: %s\n', datestr(now));
fprintf(fid, '  Final Mask Coverage: %.1f%% of volume\n', 100*sum(finalMask(:))/numel(finalMask));
fprintf(fid, '  Number of Processing Sections: 18\n');
fprintf(fid, '  Total Files Generated: %d TIFF files + %d PNG figures\n', 18, 18);
fclose(fid);

fprintf('Parameter log saved to: %s\n', parameterFile);

%% ========================================================================
%  COMPLETION SUMMARY
%% ========================================================================

fprintf('\n=== PREPROCESSING COMPLETED SUCCESSFULLY! ===\n');
fprintf('\nSummary of Generated Files:\n');
fprintf('  Main Output Directory: %s\n', mainOutputDir);
fprintf('  Final Segmentation Mask: final_segmentation_mask.tif\n');
fprintf('  Final Intensity Image: final_segmentation_intensity.tif\n');
fprintf('  MovieData Object: MovieData.mat\n');
fprintf('  Parameter Log: processing_parameters.txt\n');
fprintf('  Processing Summary Figure: processing_summary.png\n');
fprintf('\nFolder Structure Created:\n');
fprintf('  Level1_Inside_Processing/ (Sections 01-11)\n');
fprintf('  Level2_Cell_Processing/ (Sections 12-13)\n');
fprintf('  Level3_Surface_Processing/ (Sections 14-15)\n');
fprintf('  Final_Combination/ (Sections 16-18)\n');
fprintf('\nTotal Files Generated:\n');
fprintf('  - 18 intermediate TIFF volumes\n');
fprintf('  - 18 visualization PNG figures\n');
fprintf('  - 2 final mask files (binary + intensity)\n');
fprintf('  - 1 MovieData object\n');
fprintf('  - 1 parameter log file\n');
fprintf('  - 1 summary figure\n');

fprintf('\n=== NEXT STEPS ===\n');
fprintf('1. Examine the processing_summary.png to verify results\n');
fprintf('2. Check individual section folders for detailed analysis\n');
fprintf('3. Use final_segmentation_mask.tif with bleb3d software\n');
fprintf('4. Adjust parameters in script header if needed and re-run\n');

fprintf('\n=== SCRIPT EXECUTION COMPLETE ===\n');

end

%% ========================================================================
%  USER INSTRUCTIONS FOR RUNNING THIS SCRIPT
%% ========================================================================
%
%  HOW TO USE THIS SCRIPT:
%
%  1. MODIFY INPUT PATHS (Lines 48-52):
%     - Set 'imageDirectory' to your 3D TIFF image folder
%     - Set 'outputBaseDirectory' to where you want results saved
%     - Update pixel sizes for your microscope settings
%
%  2. OPTIONAL: ADJUST PROCESSING PARAMETERS (Lines 57-63):
%     - 'scales': Membrane detection scales [1.5, 2, 4]
%     - 'nSTDsurface': Surface sensitivity (2 = moderate)
%     - 'insideGamma': Interior enhancement (0.6 = moderate)
%     - Other morphological parameters as needed
%
%  3. RUN THE SCRIPT:
%     >> ThreeLevelSegmentation_Preprocessing_Script()
%
%  4. EXAMINE RESULTS:
%     - Check 'processing_summary.png' for overview
%     - Browse section folders for detailed analysis
%     - Use 'final_segmentation_mask.tif' for mesh generation
%
%  5. TROUBLESHOOTING:
%     - If mask looks wrong, adjust parameters and re-run
%     - Check individual sections to identify problem areas
%     - Consult parameter log file for threshold values used
%
%  FOR BEGINNERS:
%  - Start with default parameters
%  - Focus on examining the visualization figures
%  - The script includes detailed comments explaining each step
%  - Each section can be run independently for testing
%
%  ADVANCED USERS:
%  - Modify parameters based on your specific cell type
%  - Add custom visualization or analysis steps
%  - Use intermediate results for parameter optimization
%
%% ========================================================================