function trainFiloStructures_Final()

%% Set Directories
analysisDirectory = '/Volumes/T7/180228_osna/1/GPUdecon/Results';
motifModelDirectory = '/Volumes/T7/180228_osna/1/GPUdecon/Models';

%% FIX: Handle macOS metadata file
macosFile = fullfile(analysisDirectory, '._movieData.mat');
if exist(macosFile, 'file')
    tempName = [macosFile '.temp'];
    movefile(macosFile, tempName);
end

%% Setup
% For single cell analysis - create proper cell structure
p.cellsList{1} = '';  % FIXED: Empty string instead of 'Cell1'

% Training parameters
p.mainInDirectory = analysisDirectory;
p.clickMode = 'clickOnCertain';
p.nameOfClicker = 'Philipp';
p.classNames = {'filopodia'};              % Or {'blebs'}, {'branches'}, etc.
p.numClasses = 2;
p.mode = 'restart';
p.surfaceMode = 'surfaceSegment';
p.framesPerCell = 1;

%% Collect training data by clicking
clickOnProtrusions(p)

%% FIX: Restore macOS metadata file
if exist([macosFile '.temp'], 'file')
    movefile([macosFile '.temp'], macosFile);
end

%% Train and validate classifier
% Setup paths for training
p.MDsPathList{1} = p.cellsList{1};  
p.clicksPathList{1} = fullfile(p.MDsPathList{1}, 'TrainingData', p.nameOfClicker);

p.saveNameModel = 'FiloClassifier';
p.saveNameCells = 'testFibroCell';
p.mainDirectory = analysisDirectory;
p.saveDirectory = motifModelDirectory;

% Train the classifier
disp('Training the classifier')
trainProtrusionsClassifier(p);

% Validate the classifier
validateBlebClassifier(p);  % Works for any motif, not just blebs

end