function STMuNet_test(varargin)

if ~isempty(gcp('nocreate'))
    delete(gcp)
end

opts.train.gpus = [ 1 ] ;

opts = cnn_setup_environment(opts);
opts.nSplit =  1 ;

% opts.dataSet = 'hmdb51';
opts.dataSet = 'ucf101';

model = ['ST-MulNet-img50-flow50-final-split=' num2str(opts.nSplit)];
model = ['ST-MulNet-img50-flow152-split=' num2str(opts.nSplit)];

opts.train.memoryMapFile = fullfile(tempdir, 'ramdisk', ['matconvnet' num2str(opts.nSplit ) '.bin']) ;
opts.dataDir = fullfile(opts.dataPath, opts.dataSet) ;
opts.splitDir = 'ucf101_splits'; nClasses = 101;
opts.imdbPath = fullfile(opts.dataDir, [opts.dataSet '_split' num2str(opts.nSplit) 'imdb.mat']);
opts.model = fullfile(opts.modelPath, [opts.dataSet model '.mat']) ;
opts.expDir = fullfile(opts.dataDir, [opts.dataSet '-' model]) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.train.saveAllPredScores = 1;
opts.train.denseEval = 1;
opts.train.plotDiagnostics = 0 ;
opts.train.continue = 1 ;
opts.train.prefetch = 1 ;
opts.train.expDir = opts.expDir ;
opts.train.numAugments = 1;
opts.train.frameSample = 'random';

opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%                                                   Database initialization
% -------------------------------------------------------------------------
if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
  imdb.flowDir = opts.flowDir;
else
  imdb = cnn_setup_data(opts) ;
  save(opts.imdbPath, '-struct', 'imdb', '-v6') ;
end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------
if ~exist(opts.model,'file')
  [~, baseModel] = fileparts(opts.model);
  fprintf('Downloading base model file: %s ...\n', baseModel);
  mkdir(fileparts(opts.model)) ;
  urlwrite(...
  ['http://ftp.tugraz.at/pub/feichtenhofer/st-mul/final/' baseModel '.mat'], ...
    opts.model) ;
end
net = load(opts.model) ;

if isfield(net, 'net'), net=net.net;end
net = dagnn.DagNN.loadobj(net);

opts.train.augmentation = 'f25noCtr';
opts.train.frameSample = 'temporalStrideRandom';
opts.train.nFramesPerVid = 1; 
opts.train.temporalStride = 1:15;  
opts.train.valmode = 'temporalStrideRandom';
opts.train.numValFrames = 25 ;
opts.train.saveAllPredScores = 1 ;
opts.train.denseEval = 1;
opts.train.temporalFullConvTest = 1;
 
opts.train.train = NaN;

net.rebuild() ;

net.conserveMemory = 1 ;
fn = getBatchWrapper_rgbflow(net.meta.normalization, opts.numFetchThreads, opts.train) ;
[info] = cnn_train_dag(net, imdb, fn, opts.train) ;