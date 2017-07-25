function STResNet_stage2(varargin)

if ~isempty(gcp('nocreate'))
    delete(gcp)
end

opts.train.gpus = [ 1 ] ;

% addpath('../network_surgery');
opts = cnn_setup_environment(opts);

% opts.dataSet = 'hmdb51';
opts.dataSet = 'ucf101';

opts.dropOutRatio = NaN;
opts.backpropFuseFrom = 1;
opts.nSplit =  1 ;
opts.addPool3D = 1 ;
opts.singleSoftMax = 0;
opts.nFrames = 11 ;
% opts.train.backpropDepth = 'pool5';
opts.train.learningRate =  1*[ 1e-4*ones(1,1) 1e-5*ones(1,1)]  ;
opts.train.augmentation = 'f25noCtr';

opts.train.epochFactor = 1 ;

opts.train.batchSize = 128  ;
opts.train.numSubBatches = 32 ;
opts.train.cheapResize = 0 ;
opts.poolMethod = 'max';
opts.poolSz=5 ;
opts.poolStride=2 ;

model = ['ST-ResNet50-final-split=' num2str(opts.nSplit)];

opts.train.numSubBatches =  ceil(opts.train.numSubBatches / max(numel(opts.train.gpus),1));

opts.train.memoryMapFile = fullfile(tempdir, 'ramdisk', ['matconvnet' num2str(opts.nSplit ) '.bin']) ;
opts.dataDir = fullfile(opts.dataPath, opts.dataSet) ;
opts.splitDir = 'ucf101_splits'; nClasses = 101;
opts.imdbPath = fullfile(opts.dataDir, [opts.dataSet '_split' num2str(opts.nSplit) 'imdb.mat']);
    
opts.model = fullfile(opts.modelPath, [opts.dataSet '-ST-ResNet50-split=' num2str(opts.nSplit) '.mat']) ;

if strcmp(opts.dataSet, 'hmdb51')
  opts.splitDir = 'hmdb51_splits'; nClasses = 51;
  opts.flowDir = strrep(opts.flowDir, 'ucf101','hmdb51');
  opts.imdbPath = fullfile(opts.dataDir, ['hmdb_split' num2str(opts.nSplit) 'imdb.mat']);
end

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
opts.train.nFramesPerVid = 1;

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
  ['http://ftp.tugraz.at/pub/feichtenhofer/st-res/stage1/' baseModel '.mat'], ...
    opts.model) ;
end
net = load(opts.model) ;

if isfield(net, 'net'), net=net.net;end
net = dagnn.DagNN.loadobj(net);

if opts.addPool3D

%   poolLayers = {'res2a', 'res3a',  'res4a', 'res5a'; };
%   poolLayers = {'pool5'; };
  poolLayers = {'res5c_relu'; };

  for j=1:numel(poolLayers)
    for s = {'spatial', 'temporal'}
      i_pool = find(strcmp({net.layers.name},[poolLayers{j} '_' char(s)]));    
      
      block = dagnn.PoolTime() ;
      block.poolSize = [1 opts.poolSz];  
      block.pad = [0 0 0 0]; 
      block.stride = [1 opts.poolStride];     
      block.method = opts.poolMethod;     
 
      name = [poolLayers{j} '_pool_' char(s)];
      
      disp(['injecting ' name ' as PoolTime'])

      net.addLayerAt(i_pool, name, block, ...
                    [net.layers(i_pool).outputs], {name}) ; 

      % chain input of l that has layer as input
      for l = 1:numel(net.layers)    
          if ~strcmp(net.layers(l).name, name)
            sel = find(strcmp(net.layers(l).inputs, net.layers(i_pool).outputs{1})) ;
            if any(sel)
                net.layers(l).inputs{sel} = name;
            end;   
          end
      end

    end
  end
end % add pool3d

if opts.addPool3D
  opts.train.augmentation = 'f25noCtr';
  opts.train.frameSample = 'temporalStrideRandom';
  opts.train.nFramesPerVid = opts.nFrames * 1; 
  opts.train.temporalStride = 1:15;  
  opts.train.valmode = 'temporalStrideRandom';
  opts.train.numValFrames = 25 ;
  opts.train.saveAllPredScores = 1 ;
  opts.train.denseEval = 1;
  opts.train.temporalFullConvTest = 1;
end  
 
net.meta.normalization.rgbVariance = [];
opts.train.train = find(ismember(imdb.images.set, [1])) ;
opts.train.train = repmat(opts.train.train,1,opts.train.epochFactor);

zero_drs =  find(arrayfun(@(x) isa(x.block,'dagnn.DropOut') && x.block.rate == 0, net.layers)) ;
net.removeLayer({net.layers(zero_drs).name});

if ~isnan(opts.dropOutRatio)
  dr_layers = find(arrayfun(@(x) isa(x.block,'dagnn.DropOut'), net.layers)) ;
  if opts.dropOutRatio > 0
    net.layers(dr_layers).block.rate = opts.dropOutRatio;
  else
    net.removeLayer({net.layers(dr_layers).name});
  end
end

if opts.singleSoftMax
  pred_layers = [];
  for l=1:numel(net.layers)
    if isempty( net.layers(l).params ), continue; end;
    if size(net.params(net.getParamIndex(net.layers(l).params{1})).value,4) == nClasses || ...
        size(net.params(net.getParamIndex(net.layers(l).params{1})).value,5) == nClasses % 3D FC layer
          pred_layers = [pred_layers l];
          net.vars(net.layers(l).outputIndexes).precious = 1;
    end
  end
  pred_layers = fliplr(pred_layers) ; % remove the spatial layer
  paramsIdx1 = net.getParamIndex(net.layers(pred_layers(1)).params);
  paramsIdx2 = net.getParamIndex(net.layers(pred_layers(2)).params);
  for p = 1:numel(paramsIdx1)
    sz = size(net.params(paramsIdx1(p)).value);
    if numel(sz) > 2
      net.params(paramsIdx1(p)).value = cat(3,net.params(paramsIdx1(p)).value, net.params(paramsIdx2(p)).value);
    else
      net.params(paramsIdx1(p)).value = net.params(paramsIdx1(p)).value + net.params(paramsIdx2(p)).value;
    end
  end
  block = dagnn.Concat() ;
  newName = ['singleSoftMaxConcat']; 
  net.addLayer(newName, block, ...
               [net.layers(pred_layers).inputs], ...
               newName) ;   
  net.layers(pred_layers(1)).inputs =    newName ; 
  % remove layers of the other prediction
  for l = numel(net.layers):-1:1
    for f = net.layers(pred_layers(2)).outputs
       sel = find(strcmp(f, net.layers(l).inputs )) ;
       if ~isempty(sel)
         fprintf('removing ayer %s \n', net.layers(l).name);
         net.removeLayer({net.layers(l).name});
       end
    end
  end
  net.removeLayer({net.layers(pred_layers(2)).name});
end

net.layers(~cellfun('isempty', strfind({net.layers(:).name}, 'err'))) = [] ;

net.rebuild() ;

opts.train.derOutputs = {} ;
for l=1:numel(net.layers)
  if isa(net.layers(l).block, 'dagnn.Loss') && isempty(strfind(net.layers(l).block.loss, 'err'))
    if opts.backpropFuseFrom || ~isempty(strfind(net.layers(l).name, opts.train.fuseInto ))
      fprintf('setting derivative for layer %s \n', net.layers(l).name);
      opts.train.derOutputs = [opts.train.derOutputs, net.layers(l).outputs, {1}] ;
    end
    net.addLayer(['err1_' net.layers(l).name(end-7:end) ], dagnn.Loss('loss', 'classerror'), ...
             net.layers(l).inputs, 'error') ;  
  end
end

net.conserveMemory = 1 ;
fn = getBatchWrapper_rgbflow(net.meta.normalization, opts.numFetchThreads, opts.train) ;
[info] = cnn_train_dag(net, imdb, fn, opts.train) ;