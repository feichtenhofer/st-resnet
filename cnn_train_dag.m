function [net,stats] = cnn_train_dag_mcn2(net, imdb, getBatch, varargin)
%CNN_TRAIN_DAG Demonstrates training a CNN using the DagNN wrapper
%    CNN_TRAIN_DAG() is similar to CNN_TRAIN(), but works with
%    the DagNN wrapper instead of the SimpleNN wrapper.

% Copyright (C) 2014-16 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
addpath(fullfile(vl_rootnn, 'examples'));

opts.expDir = fullfile('data','exp') ;
opts.continue = true ;
opts.batchSize = 256 ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.gpus = [] ;
opts.prefetch = false ;
opts.epochSize = inf;
opts.numEpochs = 300 ;
opts.learningRate = 0.001 ;
opts.weightDecay = 0.0005 ;

opts.solver = [] ;  % Empty array means use the default SGD solver
[opts, varargin] = vl_argparse(opts, varargin) ;
if ~isempty(opts.solver)
  assert(isa(opts.solver, 'function_handle') && nargout(opts.solver) == 2,...
    'Invalid solver; expected a function handle with two outputs.') ;
  % Call without input arguments, to get default options
  opts.solverOpts = opts.solver() ;
end

opts.momentum = 0.9 ;
opts.saveSolverState = true ;
opts.nesterovUpdate = false ;
opts.randomSeed = 0 ;
opts.profile = false ;

opts.parameterServer.method = 'tmove'; %'tmove' or 'mmap';
opts.parameterServer.prefix = ['matconvnet' num2str(feature('getpid'))] ;
opts.parameterServer.memoryMapFile = fullfile(tempdir, 'ramdisk', opts.parameterServer.prefix) ;

opts.derOutputs = {'objective', 1} ;
opts.extractStatsFn = @extractStats ;
opts.plotStatistics = true;
opts.postEpochFn = [] ;  % postEpochFn(net,params,state) called after each epoch; can return a new learning rate, 0 to stop, [] for no change

opts.valmode = '30samples';
opts.temporalStride = 1;
opts.backpropDepth = [];
opts.numValFrames = 3;
opts.nFramesPerVid = 5;
opts.saveAllPredScores = false;
opts.denseEval = 0;
opts.cudnnWorkspaceLimit = [];
opts.plotDiagnostics = false;
opts.augmentation = '';
opts.temporalFullConvTest = true;
[opts, varargin] = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isscalar(opts.train) && isnumeric(opts.train) && isnan(opts.train)
  opts.train = [] ;
end
if isscalar(opts.val) && isnumeric(opts.val) && isnan(opts.val)
  opts.val = [] ;
end

% -------------------------------------------------------------------------
%                                                            Initialization
% -------------------------------------------------------------------------

evaluateMode = isempty(opts.train) ;
if ~evaluateMode
  if isempty(opts.derOutputs)
    error('DEROUTPUTS must be specified when training.\n') ;
  end
end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;

start = opts.continue * findLastCheckpoint(opts.expDir) ;
if start >= 1
  fprintf('%s: resuming by loading epoch %d\n', mfilename, start) ;
  [net, state, stats] = loadState(modelPath(start)) ;
else
  state = [] ;
end

for epoch=start+1:opts.numEpochs

  % Set the random seed based on the epoch and opts.randomSeed.
  % This is important for reproducibility, including when training
  % is restarted from a checkpoint.

  rng(epoch + opts.randomSeed) ;
  prepareGPUs(opts, epoch == start+1) ;

  % Train for one epoch.
  params = opts ;
  params.epoch = epoch ;
  params.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
  params.train = opts.train(randperm(numel(opts.train))) ; % shuffle
  params.train = params.train(1:min(opts.epochSize, numel(opts.train)));
  params.val = opts.val ;
  params.imdb = imdb ;
  params.getBatch = getBatch ;
  if iscell(opts.backpropDepth)
    if epoch <= numel(opts.backpropDepth)
      params.backpropDepth = opts.backpropDepth{epoch};
    else
      params.backpropDepth = [];
    end
  else
    params.backpropDepth = opts.backpropDepth;
  end
  
  if numel(opts.gpus) <= 1
    [net, state] = processEpoch(net, state, params, 'train') ;
    [net, state] = processEpoch(net, state, params, 'val') ;
    if ~evaluateMode
      saveState(modelPath(epoch), net, state) ;
    end
    lastStats = state.stats ;
  else
    assert(isempty(params.backpropDepth), 'Cannot use backpropDepth in multi GPU mode.')
    spmd
      [net, state] = processEpoch(net, state, params, 'train') ;
      [net, state] = processEpoch(net, state, params, 'val') ;
      if labindex == 1 && ~evaluateMode
        saveState(modelPath(epoch), net, state) ;
      end
      lastStats = state.stats ;
    end
    lastStats = accumulateStats(lastStats) ;
  end

  stats.train(epoch) = lastStats.train ;
  stats.val(epoch) = lastStats.val ;
  clear lastStats ;
  saveStats(modelPath(epoch), stats, opts) ;

  if opts.plotStatistics
    switchFigure(1) ; clf ;
    values = [] ;   values_loss = [] ;
    leg = {} ;   leg_loss = {} ;
    for s = {'train', 'val'}
      s = char(s) ;
      for f = setdiff(fieldnames(stats.(s))', {'num', 'time','scores', 'allScores'})
        f = char(f) ;
        if isempty(strfind(f,'err'))
          leg_loss{end+1} = sprintf('%s (%s)', f, s) ;
          tmp = [stats.(s).(f)] ;         
          values_loss(end+1,:) = tmp(1,:)' ;
        else
          leg{end+1} = sprintf('%s (%s)', f, s) ;
          tmp = [stats.(s).(f)] ;        
          values(end+1,:) = tmp(1,:)' ;
        end
        tmp = [stats.(s).(f)];
        fprintf('%s (%s):%.3f\n', f, s, tmp(end))
      end
    end

    if ~isempty(values_loss)
      subplot(1,2,1) ; plot(1:epoch, values_loss','o-') ; 
      legend(leg_loss{:},'Location', 'northoutside'); xlabel('epoch') ; ylabel('objective') ;
      subplot(1,2,2) ; plot(1:epoch, values','o-') ; ylim([0 1])
      legend(leg{:},'Location', 'northoutside') ; xlabel('epoch') ; ylabel('error') ;
      grid on ;
      drawnow ;
      print(1, modelFigPath, '-dpdf') ;
    end
  end  
  
  if ~isempty(opts.postEpochFn)
    if nargout(opts.postEpochFn) == 0
      opts.postEpochFn(net, params, state) ;
    else
      lr = opts.postEpochFn(net, params, state) ;
      if ~isempty(lr), opts.learningRate = lr; end
      if opts.learningRate == 0, break; end
    end
  end
end

% With multiple GPUs, return one copy
if isa(net, 'Composite'), net = net{1} ; end

% -------------------------------------------------------------------------
function [net, state] = processEpoch(net, state, params, mode)
% -------------------------------------------------------------------------
% Note that net is not strictly needed as an output argument as net
% is a handle class. However, this fixes some aliasing issue in the
% spmd caller.

% initialize with momentum 0
if isfield(state, 'momentum'), state.solverState = state.momentum; state.momentum = []; end
if isempty(state) || isempty(state.solverState)
  state.solverState = cell(1, numel(net.params)) ;
  state.solverState(:) = {0} ;
end

% move CNN  to GPU as needed
numGpus = numel(params.gpus) ;
if numGpus >= 1
  net.move('gpu') ;
  for i = 1:numel(state.solverState)
    s = state.solverState{i} ;
    if isnumeric(s)
      state.solverState{i} = gpuArray(s) ;
    elseif isstruct(s)
      state.solverState{i} = structfun(@gpuArray, s, 'UniformOutput', false) ;
    end
  end
end
if numGpus > 1
  parserv = ParameterServer(params.parameterServer) ;
  net.setParameterServer(parserv) ;
else
  parserv = [] ;
end

% profile
if params.profile
  if numGpus <= 1
    profile clear ;
    profile on ;
  else
    mpiprofile reset ;
    mpiprofile on ;
  end
end

num = 0 ;
epoch = params.epoch ;
subset = params.(mode) ;
adjustTime = 0 ;
moreopts = [];

stats.num = 0 ; % return something even if subset = []
stats.time = 0 ;

stats.scores = [] ;
stats2.err1 = 0;
stats2.err5 = 0;
net.backpropDepth = params.backpropDepth;

if ~strcmp(mode,'train')
  net.mode = 'test';
  dataset =  ceil(params.imdb.images.set(subset(1))/2); 
  nClasses = numel(params.imdb.classes.name);
  if nClasses < 5
    nClasses = numel(params.imdb.classes.name{dataset});
  end
  stats2.scores = zeros(nClasses, numel(subset));

  moreopts.frameSample = 'uniformly';
  moreopts.augmentation = 'uniform';
  moreopts.keepFramesDim = true; % make getBatch output 5 dimensional 

  if strcmp(params.valmode,'30samples')
    % sample less frames and crops:
    moreopts.numAugments = 6;
    moreopts.nFramesPerVid = 5;
  elseif strcmp(params.valmode,'centreSamplesFast')
    % sample less frames and crops:
    moreopts.numAugments = 2;
    moreopts.nFramesPerVid = 3;
  elseif strcmp(params.valmode,'250samples') , 
    moreopts.numAugments = 10;
    moreopts.nFramesPerVid = 25;
  elseif strcmp(params.valmode,'dense')
    moreopts.augmentation = 'none';
    moreopts.numAugments = 0;
    moreopts.nFramesPerVid = 25;
    moreopts.keepFramesDim = true; 
    params.batchSize = numlabs;
    params.numSubBatches = numlabs;
  elseif strcmp(params.valmode,'temporalStrideRandom')
    moreopts.nFrameStack = params.nFramesPerVid; 
    moreopts.temporalStride = ceil(median(params.temporalStride));
    moreopts.temporalStride = max(params.temporalStride);
    params.batchSize =  32*numlabs ;
    params.numSubBatches = params.batchSize; % has to be
    moreopts.nFramesPerVid = params.numValFrames;
  end
  
  if params.denseEval 
    moreopts.augmentation = 'none';
    moreopts.numAugments = 2;
  end
  if params.temporalFullConvTest
    params.nFramesPerVid = params.numValFrames;
  end
  pred_layers = [];
  for l=1:numel(net.layers)
    if isempty( net.layers(l).params ), continue; end;
    if size(net.params(net.getParamIndex(net.layers(l).params{1})).value,4) == nClasses || ...
        size(net.params(net.getParamIndex(net.layers(l).params{1})).value,5) == nClasses % 3D FC layer
          pred_layers = [pred_layers net.layers(l).outputIndexes];
          net.vars(net.layers(l).outputIndexes).precious = 1;
    end
    if isa(net.layers(l).block, 'dagnn.LSTM')
      pred_layers = [pred_layers net.layers(l).outputIndexes(1)];
      net.vars(net.layers(l).outputIndexes(1)).precious = 1 ;
    end
  end

  if params.saveAllPredScores
    stats2.allScores = zeros(numel(pred_layers),moreopts.numAugments* moreopts.nFramesPerVid/params.nFramesPerVid, nClasses,  numel(subset));
  end
else
  net.mode = 'normal';
  moreopts = [];
end


start = tic ;
for t=1:params.batchSize:numel(subset)
  fprintf('%s: epoch %02d: %3d/%3d:', mode, epoch, ...
          fix((t-1)/params.batchSize)+1, ceil(numel(subset)/params.batchSize)) ;
  batchSize = min(params.batchSize, numel(subset) - t + 1) ;

  for s=1:params.numSubBatches
    % get this image batch and prefetch the next
    batchStart = t + (labindex-1) + (s-1) * numlabs ;
    batchEnd = min(t+params.batchSize-1, numel(subset)) ;
    batch = subset(batchStart : params.numSubBatches * numlabs : batchEnd) ;
    num = num + numel(batch) ;
    if numel(batch) == 0, continue ; end

    inputs = params.getBatch(params.imdb, batch, moreopts) ;

    moreopts.frameList = [];
    if strcmp(net.mode, 'test') && strcmp(params.valmode,'temporalStrideRandom')
      for i = 2:4:numel(inputs)
        sz = size(inputs{i});
        inputs{i} = gather(inputs{i});
        nFramesPerVid = sz(5)/moreopts.numAugments;
        chunks = ceil(nFramesPerVid / params.nFramesPerVid); 
        inputs{i} = reshape(inputs{i}, sz(1), sz(2), sz(3),   [], params.nFramesPerVid);
        inputs{i} = permute(inputs{i} , [1 2 3 5 4]);
      end
    end
    net.meta.curNumFrames = repmat(size(inputs{2},4) / numel(inputs{4}),1,numel(net.layers)); % nFrames = instances/labels
        
    net.meta.curBatchSize = numel(batch);
    inputs{end+1} = 'inputSet'; inputs{end+1} = ceil(params.imdb.images.set(batch)/2); % dataset

    if params.prefetch
      if s == params.numSubBatches
        batchStartNext = t + (labindex-1) + params.batchSize ;
        batchEndNext = min(t+2*params.batchSize-1, numel(subset)) ;
      else
        batchStartNext = batchStart + numlabs ; batchEndNext = batchEnd;
      end
      nextBatch = subset(batchStartNext : params.numSubBatches * numlabs : batchEndNext) ;
      if ~isempty(nextBatch)
        moreopts.frameList = params.getBatch(params.imdb, nextBatch, moreopts) ;
      else 
        moreopts.frameList = NaN ;
      end
    end
    if ndims(inputs{2})>4  % average over frames
      dataset = inputs{end}(1);
      nClasses = numel(params.imdb.classes.name);
      if nClasses < 5
        nClasses = numel(params.imdb.classes.name{dataset});
      end


      frame_predictions = cell(numel(pred_layers),size(inputs{2},5));
      
      for fr = 1:size(inputs{2},5)
        frame_inputs = inputs; 
        net.meta.curNumFrames = repmat(size(inputs{2},4) / numel(inputs{4}),1,numel(net.layers)); % nFrames = instances/labels
        for i = 2:4:numel(inputs)
          if size(frame_inputs{i},5) > 1
            frame_inputs{i}=frame_inputs{i}(:,:,:,:,fr);
          end
        end
        if strcmp(mode, 'train')
          net.accumulateParamDers = (s ~= 1) ;
          net.eval(frame_inputs, derOutputs) ;
        else
          net.eval(frame_inputs) ;
        end   
        [frame_predictions{:,fr}] = deal(net.vars(pred_layers).value) ;
      end
      
      
      tmp = [];
      for k = 1:numel(pred_layers)
        frame_predictions(k,:)= cellfun(@(x) mean(mean(x,1),2), frame_predictions(k,:), 'UniformOutput', false);
        if numel(batch) == 1
          frame_predictions(k,:)= cellfun(@(x) mean(x,4), frame_predictions(k,:), 'UniformOutput', false);
        end
        tmp = [tmp; frame_predictions{k,:}];
      end
      frame_predictions = tmp;

      if min(net.meta.curNumFrames) > 1
        frame_predictions = mean(frame_predictions,4);
      end
      if  params.saveAllPredScores
        stats2.allScores(:,:,:,batchStart : params.numSubBatches * numlabs : batchEnd) = gather(frame_predictions);
      end
      % average over time (dim+1) and spatial locations and batches
      frame_predictions = mean(mean(mean(frame_predictions),1),2);
      [err1, err5] = error_multiclass(params, inputs{4}, gather(frame_predictions));
      stats2.err1 = (stats2.err1 + err1);
      stats2.err5 = (stats2.err5 + err5);
    else  % inputs four dimensional  
      if strcmp(mode, 'train')
        net.mode = 'normal' ;
        net.accumulateParamDers = (s ~= 1) ;
        net.eval(inputs, params.derOutputs, 'holdOn', s < params.numSubBatches) ;
      else
        net.mode = 'test' ;
        net.eval(inputs) ;
      end
    end
    if strcmp(mode, 'val') && ndims(inputs{2})>4 
      stats2.scores(:, batchStart : params.numSubBatches * numlabs : batchEnd) = squeeze(gather(frame_predictions));
    end
  end

  % Accumulate gradient.
  if strcmp(mode, 'train')
    if ~isempty(parserv), parserv.sync() ; end
    state = accumulateGradients(net, state, params, batchSize, parserv) ;
  end

  % Get statistics.
  time = toc(start) + adjustTime ;
  batchTime = time - stats.time ;
  stats.num = num ;
  stats.time = time ;
  stats = params.extractStatsFn(stats,net) ;
  if ndims(inputs{2})>4  % average over frames
    for f = fieldnames(stats2)'
        f = char(f) ;  stats.(f) = stats2.(f);
    end 
  end
  currentSpeed = batchSize / batchTime ;
  averageSpeed = (t + batchSize - 1) / time ;
  if t == 3*params.batchSize + 1
    % compensate for the first three iterations, which are outliers
    adjustTime = 4*batchTime - time ;
    stats.time = time + adjustTime ;
  end

  fprintf('lr: %.0e, %.1f (%.1f) Hz',params.learningRate, averageSpeed, currentSpeed) ;
  for f = setdiff(fieldnames(stats)',  {'num', 'time','scores', 'allScores'})
    f = char(f) ;
    if ndims(inputs{2})>4  && any(strcmp(f, {'err1', 'err5'}))
      n = (t + batchSize - 1) / max(1,numlabs) ;
      stats.(f) = stats.(f) / n;
    end
      fprintf(' %s:%.3f', f, stats.(f)) ;
  end
  fprintf('\n') ;
  
  %  debug info
  if params.plotDiagnostics && numGpus <= 1
    figure(2) ; net.diagnose('Vars',1,'Params',1,'Time',1) ; drawnow ;
  end
end

% Save back to state.
state.stats.(mode) = stats ;
if params.profile
  if numGpus <= 1
    state.prof.(mode) = profile('info') ;
    profile off ;
  else
    state.prof.(mode) = mpiprofile('info');
    mpiprofile off ;
  end
end
if ~params.saveSolverState
  state.solverState = [] ;
else
  for i = 1:numel(state.solverState)
    s = state.solverState{i} ;
    if isnumeric(s)
      state.solverState{i} = gather(s) ;
    elseif isstruct(s)
      state.solverState{i} = structfun(@gather, s, 'UniformOutput', false) ;
    end
  end
end

net.reset() ;
net.move('cpu') ;

% -------------------------------------------------------------------------
function state = accumulateGradients(net, state, params, batchSize, parserv)
% -------------------------------------------------------------------------
numGpus = numel(params.gpus) ;
otherGpus = setdiff(1:numGpus, labindex) ;

for p=1:numel(net.params)

  if ~isempty(parserv)
    parDer = parserv.pullWithIndex(p) ;
  else
    parDer = net.params(p).der ;
  end
  if isempty(parDer)
    % fprintf('empty param for: %s\n', net.params(p).name) ;
    continue;
  end
  switch net.params(p).trainMethod

    case 'average' % mainly for batch normalization
      thisLR = net.params(p).learningRate ;
      net.params(p).value = vl_taccum(...
          1 - thisLR, net.params(p).value, ...
          (thisLR/batchSize/net.params(p).fanout),  parDer) ;

    case 'gradient'
      thisDecay = params.weightDecay * net.params(p).weightDecay ;
      thisLR = params.learningRate * net.params(p).learningRate ;

      if thisLR>0 || thisDecay>0
        % Normalize gradient and incorporate weight decay.
        parDer = vl_taccum(1/batchSize, parDer, ...
                           thisDecay, net.params(p).value) ;

        if isempty(params.solver)
          % Default solver is the optimised SGD.
          % Update momentum.
          state.solverState{p} = vl_taccum(...
            params.momentum, state.solverState{p}, ...
            -1, parDer) ;

          % Nesterov update (aka one step ahead).
          if params.nesterovUpdate
            delta = params.momentum * state.solverState{p} - parDer ;
          else
            delta = state.solverState{p} ;
          end

          % Update parameters.
          net.params(p).value = vl_taccum(...
            1,  net.params(p).value, thisLR, delta) ;

        else
          % call solver function to update weights
          [net.params(p).value, state.solverState{p}] = ...
            params.solver(net.params(p).value, state.solverState{p}, ...
            parDer, params.solverOpts, thisLR) ;
        end
      end
    otherwise
      error('Unknown training method ''%s'' for parameter ''%s''.', ...
        net.params(p).trainMethod, ...
        net.params(p).name) ;
  end
end

% -------------------------------------------------------------------------
function stats = accumulateStats(stats_)
% -------------------------------------------------------------------------

for s = {'train', 'val'}
  s = char(s) ;
  total = 0 ;

  % initialize stats stucture with same fields and same order as
  % stats_{1}
  stats__ = stats_{1} ;
  names = fieldnames(stats__.(s))' ;
  values = zeros(1, numel(names)) ;
  fields = cat(1, names, num2cell(values)) ;
  stats.(s) = struct(fields{:}) ;

  for g = 1:numel(stats_)
    stats__ = stats_{g} ;
    num__ = stats__.(s).num ;
    total = total + num__ ;

    for f = setdiff(fieldnames(stats__.(s))', 'num')
      f = char(f) ;
      stats.(s).(f) = stats.(s).(f) + stats__.(s).(f) * num__ ;

      if g == numel(stats_)
        stats.(s).(f) = stats.(s).(f) / total ;
      end
    end
  end
  stats.(s).num = total ;
end

% -------------------------------------------------------------------------
function stats = extractStats(stats, net)
% -------------------------------------------------------------------------
sel = find(cellfun(@(x) isa(x,'dagnn.Loss')||isa(x,'dagnn.Loss'), {net.layers.block})) ;
for i = 1:numel(sel)
  if net.layers(sel(i)).block.ignoreAverage, continue; end
  stats.(net.layers(sel(i)).name) = net.layers(sel(i)).block.average ;
end

% -------------------------------------------------------------------------
function saveState(fileName, net_, state)
% -------------------------------------------------------------------------
net = net_.saveobj() ;
state.stats = [];
save(fileName, 'net', 'state') ;

% -------------------------------------------------------------------------
function saveStats(fileName, stats, opts)
% -------------------------------------------------------------------------
if exist(fileName)
  save(fileName, 'stats', 'opts', '-append') ;
else
  save(fileName, 'stats', 'opts') ;
end

% -------------------------------------------------------------------------
function [net, state, stats] = loadState(fileName)
% -------------------------------------------------------------------------
state = [];
load(fileName, 'net', 'state', 'stats') ;
net = dagnn.DagNN.loadobj(net) ;
if isempty(whos('stats'))
  if isfield(state, 'stats')
    stats = state.stats;
  else
    error('Epoch ''%s'' was only partially saved. Delete this file and try again.', ...
        fileName) ;
  end
end

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;

% -------------------------------------------------------------------------
function switchFigure(n)
% -------------------------------------------------------------------------
if get(0,'CurrentFigure') ~= n
  try
    set(0,'CurrentFigure',n) ;
  catch
    figure(n) ;
  end
end

% -------------------------------------------------------------------------
function clearMex()
% -------------------------------------------------------------------------
clear vl_tmove vl_imreadjpeg ;

% -------------------------------------------------------------------------
function prepareGPUs(opts, cold)
% -------------------------------------------------------------------------
numGpus = numel(opts.gpus) ;
if numGpus > 1
  % check parallel pool integrity as it could have timed out
  pool = gcp('nocreate') ;
  if ~isempty(pool) && pool.NumWorkers ~= numGpus
    delete(pool) ;
  end
  pool = gcp('nocreate') ;
  if isempty(pool)
    parpool('local', numGpus) ;
    cold = true ;
  end

end
if numGpus >= 1 && cold
  fprintf('%s: resetting GPU\n', mfilename)
  clearMex() ;
  if numGpus == 1
    gpuDevice(opts.gpus)
  else
    spmd
      clearMex() ;
      gpuDevice(opts.gpus(labindex))
    end
  end
end

function [err1, err5] = error_multiclass(opts, labels, predictions)
% -------------------------------------------------------------------------
[~,predictions] = sort(predictions, 3, 'descend') ;
error = ~bsxfun(@eq, predictions, reshape(labels, 1, 1, 1, [])) ;
err1 = sum(sum(sum(error(:,:,1,:)))) ;
err5 = sum(sum(sum(min(error(:,:,1:5,:),[],3)))) ;
