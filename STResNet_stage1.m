function STResNet_stage1(varargin)

if ~isempty(gcp('nocreate'))
    delete(gcp)
end

opts.train.gpus = [ 1 ] ;
opts = cnn_setup_environment(opts);

addpath('network_surgery');
opts.dataSet = 'ucf101'; 
opts.inputdim  = [ 224,  224, 20] ;
initMethod = 'sumAB';
opts.dropOutRatio = 0;
opts.dataDir = fullfile(opts.dataPath, opts.dataSet) ;
opts.splitDir = [opts.dataSet '_splits']; 
opts.train.fuseInto = 'spatial'; opts.train.fuseFrom = 'temporal';
opts.train.removeFuseFrom = 0;
opts.backpropFuseFrom = 1;
opts.nSplit = 1 ;
opts.addConv3D = 1 ;
opts.addPool3D  = 0 ; 
opts.doSum = 1 ;
opts.doScale = 0 ;
opts.doBiDirectional = 0 ;
opts.nFrames = 5 ;
opts.singleSoftMax = 0;

opts.train.learningRate =  1*[ 1e-3*ones(1,3) 1e-4*ones(1,3)  1e-5*ones(1,3) 1e-6*ones(1,3)]  ;
opts.train.augmentation = 'multiScaleRegular';
opts.train.augmentation = 'f25noCtr';
opts.train.epochFactor = 10 ;
opts.train.batchSize = 128  ;
opts.train.numSubBatches = 32 ;
opts.train.cheapResize = 0 ;
opts.train.fusionLayer = {'res2a_relu', 'res2a'; 'res3a_relu', 'res3a'; 'res4a_relu', 'res4a'; ...
  'res5a_relu', 'res5a'; };


model = ['ST-ResNet50-split=' num2str(opts.nSplit)];
opts.train.numSubBatches =  ceil(opts.train.numSubBatches / max(numel(opts.train.gpus),1));
opts.train.memoryMapFile = fullfile(tempdir, 'ramdisk', ['matconvnet' num2str(opts.nSplit ) '.bin']) ;

opts.modelA = fullfile(opts.modelPath, [opts.dataSet '-img-resnet50-split' num2str(opts.nSplit) '-dr0.mat']) ;
opts.modelB = fullfile(opts.modelPath, [opts.dataSet '-resnet50-split' num2str(opts.nSplit) '-dr0.8.mat']) ;

opts.expDir = fullfile(opts.dataDir, [opts.dataSet '-' model]) ;

opts.train.startEpoch = 1 ;
opts.train.epochStep = 1 ;
opts.train.numEpochs = 8 ;
[opts, varargin] = vl_argparse(opts, varargin) ;


opts.imdbPath = fullfile(opts.dataDir, [opts.dataSet '_split' num2str(opts.nSplit) 'imdb.mat']);

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
nClasses = length(imdb.classes.name);

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------
for model = {opts.modelA opts.modelB}
  model = model{:};
  if ~exist(model,'file')
    [~, baseModel] = fileparts(model);
    fprintf('Downloading base model file: %s ...\n', baseModel);
    mkdir(fileparts(model)) ;
    urlwrite(...
    ['http://ftp.tugraz.at/pub/feichtenhofer/st-res/ts-base/' baseModel '.mat'], ...
      model) ;
  end
end
netA = load(opts.modelA) ;
netB = load(opts.modelB) ;
if isfield(netA, 'net'), netA=netA.net;end
if isfield(netB, 'net'), netB=netB.net;end

if ~isfield(netA, 'meta')
  netA = update_legacy_net(netA);
  netA = dagnn.DagNN.fromSimpleNN(netA) ;
  netA = netA.saveobj() ;
end
if ~isfield(netB, 'meta'),    netB = dagnn.DagNN.fromSimpleNN(netB) ; netB = netB.saveobj() ; end;

if ~isfield(opts.train, 'fusionLayer');
  sum_layersA = find(strcmp({netA.layers(:).type},'dagnn.Sum')) + 1 ; % +1 for relu
  sum_layersB = find(strcmp({netB.layers(:).type},'dagnn.Sum')) + 1 ;
  opts.train.fusionLayer = {netA.layers(sum_layersA).name; netB.layers(sum_layersB).name};
  opts.train.fusionLayer = reshape({opts.train.fusionLayer{:}},2,[])';
end

f = find(strcmp({netA.layers(:).type}, 'dagnn.Loss'));
netA.layers(f(1)-1).name = 'prediction';
f = find(strcmp({netB.layers(:).type}, 'dagnn.Loss'));
netB.layers(f(1)-1).name = 'prediction';

fusionLayerA = []; fusionLayerB = [];
if ~isempty(opts.train.fusionLayer)
  for i=1:numel(netA.layers)
   if isfield(netA.layers(i),'name') && any(strcmp(netA.layers(i).name,opts.train.fusionLayer(:,1)))
     fusionLayerA = [fusionLayerA i]; 
   end                
  end
  for i=1:numel(netB.layers)
   if  isfield(netB.layers(i),'name') && any(strcmp(netB.layers(i).name,opts.train.fusionLayer(:,2)))
     fusionLayerB = [fusionLayerB i]; 
   end                
  end
end

netNormalization = netB.meta.normalization;
netA.meta.normalization.averageImage = mean(mean(netA.meta.normalization.averageImage, 1), 2);
netB.meta.normalization.averageImage = mean(mean(netB.meta.normalization.averageImage, 1), 2);



netB.meta.normalization.averageImage = gather(cat(3,netB.meta.normalization.averageImage, netA.meta.normalization.averageImage));

% rename layers, params and vars
for x=1:numel(netA.layers)
  if isfield(netA.layers(x), 'name'), netA.layers(x).name = [netA.layers(x).name '_spatial'] ;  end
end
for x=1:numel(netB.layers)
  if isfield(netB.layers(x), 'name'), netB.layers(x).name = [netB.layers(x).name '_temporal']; end
end
  
netA =  dagnn.DagNN.loadobj(netA);
for i = 1:numel(netA.vars),  if~strcmp(netA.vars(i).name,'label'), netA.renameVar(netA.vars(i).name, [netA.vars(i).name '_spatial']); end; end; 
for i = 1:numel(netA.params),  netA.renameParam(netA.params(i).name, [netA.params(i).name '_spatial']); end; 

netB =  dagnn.DagNN.loadobj(netB);
for i = 1:numel(netB.vars), if~strcmp(netB.vars(i).name,'label'), netB.renameVar(netB.vars(i).name, [netB.vars(i).name '_temporal']); end;end; 
for i = 1:numel(netB.params),  netB.renameParam(netB.params(i).name, [netB.params(i).name '_temporal']); end; 
if opts.addConv3D & any(~cellfun(@isempty,(strfind(opts.train.fusionLayer, 'prediction'))))
  if strcmp(opts.train.fuseInto,'temporal')
    [ netB ] = insert_conv_layers( netB, fusionLayerB(end), 'initMethod', initMethod, 'batchNormalization', bnormFusion, 'dropOutRatio', 0  );
  else
    [ netA ] = insert_conv_layers( netA, fusionLayerA(end), 'initMethod', initMethod, 'batchNormalization', bnormFusion, 'dropOutRatio', 0  );
  end
end
if ~opts.addConv3D && ~opts.doSum
  if strcmp(opts.train.fuseInto,'temporal')
    [ netB ] = insert_conv_layers( netB, fusionLayerB, 'initMethod', initMethod, 'batchNormalization', bnormFusion, 'dropOutRatio', 0  );
  else
    [ netA ] = insert_conv_layers( netA, fusionLayerA, 'initMethod', initMethod, 'batchNormalization', bnormFusion, 'dropOutRatio', 0  );
  end
end

if opts.train.removeFuseFrom, 
  switch opts.train.fuseFrom
    case 'spatial'
      netA.layers = netA.layers(1:fusionLayerA(end)); netA.rebuild;
    case'temporal'
      netB.layers = netB.layers(1:fusionLayerB(end)); netB.rebuild;
  end
end

netA = netA.saveobj() ;
netB = netB.saveobj() ;

net.layers = [netA.layers netB.layers] ;
net.params =  [netA.params netB.params] ;     
net.meta = netB.meta;
net = dagnn.DagNN.loadobj(net);

net = dagnn.DagNN.setLrWd(net, 'convFiltersLRWD', [1 1], 'convBiasesLRWD', [2 0], ...
  'fusionFiltersLRWD', [1 1], 'fusionBiasesLRWD', [2 0], ...
  'filtersLRWD' , [1 1], 'biasesLRWD' , [2 0] ) ;

clear netA netB;
if opts.doBiDirectional, nFuse = 2; else nFuse=1; end;
for s=1:nFuse
  if s==2
    [opts.train.fuseInto, opts.train.fuseFrom] = deal(opts.train.fuseFrom, opts.train.fuseInto);
  end
  for i = 1:size(opts.train.fusionLayer,1)
    if strcmp(opts.train.fuseInto,'spatial')
      i_fusion = find(~cellfun('isempty', strfind({net.layers.name}, ...
        [opts.train.fusionLayer{i,1} '_' opts.train.fuseInto])));
    else
        i_fusion = find(~cellfun('isempty', strfind({net.layers.name}, ...
        [opts.train.fusionLayer{i,2} '_' opts.train.fuseInto])));
    end

    if opts.doSum 
      inputVars =  net.layers(strcmp({net.layers.name},[opts.train.fusionLayer{i,1}, ...
          '_' opts.train.fuseFrom])).outputs;

      if opts.doScale
        name = [net.layers(i_fusion(end)).name '_scale'];

        block = dagnn.Scale() ;
        for j = i_fusion(end):-1:1
          p_from = net.layers(j).params; 
          if numel(p_from) == 1 ||  numel(p_from) == 2, break; end;
        end

        block.size = [1 1 size(net.params(net.getParamIndex(p_from{1})).value,4)];

        pars = {[name '_f'], [name '_b']} ;
        net.addLayerAt(i_fusion(end), name, ...
          block, ...
          inputVars, ...
          name, ...
          pars) ;
         params = block.initParams(0.1);
        [net.params(net.getParamIndex(pars)).value] = deal(params{:}) ;


        inputVars = {name};
        i_fusion(end) = i_fusion(end)+1;
      end

      name = [net.layers(i_fusion(end)).name '_sum'];
      
      block = dagnn.Sum() ;

      inputVars = [inputVars{:} net.layers(strcmp({net.layers.name},[opts.train.fusionLayer{i,2}, ...
          '_' opts.train.fuseInto])).outputs];
      net.addLayerAt(i_fusion(end), name, block, ...
                 inputVars, ...
                  name) ;      
        % next line is for disabling ReLUshortcut          
        net.layers(cellfun(@(x) any(isequal(x,net.getVarIndex(inputVars(end)))), {net.layers.inputIndexes} )).block.useShortCircuit = 0;

    else
      name = [net.layers(i_fusion(end)).name '_concat'];
      block = dagnn.Concat() ;
      net.addLayerAt(i_fusion(end), name, block, ...
                 [net.layers(strcmp({net.layers.name},[opts.train.fusionLayer{i,1} '_spatial'])).outputs ...
                  net.layers(strcmp({net.layers.name},[opts.train.fusionLayer{i,2} '_temporal'])).outputs], ...
                  name) ;   
    end
  
    net.layers(i_fusion(end)+2).inputs{1} = name;

  end
end

net.addVar('input_flow')
net.vars(net.getVarIndex('input_flow')).fanout = net.vars(net.getVarIndex('input_flow')).fanout + 1 ;

% set input for conv layers
i_conv1= find(~cellfun('isempty', strfind({net.layers.name},'temporal')));
net.layers(i_conv1(1)).inputs = {'input_flow'};

net.renameVar(net.vars(1).name, 'input');


if opts.addConv3D
    conv_layers = find(arrayfun(@(x) isa(x.block,'dagnn.Conv') && isequal(x.block.size(1),1) ... 
      && (~isempty(strfind(x.name,'b_branch2a')) || ~isempty(strfind(x.name,'b_branch2c'))) && isequal(x.block.stride(1),1), net.layers ))

    for l=conv_layers
       disp(['converting ' net.layers(l).name ' to ConvTime'])
       block = dagnn.ConvTime() ;   block.net = net ;

       kernel = net.params(net.getParamIndex(net.layers(l).params{1})).value;
       sz = size(kernel); 
       kernel = cat(2, kernel/3, kernel/3, kernel/3  );

       
       net.params(net.getParamIndex(net.layers(l).params{1})).value = kernel;
       pads = size(kernel); pads = ceil(pads(1:2) / 2) - 1;
       
       block.pad = [pads(1),pads(1), pads(2),pads(2)] ; 
       block.size = size(kernel);
       block.hasBias = net.layers(l).block.hasBias;

       block.stride(1) =  net.layers(l).block.stride(1)*net.layers(l).block.stride(2);
       net.layers(l).block = block;    
    end
end
if opts.addPool3D

  poolLayers = {'pool5'; };


  for j=1:numel(poolLayers)
    for s = {'spatial', 'temporal'}
      i_pool = find(strcmp({net.layers.name},[poolLayers{j} '_' char(s)]));    
      block = dagnn.PoolTime() ;
      block.poolSize = [1 opts.nFrames ];  
      block.pad = [0 0 0 0]; 
      block.stride = [1 1];  
      block.method = 'avg';     
      nPool = 1 ;
      if opts.addPool3D == 2
        block.method = 'max';  
      end

      name = {};
      outputVars = {};
      inputVars = net.layers(i_pool).outputs;
      for k=1:nPool
        name{end+1} = [poolLayers{j} '_pool_' char(s) num2str(k)];
        disp(['injecting ' name{end} ' as PoolTime'])
        net.addLayerAt(i_pool, name{end}, block, ...
                      [inputVars], {name{end}}) ;        
      end
                  
      % chain input of l that has layer as input
      for l = 1:numel(net.layers)    
          if ~strcmp(net.layers(l).name, name)
            sel = find(strcmp(net.layers(l).inputs, net.layers(i_pool).outputs{1})) ;
            if any(sel)
                net.layers(l).inputs{sel} = inputVars;
            end  
          end
      end

    end
  end
end % add pool3d

net = dagnn.DagNN.insertLossLayers(net, 'numClasses', nClasses) ;


opts.train.augmentation = 'f25noCtr';
opts.train.frameSample = 'temporalStrideRandom';
opts.train.nFramesPerVid = opts.nFrames * 1;
opts.train.temporalStride = 5:15;  

opts.train.valmode = 'temporalStrideRandom';
opts.train.numValFrames = 25 ;
opts.train.saveAllPredScores = 1 ;
opts.train.denseEval = 1;
opts.train.opts.nFramestack = opts.nFrames;

  
 
net.meta.normalization.rgbVariance = [];
opts.train.train = find(ismember(imdb.images.set, [1])) ;
opts.train.train = repmat(opts.train.train,1,opts.train.epochFactor);

% opts.train.train = NaN; % for testing only
opts.train.denseEval = 1;

%%
zero_drs =  find(arrayfun(@(x) isa(x.block,'dagnn.DropOut') && x.block.rate == 0, net.layers)) ;
net.removeLayer({net.layers(zero_drs).name}, true);

if ~isnan(opts.dropOutRatio)
  dr_layers = find(arrayfun(@(x) isa(x.block,'dagnn.DropOut'), net.layers)) ;
  if opts.dropOutRatio > 0
    net.layers(dr_layers).block.rate = opts.dropOutRatio;
  else
    net.removeLayer({net.layers(dr_layers).name}, true);
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
   newName = ['opts.singleSoftMaxConcat']; 
   net.addLayer(newName, block, ...
                 [net.layers(pred_layers).inputs], ...
                 newName) ;   
  net.layers(pred_layers(1)).inputs =    newName ; 
  % remove layers of the second prediction path
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
