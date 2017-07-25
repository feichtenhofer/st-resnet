function [ opts ] = cnn_setup_environment( varargin )
% 
% run(fullfile(fileparts(mfilename('fullpath')), ...
%   '..', '..', '..', 'code', 'toolboxes', 'matconvnet','matlab', 'vl_setupnn.m')) ;
run(fullfile(fileparts(mfilename('fullpath')), ...
  'matconvnet','matlab', 'vl_setupnn.m')) ;
opts.cudnnWorkspaceLimit = [];
opts.train.gpus = 1;

opts = vl_argparse(opts, varargin);
if ispc 
    if strcmp(getenv('USERNAME'), 'chris')
      opts.dataPath = 'D:/datasets/';
      opts.modelPath =    'D:/MatConvNet/models';
%       opts.flowDir = 'E:/ucf101/brox_flow_scaled';
%       opts.flowDir = 'E:\ucf101\tvl1_flow';
            opts.flowDir = 'D:\datasets\ucf101\tvl1_flow';
        opts.imageDir = 'D:\datasets\ucf101\jpegs_256';

      
      opts.numFetchThreads = 8 ;
 
    elseif  strcmp(getenv('USERNAME'), 'christoph')
      opts.dataPath = 'H:/datasets/';
      opts.modelPath =    'H:/MatConvNet/models';
%       opts.flowDir = 'H:/datasets/ucf101\brox_flow_scaled';
      opts.flowDir = 'H:\datasets\ucf101\tvl1_flow';
            opts.imageDir = 'H:\datasets\ucf101\jpegs_256';

      opts.numFetchThreads = 8 ;

    end
else
    homedir = getenv('HOME'); 
    if strfind(homedir, 'christoph')
%       opts.train.cudnnWorkspaceLimit = 1*512*1024*1204 ; % 1GB
      opts.dataPath = '/media/christoph/ssd1/datasets';
%       opts.dataPath = '/media/christoph/3TBRaid0/';

      opts.modelPath ='/media/christoph/hdd1//MatConvNet/models';

      opts.flowDir = '/media/christoph/ssdraid0/ucf101/brox_flow_scaled';
      opts.flowDir = '/home/christoph/ucf101/tvl1_flow';
      opts.imageDir  = '/media/christoph/ssd1/datasets/ucf101/jpegs_256';
      
      opts.numFetchThreads = 12 ;
%       run(fullfile(fileparts(mfilename('fullpath')), ...
%       '..', '..', '..', 'code', 'toolboxes', 'matconvnet-cuda65', 'matlab', 'vl_setupnn.m')) ;
%       run(fullfile(fileparts(mfilename('fullpath')), ...
%       '..', '..', '..', 'code', 'toolboxes', 'matconvnet', 'matlab', 'vl_setupnn.m')) ;
  
      if strfind(getenv('THISPC'), 'TITAN')   
%           opts.imageDir  = '/media/christoph/data/data/ucf101/jpegs_256';
          opts.imageDir  = '/media/christoph/ssd1/datasets/ucf101/jpegs_256';
           opts.flowDir  = '/media/christoph/ssd1/datasets/ucf101/tvl1_flow';
%             opts.modelPath ='/home/christoph/OneDrive/Research/models_new';
%         
          opts.numFetchThreads = 16 
      else
        if ~isempty(opts.train.gpus) && opts.train.gpus(1) > 1
          root = vl_rootnn() ;
          addpath(fullfile(root, 'matlab', 'mex35')) ;
        end
      end
      
    elseif strfind(getenv('SESSION_MANAGER'), 'fermat')     ; 
      opts.dataPath = '/media/chris/ssd1/datasets';
      opts.modelPath ='/home/chris/OneDrive/Research/models';
      opts.flowDir = '/media/chris/ssd1/datasets/ucf101/tvl1_flow';
      opts.imageDir = '/media/chris/ssd1/datasets/ucf101/jpegs_256';
%       run(fullfile(fileparts(mfilename('fullpath')), ...
%       '..', '..', '..', 'code', 'toolboxes', 'matconvnet', 'matlab', 'vl_setupnn.m')) ;
      opts.numFetchThreads = 16 ;
    elseif strfind(getenv('SESSION_MANAGER'), 'abel')     ; 
      opts.dataPath = '/media/chris/disk0/datasets/';
      opts.modelPath ='/media/rwlab/MatConvNet';
      opts.flowDir = '/media/chris/disk0/datasets/ucf101/tvl1_flow';
      opts.imageDir = '/media/chris/disk0/datasets/ucf101/jpegs_256';
      
%       opts.train.cudnnWorkspaceLimit = 1*1024*1024*1204 ; % 1GB

      opts.numFetchThreads = 16 ;
    else
      opts.dataPath = '/media/chris/Data/datasets/';
      opts.modelPath =    '/media/chris/Data/MatConvNet/models';
%       opts.flowDir = '/home/chris/hmdb51/tvl1_flow';
%       opts.imageDir = '/home/chris/hmdb51/jpegs_256';
      opts.flowDir = '/media/chris/FastData/ucf101/tvl1_flow';
%       opts.flowDir = '/media/chris/FastData/ucf101/brox_flow_scaled';

      opts.numFetchThreads = 8 ;
%       run(fullfile('/home/chris/git/matconvnet-devel', ...
%       'matlab', 'vl_setupnn.m')) ;

    opts.train.gpus = [1];

    end
end

end

