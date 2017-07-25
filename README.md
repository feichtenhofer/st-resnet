================================================================================
# Spatiotemporal Residual Networks for Video Action Recognition

This repository contains the code for our NIPS'16 and CVPR'17 papers:

    Christoph Feichtenhofer, Axel Pinz, Richard P. Wildes
    "Spatiotemporal Residual Networks for Video Action Recognition"
    in Proc. NIPS 2016

    Christoph Feichtenhofer, Axel Pinz, Richard P. Wildes
    "Spatiotemporal Multiplier Networks for Video Action Recognition"
    in Proc. CVPR 2017

If you use our code/models/data for your research, please cite our papers:

        @inproceedings{feichtenhofer2016spatiotemporal,
          title={Spatiotemporal residual networks for video action recognition},
          author={Feichtenhofer, Christoph and Pinz, Axel and Wildes, Richard},
          booktitle={Advances in Neural Information Processing Systems (NIPS)},
          pages={3468--3476},
          year={2016}
        }

        @inproceedings{feichtenhofer2017multiplier,
          title={Spatiotemporal multiplier networks for video action recognition},
          author={Feichtenhofer, Christoph and Pinz, Axel and Wildes, Richard P}
          booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
          year={2018}
        }

# Requirements

The code was tested on Ubuntu 14.04, 16.04 and Windows 10 using MATLAB R2016b and
 NVIDIA Titan X GPUs. 

If you have questions regarding the implementation please contact:

    Christoph Feichtenhofer <feichtenhofer AT tugraz.at>

================================================================================

# Setup

1. Download the code ```git clone --recursive https://github.com/feichtenhofer/st-resnet```

2. Compile the code by running ```compile.m```.
    *  This will also compile our [own branch](https://github.com/feichtenhofer/matconvnet) of the 
[MatConvNet](http://www.vlfeat.org/matconvnet) toolbox. In case of any issues, 
please follow the [installation](http://www.vlfeat.org/matconvnet/install/) instructions on the
 MatConvNet [homepage](http://www.vlfeat.org/matconvnet).

3. Edit the file cnn_setup_environment.m to adjust the models and data paths. 

4. (Optional) Download pretrained model files and the datasets, linked below and unpack them into your models/data directory.
    Otherwise the scripts will attempt to download the models at runtime. 
* Optionally you can also pretrain your own twostream base models by running
    1. `base_streams/cnn_ucf101_spatial();` to train the appearance network streams.
    1. `base_streams/cnn_ucf101_temporal();` to train the optical flow network streams.

5. Training
`STResNet_stage1();`, `STResNet_stage2();` to train the architecture in our NIPS 2016 paper. 
`STMulNet();` to train the architecture in our CVPR 2017 paper. 
    - In case you did not download or trained the base models, the script will attempt to download these accordingly.
    - In case you would like to train on the CPU, clear the variable `opts.train.gpus`
    - In case you encounter memory issues on your GPU, consider decreasing the `cudnnWorkspaceLimit` (512MB is default)

# Models: ST-ResNet 
- Download final models here: (in case you do not, `STResNet_stage1();`, `STResNet_stage2();`, and `STResNet_test();` will attempt to download the respective models at runtime.)
    - [STResNet_base](http://ftp.tugraz.at/pub/feichtenhofer/st-res/ts-base/)
    - [STResNet_stage1](http://ftp.tugraz.at/pub/feichtenhofer/st-res/stage1/)
	- [STResNet_stage2](http://ftp.tugraz.at/pub/feichtenhofer/st-res/stage2/)
    
# Models: ST-MulNet 
- Download final models here: (in case you do not, `STMulNet();` and `STMulNet_test();` will attempt to download the models at runtime.)
    - [STMulNet_base](http://ftp.tugraz.at/pub/feichtenhofer/st-mul/ts-base/)
    - [STMulNet_final](http://ftp.tugraz.at/pub/feichtenhofer/st-mul/final/)

# Data
Pre-computed optical flow images and resized rgb frames for the [UCF101](http://crcv.ucf.edu/data/UCF101.php) and [HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) datasets
- UCF101 RGB: [part1](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.001)
[part2](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.002)
[part3](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_jpegs_256.zip.003)

- UCF101 Flow: [part1](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.001)
[part2](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.002)
[part3](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/ucf101_tvl1_flow.zip.003)

- HMDB51 RGB: [part1](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/hmdb51_jpegs_256.zip)
- HMDB51 Flow: [part1](http://ftp.tugraz.at/pub/feichtenhofer/tsfusion/data/hmdb51_tvl1_flow.zip)

# Use it on your own dataset
- Our [Optical flow extraction tool](https://github.com/feichtenhofer/gpu_flow) provides OpenCV wrappers for optical flow extraction on a GPU.