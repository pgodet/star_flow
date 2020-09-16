# STaRFlow

<img src=results.png>

This repository is the PyTorch implementation of STaRFlow, a recurrent convolutional neural network for multi-frame optical flow estimation. This algorithm is presented in our paper **STaRFlow: A SpatioTemporal Recurrent Cell for Lightweight Multi-Frame Optical Flow Estimation**, Pierre Godet, [Alexandre Boulch](https://github.com/aboulch), [Aurélien Plyer](https://github.com/aplyer), and Guy Le Besnerais.
[[Preprint]](https://arxiv.org/pdf/2007.05481.pdf)


Please cite our paper if you find our work useful.  

    @article{godet2020starflow,
      title={STaRFlow: A SpatioTemporal Recurrent Cell for Lightweight Multi-Frame Optical Flow Estimation},
      author={Godet, Pierre and Boulch, Alexandre and Plyer, Aur{\'e}lien and Le Besnerais, Guy},
      journal={arXiv preprint arXiv:2007.05481},
      year={2020}
    }

Contact: pierre.godet@onera.fr

## Getting started
This code has been developed and tested under Anaconda(Python 3.7, scipy 1.1, numpy 1.16), Pytorch 1.1 and CUDA 10.1 on Ubuntu 18.04.

1. Please install the followings:

   - Anaconda (Python 3.7)
   - __PyTorch 1.1__ (Linux, Conda, Python 3.7, CUDA 10) (`conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch`)  
   - Depending on your system, configure `-gencode`, `-ccbin`, `cuda-path` in `models/correlation_package/setup.py` accordingly
   - scipy 1.1 (`conda install scipy=1.1`)
   - colorama (`conda install colorama`)
   - tqdm 4.32 (`conda install -c conda-forge tqdm=4.32`)
   - pypng (`pip install pypng`)

2. Then, install the correlation package:
   ```
   ./install.sh
   ```


## Pretrained Models

The `saved_checkpoint` folder contains the pre-trained models of STaRFlow trained on

 1. FlyingChairsOcc -> FlyingThings3D, or
 2. FlyingChairsOcc -> FlyingThings3D -> MPI Sintel, or
 3. FlyingChairsOcc -> FlyingThings3D -> KITTI (2012 and 2015).  


## Inference

The script `inference.py` can be used for testing the pre-trained models. Example:

    python inference.py \
      --model StarFlow \
      --checkpoint saved_checkpoint/StarFlow_things/checkpoint_best.ckpt \
      --data-root /data/mpisintelcomplete/training/final/ambush_6/ \
      --file-list frame_0004.png frame_0005.png frame_0006.png frame_0007.png

By default, it saves the results in `./output/`.


## Training

Data-loaders for multi-frame training can be found in the `datasets` folder, multi-frame losses are in `losses.py`, and every architecture used in the experiments presented in our paper is available in the `models` folder.

### Datasets

The datasets used for this project are followings:

- [FlyingChairsOcc dataset](https://github.com/visinf/irr/tree/master/flyingchairsocc)
- [MPI Sintel Dataset](http://sintel.is.tue.mpg.de/downloads)
- [KITTI Optical Flow 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow) and [KITTI Optical Flow 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow)
- [FlyingThings3D subset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)


### Scripts for training

The `scripts` folder contains training scripts for STaRFlow.  
To train the model, you can simply run the script file, e.g., `./train_starflow_chairsocc.sh`.  
In script files, please configure your own experiment directory (EXPERIMENTS_HOME) and dataset directory in your local system (e.g., SINTEL_HOME or KITTI_HOME).


## Acknowledgement

This repository is a fork of the [IRR-PWC](https://github.com/visinf/irr) implementation from Junhwa Hur and Stefan Roth.
