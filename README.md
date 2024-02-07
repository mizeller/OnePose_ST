# Robust Visual Pose Estimation for in the Wild Videos of Spot
#### Semester Thesis, ETH Zurich, Autumn Semester 2023
<!-- TODO: add links -->
<h4 align="center"><a href="https://drive.google.com/file/d/1XU4ywK_CfDcl1PXfGDnL9Eu9dXu7_x_I/view?usp=sharing">Thesis Report</a> | <a href="">Slides</a></h3>
<p align="center"><img src="assets/preview.gif"/></p>

Set-up, training and testing custom pose estimation pipelines is non-trivial. It can be a tedious and time-consuming process. This repository aims to simplify this.

The main contributions can be summarized as follows:
- a Docker container ready to run an extended version of OnePose++
- OnePose++ extended with:
    - [DeepSingleCameraCalibration](https://github.com/AlanSavio25/DeepSingleImageCalibration/) for running inference on in-the-wild videos
    - [CoTracker2]() for pose estimation optimization, improving the pose *tracking* performance by leveraging temporal cues as well[^2].
 
- A low-entry demo to help understand the whole pipeline and readily debug/test the code.

- custom data for Spot & instructions on how you can create the synthetic data for your own use-case
   
[^2]: **Note:** As of this writing, CoTracker2 is still a work-in-progress. The online tracker can only run on every 4th frame which does not suffice for optimizing the pose estimation. That's why we currently use CoTracker as a post-processing step to optimize the poses for a given sequence. The *'yet'* in [this](https://github.com/facebookresearch/co-tracker/issues/56#issuecomment-1878778614) reply by the authors suggests that this feauture will be added to CoTracker in the future. A possible initial implementation is on this [feature branch](https://github.com/mizeller/OnePose_ST/tree/feat-online-tracker). It has not been updated in a while...

## Installation
### Hardware 
Having a CUDA-enabled GPU is a must. The code was tested on the following GPUs:
- NVIDIA GeForce RTX 2080

with the following OS & driver versions:
```shell
DISTRIB_DESCRIPTION="Ubuntu 20.04.6 LTS"
NVIDIA-SMI (Driver Versions) 470.223.02   
CUDA Version: 11.4
Docker Version: 24.0.7, build afdd53b
```

### Code
Set up the code by cloning the repository, initializing the submodules and downloading the necessary models and demo data: 
```shell
git clone git@github.com:mizeller/OnePose_ST.git
cd OnePose_ST
git submodule update --init --recursive
mkdir -p data weight 
```
The pre-trained models for OnePose++, LoFTR and CoTracker2 as well as the demo data can be found [here](https://drive.google.com/drive/folders/1VIuflRl8WdJVcwpsHOFlmeoM7b3I1HlV?usp=sharing). Place the model files in `weight/` and the demo data in `data/`.

At this point, the project directory should look like this:
```shell
.
├── assets
...
├── data
│   └── spot_demo
├── submodules
│   ├── CoTracker
│   ├── DeepLM
│   └── LoFTR
└── weight
    ├── LoFTR_wsize9.ckpt 
    ├── OnePosePlus_model.ckpt
    └── cotracker2.pth
```

### Docker
To set up the docker container either build it locally
```shell
docker build -t="mizeller/spot_pose_estimation:00" .
```
or pull a pre-built container from DockerHub:
```shell
docker pull mizeller/spot_pose_estimation:00
```
Next, the container needs to be run. Again, there are several options to do this.

In case you're using VSCode's `devcontainer` feature, simply press `CTRL+SHIFT+P` and select `Rebuild and Reopen in Container`.
This will re-open the project in a docker container.

Alternatively, you can run the docker container directly from the terminal. The following command mounts the `${REPO_ROOT}` in the container. Note that the shared memory size is set to 32GB, change it to your hardware if necessary.

```shell
REPO_ROOT=$(pwd)
docker run --gpus all --shm-size=32g -w /workspaces/OnePose_ST -v ${REPO_ROOT}:/workspaces/OnePose_ST -it mizeller/spot_pose_estimation:00
```

## Demo: Training & Inference
To test the set up (training and inference), run the demo script from a terminal in the docker container: `sh demo.sh`. This will run the following steps:
1. Parse the demo data
2. Train the OnePose++ model for Spot
3. Run inference on the demo data captured using my phone

The results will be saved in the `temp/` directory. 

FYI: There are also custom debug entry points for each step of the pipeline. Have a look at the `.vscode/launch.json`.

### Training Data
**TODO: add comments about synthetic data pipeline & clean up the other repo as well**

## Acknowledgement & License
This repository is essentially a fork of the original OnePose++ repository - for more details, have a look at the original source [here](https://github.com/zju3dv/OnePose_Plus_Plus). Thanks to the original authors for their great work!

This repository uses several submodules, please refer to the respective repositories for their licenses.

## Credits
This project was developed as part of the Semester Thesis for my (Michel Zeller) MSc. Mechanical Engineering at ETH Zurich. The project was supervised by Dr. Hermann Blum (ETH, Computer Vision and Geometry Group) and Francesco Milano (ETH, Autonomous Systems Lab). 

