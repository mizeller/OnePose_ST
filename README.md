# Spot Pose Estimation

Set-up, training and testing custom pose estimation pipelines is non-trivial. It can be a tedious and time-consuming process. This repository aims to simplify this.

 <!-- by providing a Docker container and a minimal working example for training a custom pose estimation model for Boston Dynamics' Spot Robot using an extended version of [OnePose++](https://github.com/zju3dv/OnePose_Plus_Plus).
 -->

The main contributions can be summarized as follows:
- a Docker container ready to run an extended version of OnePose++
- OnePose++ extended with:
    - [DeepSingleCameraCalibration](https://github.com/AlanSavio25/DeepSingleImageCalibration/) for running inference on in-the-wild videos
    - [CoTracker2]() for pose estimation optimization, improving the pose *tracking* performance by leveraging temporal cues as well[^2].
 
- `launch.json` & `inference.py` to help understand the whole pipeline and readily debug/test the code. All modifications should be made in `inference.py` - no need to get familiar with different config files. 

- custom data for Spot & instructions on how you can create the synthetic data for your own use-case
   
[^2]: **Note:** As of this writing, CoTracker2 is still a work-in-progress. The online tracker can only run on every 4th frame which does not suffice for optimizing the pose estimation. That's why we currently use CoTracker as a post-processing step to optimize the poses for a given sequence. The *'yet'* in [this](https://github.com/facebookresearch/co-tracker/issues/56#issuecomment-1878778614) reply by the authors suggests that this feauture will be added to CoTracker2 in the future. A possible initial implementation is on this [feature branch](https://github.com/mizeller/OnePose_ST/tree/feat-online-tracker). It has not been updated in a while...

## Set-Up

```shell
# clone/enter repository
git clone git@github.com:mizeller/OnePose_ST.git
cd OnePose_ST
REPO_ROOT=$(pwd)

# init submodules
git submodule update --init --recursive
```
Download the [pretrained models](https://zjueducn-my.sharepoint.com/:f:/g/personal/12121064_zju_edu_cn/EhRhr5PMG-ZLkQjClFCUYhIB_6-307bBmepX_5Cej4Z_wg?e=tSNHMn) and the [demo data](https://zjueducn-my.sharepoint.com/personal/12121064_zju_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F12121064_zju_edu_cn%2FDocuments%2Fdemo_data&ga=1).


```shell
# unzip the demo data
mkdir ${REPO_ROOT}/data/demo
unzip <path/to/demo_cam.zip> -d ${REPO_ROOT}/data/demo

# unzip the pretrained models
mkdir ${REPO_ROOT}/weight
unzip  -j <path/to/pretrained_model.zip> -d ${REPO_ROOT}/weight

# and finally 
docker build -t="spot_pose_estimation:00" .
```
<details>

<summary>Hardware Specifications</summary>

This set up was tested and ran succesfully on a machine with the following specifications:

- DISTRIB_DESCRIPTION="Ubuntu 20.04.6 LTS"
- NVIDIA-SMI (Driver Versions) 470.223.02   
- CUDA Version: 11.4
- Docker version 24.0.7, build afdd53b

</details>


## DEMO
In this section the previous installation should be tested. 

First launch the docker container OR start a dev container.
```shell
docker run --gpus all -w /workspaces/OnePose_ST -v ${REPO_ROOT}:/workspaces/OnePose_ST -it spot_pose_estimation:00
```
OR
```
CTRL+SHIFT+P
Dev Containers: Rebuild and Reopen in Container 
```
This should automatically mount the `${REPO_ROOT}` in the container. 

Then run the demo script
```shell
bash scripts/demo_pipeline.sh demo_cam
```

You can also use the built-in VSCode debugger to follow the demo pipeline step by step. (Or modify/extend the `launch.json` to your liking.)

## Acknowledgement
This repository is essentially a fork of the original OnePose++ repository - for more details, have a look at the original source [here](https://github.com/zju3dv/OnePose_Plus_Plus). Thanks to the original authors for their great work!

## Credits
This project was developed as part of the Semester Thesis for my (Michel Zeller) MSc. Mechanical Engineering at ETH Zurich. The project was supervised by Dr. Hermann Blum (ETH, Computer Vision and Geometry Group) and Francesco Milano (ETH, Autonomous Systems Lab). 