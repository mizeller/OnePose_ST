# Spot Pose Estimation

Training and testing custom pose estimation pipelines can be a tedious and time-consuming process when initially being confronted by all the various existing algorithms out there.

This repository is building on previous work [insert link to synthetic data pipeline] & contains both the instructions on how a minimal training set of synthetic data can be created for a custom object and subsequently used for training a custom pose estimation model using an optimized version of OnePose++.

Furthermore it contains a demo script to test the trained model on a real-world dataset as well as a docker container to simplify the set-up process.

## Main Contributions
- Docker Container ready to run OnePose++
- `launch.json` to help understand the demo pipeline
- OnePose++ extended with:
    - [DeepSingleCameraCalibration](https://github.com/AlanSavio25/DeepSingleImageCalibration/) for running inference on in-the-wild sequences
    - [CoTracker2]() for pose estimation optimisation, bridging the missing cap of OnePose++ from traditional single-frame pose estimation to pose tracking, leveraging temporal cues as well[^2].
    
    [^2] **Note:** As of this writing, CoTracker2 is still a work-in-progress. The OnlineTracker does not work in an online fashion yet, but the offline tracker is fully functional and used in this pipeline as a post-processing step to optimize the pose. The *'yet'* in [this](https://github.com/facebookresearch/co-tracker/issues/56#issuecomment-1878778614) reply to an issue on the CoTracker2 repository suggests that the online tracker will be released at some point in the future as well. 
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
This project was developed as part of the Semester Thesis for my (Matthew Hanlon) MSc. Robotics, Systems and Control at ETH Zurich. The project was supervised by Eric Vollenweider (Microsoft Mixed Reality and AI Lab Zurich), in collaboration with the Computer Vision and Geometry Group.