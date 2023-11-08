# Spot Pose Estimation

## Hey!
This is a slightly modified fork of the original OnePose++ repository to handle our own synthetic data. 

For more information about the dataset and how it was created check out this repo: [Monocular Pose Estimation Pipeline for Boston Dynamic's Spot](https://github.com/mizeller/Monocluar-Pose-Estimation-Pipeline-for-Spot)

Since quite some effort went into setting up a working-environment, and to simplify collaboration on this project, we compiled a working docker container. 

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
```