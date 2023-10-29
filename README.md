# OnePose++: Keypoint-Free One-Shot Object Pose Estimation without CAD Models
> This is a slightly modified fork of the original OnePose++ repository to handle our own synthetic data. 
> 
> For more information about the dataset and how it was created check out this repo: [Monocular Pose Estimation Pipeline for Boston Dynamic's Spot](https://github.com/mizeller/Monocluar-Pose-Estimation-Pipeline-for-Spot)

## Installation

```shell
conda create --name oneposeplus -y python=3.7
conda activate oneposeplus
pip install -r requirements.txt
conda install -c -y pytorch pytorch=1.8.0
```

[LoFTR](https://github.com/zju3dv/LoFTR) and [DeepLM](https://github.com/hjwdzh/DeepLM) are used in this project. Thanks for their great work, and we appreciate their contribution to the community. Please follow their installation instructions and LICENSE:
```shell
git submodule update --init --recursive

# Install DeepLM
cd submodules/DeepLM
sh example.sh
cp ${REPO_ROOT}/backup/deeplm_init_backup.py ${REPO_ROOT}/submodules/DeepLM/__init__.py
```
Note that the efficient optimizer DeepLM is used in our SfM refinement phase. If you face difficulty in installation, do not worry. You can still run the code by using our first-order optimizer, which is a little slower.

[COLMAP](https://colmap.github.io/) is also used in this project for Structure-from-Motion. Please refer to the official [instructions](https://colmap.github.io/install.html) for the installation.

Download the [pretrained models](https://zjueducn-my.sharepoint.com/:f:/g/personal/12121064_zju_edu_cn/EhRhr5PMG-ZLkQjClFCUYhIB_6-307bBmepX_5Cej4Z_wg?e=tSNHMn), including our 2D-3D matching and LoFTR models. Then move them to `${REPO_ROOT}/weights`.

[Optional] You may optionally try out our web-based 3D visualization tool [Wis3D](https://github.com/zju3dv/Wis3D) for convenient and interactive visualizations of feature matches and point clouds. We also provide many other cool visualization features in Wis3D, welcome to try it out.

```bash
# Working in progress, should be ready very soon, only available on test-pypi now.
pip install -i https://test.pypi.org/simple/ wis3d
```
## Demo
After the installation, you can refer to [this page](doc/demo.md) to run the demo with your custom data.





## Acknowledgement
This is a fork of the original OnePose++ repository - for more details, have a look at the original source [here](https://github.com/zju3dv/OnePose_Plus_Plus). Thanks to the original authors for their great work!

If you find this repo useful, please consider citing their paper using the following BibTeX entry.

```bibtex
@inproceedings{
    he2022oneposeplusplus,
    title={OnePose++: Keypoint-Free One-Shot Object Pose Estimation without {CAD} Models},
    author={Xingyi He and Jiaming Sun and Yuang Wang and Di Huang and Hujun Bao and Xiaowei Zhou},
    booktitle={Advances in Neural Information Processing Systems},
    year={2022}
}
```