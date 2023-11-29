from typing import List
import os.path as osp
from tqdm import tqdm
from loguru import logger
import yaml
import demo
from tqdm import tqdm
from loguru import logger
import os

os.environ[
    "TORCH_USE_RTLD_GLOBAL"
] = "TRUE"  # important for DeepLM module, this line should before import torch
import os.path as osp
import numpy as np
import torch

from src.utils import data_utils
from src.utils import vis_utils
from src.utils.metric_utils import ransac_PnP
from src.datasets.OnePosePlus_inference_dataset import OnePosePlusInferenceDataset
from src.inference.inference_OnePosePlus import build_model
from src.local_feature_object_detector.local_feature_2D_detector import (
    LocalFeatureObjectDetector,
)


####################################################################################################
######################## TODO - MAKE YOUR MODIFICATIONS HERE M######################################
####################################################################################################
class CONFIG:
    class DATAMODULE:
        # this class will never be used outside of the CONFIG scope
        load_3d_coarse: bool = True
        shape3d_val: int = 7000
        img_pad: bool = False
        df: int = 8
        pad3D: bool = False

    def __init__(self):
        self.obj_name: str = "spot" # TODO: change here
        self.data_root: str = f"/workspaces/OnePose_ST/data/{self.obj_name}" 
        # NOTE: there needs to exist a sub-directory called "color_full" in ∀ data_dirs which contain the image sequences.
        #       furthermore, an "instrinsics.txt" with the corresponding camera intrinsics is also required...
        self.data_dirs: List[str] = ["spot-test"]  # TODO: change here
        self.sfm_model_dir: str = (
            f"{self.data_root}/sfm_model/outputs_softmax_loftr_loftr/{self.obj_name}"
        )
        self.datamodule = CONFIG.DATAMODULE()
        self.model: dict = self._get_model()

    def _get_model(self) -> dict:
        with open("configs/experiment/inference_demo.yaml", "r") as f:
            onepose_config = yaml.load(f, Loader=yaml.FullLoader)
        return onepose_config["model"]


####################################################################################################
####################################################################################################


def inference_core(seq_dir):
    """Inference core function for OnePosePlus. Adapted from demo.py"""
    data_root = cfg.data_root
    sfm_model_dir = cfg.sfm_model_dir
    img_list, paths = demo.get_default_paths(data_root, seq_dir, sfm_model_dir)
    dataset = OnePosePlusInferenceDataset(
        paths["sfm_dir"],
        img_list,
        load_3d_coarse=cfg.datamodule.load_3d_coarse,
        shape3d=cfg.datamodule.shape3d_val,
        img_pad=cfg.datamodule.img_pad,
        img_resize=None,
        df=cfg.datamodule.df,
        pad=cfg.datamodule.pad3D,
        load_pose_gt=False,
        n_images=None,
        demo_mode=True,
        preload=True,
    )
    local_feature_obj_detector: LocalFeatureObjectDetector = LocalFeatureObjectDetector(
        sfm_ws_dir=paths["sfm_ws_dir"],
        output_results=True,
        detect_save_dir=paths["vis_detector_dir"],
    )
    match_2D_3D_model = build_model(
        cfg.model["OnePosePlus"], cfg.model["pretrained_ckpt"]
    )
    match_2D_3D_model.cuda()

    K, _ = data_utils.get_K(paths["intrin_full_path"])

    bbox3d = np.loadtxt(paths["bbox3d_path"])
    pred_poses = {}  # {id:[pred_pose, inliers]}
    for id in tqdm(range(len(dataset))):
        data = dataset[id]
        query_image = data["query_image"]
        query_image_path = data["query_image_path"]

        # Detect object:
        if id == 0:
            logger.warning(f"Re-Running local feature object detector for frame {id}")
            # Detect object by 2D local feature matching for the first frame:
            _, inp_crop, K_crop = local_feature_obj_detector.detect(
                query_image, query_image_path, K
            )
        else:
            # Use 3D bbox and previous frame's pose to yield current frame 2D bbox:
            previous_frame_pose, inliers = pred_poses[id - 1]

            if len(inliers) < 20:
                logger.warning(
                    f"Re-Running local feature object detector for frame {id}"
                )
                # Consider previous pose estimation failed, reuse local feature object detector:
                _, inp_crop, K_crop = local_feature_obj_detector.detect(
                    query_image, query_image_path, K
                )
            else:
                (
                    _,
                    inp_crop,
                    K_crop,
                ) = local_feature_obj_detector.previous_pose_detect(
                    query_image_path, K, previous_frame_pose, bbox3d
                )

        data.update({"query_image": inp_crop.cuda()})

        # Perform keypoint-free 2D-3D matching and then estimate object pose of query image by PnP:
        with torch.no_grad():
            match_2D_3D_model(data)
        mkpts_3d = data["mkpts_3d_db"].cpu().numpy()  # N*3
        mkpts_query = data["mkpts_query_f"].cpu().numpy()  # N*2
        pose_pred, _, inliers, _ = ransac_PnP(
            K_crop,
            mkpts_query,
            mkpts_3d,
            scale=1000,
            pnp_reprojection_error=7,
            img_hw=[512, 512],
            use_pycolmap_ransac=True,
        )
        logger.debug(f"Pose estimation inliers: {len(inliers)} for frame {id}")
        pred_poses[id] = [pose_pred, inliers]

        # Visualize:
        vis_utils.save_demo_image(
            pose_pred,
            K,
            image_path=query_image_path,
            box3d=bbox3d,
            draw_box=True,  # len(inliers) > 20,
            save_path=osp.join(paths["vis_box_dir"], f"{id}.jpg"),
        )

    # Output video to visualize estimated poses:
    logger.info(f"Generate demo video begin...")
    vis_utils.make_video(paths["vis_box_dir"], paths["demo_video_path"])


def main() -> None:
    # loop over all data_dirs specified in the CONFIG class
    for test_dir in tqdm(cfg.data_dirs, total=len(cfg.data_dirs)):
        seq_dir = osp.join(cfg.data_root, test_dir)
        logger.info(f"Eval {seq_dir}")
        inference_core(seq_dir)

    logger.info("Done")


if __name__ == "__main__":
    global cfg
    cfg: CONFIG = CONFIG()
    main()
