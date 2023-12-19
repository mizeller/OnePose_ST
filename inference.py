from typing import List
import os.path as osp
from tqdm import tqdm
from loguru import logger
import yaml
import demo
from tqdm import tqdm
from loguru import logger
import os
from collections import defaultdict

os.environ[
    "TORCH_USE_RTLD_GLOBAL"
] = "TRUE"  # important for DeepLM module, this line should before import torch
import os.path as osp
import numpy as np
import torch

from src.utils import data_io
from src.utils import data_utils
from src.utils import vis_utils
from src.utils import metric_utils
from src.datasets.OnePosePlus_inference_dataset import OnePosePlusInferenceDataset
from src.inference import inference_OnePosePlus
from src.local_feature_object_detector.local_feature_2D_detector import (
    LocalFeatureObjectDetector,
)
from src.models.OnePosePlus.OnePosePlusModel import OnePosePlus_model


####################################################################################################
######################## TODO - MAKE YOUR MODIFICATIONS HERE M######################################
####################################################################################################
class CONFIG:
    class DATAMODULE:
        # this class will never be used outside of the CONFIG scope
        img_pad: bool = False
        df: int = 8
        pad3D: bool = (
            False  # if True: pad 3D point cloud to shape3d_val else: use all points
        )
        shape3d_val: int = 7000  # #points in 3D point cloud; only used if pad3D is True

    def __init__(self):
        # TODO: adapt obj_name and data_dirs to your needs
        self.obj_name: str = "spot"  # "spot"
        self.data_root: str = f"/workspaces/OnePose_ST/data/{self.obj_name}"
        # NOTE: there must exist a "color_full" sub-directory in ∀ data_dirs
        self.data_dirs: List[str] = ["spot_small-test"]
        self.sfm_model_dir: str = (
            f"{self.data_root}/sfm_model/outputs_softmax_loftr_loftr/{self.obj_name}"
        )
        self.datamodule = CONFIG.DATAMODULE()
        self.model: dict = self._get_model()
        self.DBG: bool = False

    def _get_model(self) -> dict:
        with open("configs/experiment/inference_demo.yaml", "r") as f:
            onepose_config = yaml.load(f, Loader=yaml.FullLoader)
        return onepose_config["model"]


####################################################################################################
####################################################################################################


def inference_core(seq_dir, detection_counter: defaultdict):
    """Inference core function for OnePosePlus. Adapted from demo.py"""
    img_list, paths = demo.get_default_paths(cfg.data_root, seq_dir, cfg.sfm_model_dir)
    dataset = OnePosePlusInferenceDataset(
        paths["sfm_dir"],
        img_list,
        shape3d=cfg.datamodule.shape3d_val,
        img_pad=cfg.datamodule.img_pad,
        img_resize=None,
        df=cfg.datamodule.df,
        pad=cfg.datamodule.pad3D,
        # n_images=3, # consider all images
        DBG=cfg.DBG,
    )
    local_feature_obj_detector: LocalFeatureObjectDetector = LocalFeatureObjectDetector(
        sfm_ws_dir=paths["sfm_ws_dir"],
        output_results=True,
        detect_save_dir=paths["vis_detector_dir"],
        K_crop_save_dir=paths["vis_detector_dir"],
        DBG=cfg.DBG,
    )
    match_2D_3D_model: OnePosePlus_model = inference_OnePosePlus.build_model(
        cfg.model["OnePosePlus"], cfg.model["pretrained_ckpt"]
    )
    match_2D_3D_model.cuda()

    K, _ = data_utils.get_K(intrin_file=paths["intrin_full_path"])

    bbox3d = np.loadtxt(paths["bbox3d_path"])
    pred_poses = {}  # {id:[pred_pose, inliers]}
    for id in tqdm(range(len(dataset))):
        data = dataset[id]
        query_image = data["query_image"]
        query_image_path = data["query_image_path"]

        if K is None:
            K = data_utils.infer_K(img_path=query_image_path)

        # Detect object:
        if id == 0:
            logger.warning(f"Running local feature object detector for frame {id}")
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
        pose_pred, _, inliers, _ = metric_utils.ransac_PnP(
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
        for inlier in inliers:
            detection_counter[tuple(mkpts_3d[inlier])] += 1

        # visualise the detected keypoints on the 3D pointcloud and the query image.
        # ransac_PnP operates on these sets of keypoints
        if cfg.DBG:
            vis_utils.visualize_2D_3D_keypoints(
                data, inp_crop, inliers, mkpts_3d, mkpts_query
            )

        # Visualize:
        vis_utils.save_demo_image(
            pose_pred,
            K,
            image_path=query_image_path,
            box3d=bbox3d,
            draw_box=len(inliers) > 20,
            save_path=osp.join(paths["vis_box_dir"], f"{id}.jpg"),
        )

    # Output video to visualize estimated poses:
    logger.info(f"Generate demo video begin...")
    vis_utils.make_video(paths["vis_box_dir"], paths["demo_video_path"])

    return detection_counter


def main() -> None:
    keypoints3d = np.load(f"{cfg.sfm_model_dir}/anno/anno_3d_average.npz")[
        "keypoints3d"
    ]  # [m, 3]

    # init detection counter
    detection_counter = defaultdict(int)
    for coord in keypoints3d:
        detection_counter[tuple(coord)] = 0

    # loop over all data_dirs specified in the CONFIG class
    for test_dir in tqdm(cfg.data_dirs, total=len(cfg.data_dirs)):
        seq_dir = osp.join(cfg.data_root, test_dir)
        logger.info(f"Eval {seq_dir}")
        detection_counter = inference_core(seq_dir, detection_counter)

    if True:  # cfg.DBG:
        data_io.save_ply("detected_pointcloud", detection_counter)
    logger.info("Done")


if __name__ == "__main__":
    global cfg
    cfg: CONFIG = CONFIG()
    os.system(f"rm -rf temp/*")
    main()
