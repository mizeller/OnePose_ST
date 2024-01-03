from typing import List
import os.path as osp
from tqdm import tqdm
from loguru import logger
import yaml
import demo
from tqdm import tqdm
from loguru import logger
import os
from pathlib import Path
from collections import defaultdict
from cotracker.cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.cotracker.predictor import CoTrackerPredictor, CoTrackerOnlinePredictor


os.environ[
    "TORCH_USE_RTLD_GLOBAL"
] = "TRUE"  # important for DeepLM module, this line should before import torch
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
        self.obj_name: str = "spot_rgb"  # "spot"
        self.data_root: str = f"/workspaces/OnePose_ST/data/{self.obj_name}"
        # NOTE: there must exist a "color_full" sub-directory in ∀ data_dirs
        self.data_dirs: List[str] = [
            # "asus_00-test",
            # "asus_01-test",
            # "asus_02-test",
            # "asus_03-test",
            # "asus_04-test",
            # "asus_05-test",
            "asus_05_small-test",
            # "asus_06-test",
            # "asus_07-test",
            # "spot_yt-test",
            # "spot_yt_cropped-test",
            # "cotrack-test"
        ]
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

    queries = torch.empty(0, 3).cuda()  # replace this with your tensor

     
    for id in tqdm(range(len(dataset))):
        data = dataset[id]
        query_image = data["query_image"]

        
        # convert the query image from torch.tensor to array
        # use the query array for the co-tracking algorithm
    #    query_array = query_image.repeat(1, 3, 1, 1)
    #    query_array = query_array.numpy()
    #    query_array = np.transpose(query_array, (2, 3, 1))

        query_image_path = data["query_image_path"]

        if K is None:
            K = data_utils.infer_K(img_path=query_image_path)

        # Detect object:
        if id == 0:
            logger.warning(f"Running local feature object detector for frame {id}")
            # Detect object by 2D local feature matching for the first frame:
            _, inp_crop, K_crop, transformation_matrix = local_feature_obj_detector.detect(
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
                _, inp_crop, K_crop, transformation_matrix = local_feature_obj_detector.detect(
                    query_image, query_image_path, K
                )
            else:
                (
                    _,
                    inp_crop,
                    K_crop,
                    transformation_matrix
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
            # only use inliers from every 2nd frame; frequently ran into CUDA OOM errors otherwise...
            x, y = mkpts_query[inlier]
            M_inv = np.linalg.inv(transformation_matrix)
            original_coords = np.dot(M_inv, np.array([x,y, 1]))
            x_orig = original_coords[0] / original_coords[2]
            y_orig = original_coords[1] / original_coords[2]
            queries = torch.cat([queries, torch.tensor([[int(id), int(x_orig), int(y_orig)]], device='cuda')], dim=0)
            detection_counter[tuple(mkpts_3d[inlier])] += 1

        # visualise the detected keypoints on the 3D pointcloud and the query image.
        # ransac_PnP operates on these sets of keypoints
        if cfg.DBG:
            vis_utils.visualize_2D_3D_keypoints(
                path=Path(f"temp/{local_feature_obj_detector.query_id}"),
                data=data,
                inp_crop=inp_crop,
                inliers=inliers,
                mkpts_3d=mkpts_3d,
                mkpts_query=mkpts_query,
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
    vis_utils.make_video(
        paths["vis_box_dir"], f"temp/demo_{seq_dir.split('/')[-1]}.mp4"
    )
    
    # reset cuda cache for co-tracker 
    torch.cuda.empty_cache() 
    # Output video to visualize the co-tracking on the detected keypoints 
    video = read_video_from_path(f"{seq_dir}/clip.mp4")
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float() 
    
    model = CoTrackerPredictor()
    video = video.cuda()
    model = model.cuda()
    
    
    pred_tracks, pred_visibility = model(video, queries=queries[None], backward_tracking=True)
    vis = Visualizer(
    save_dir='temp/',
    linewidth=6,
    mode='cool',
    tracks_leave_trace=-1
    )
    vis.visualize(
    video=video,
    tracks=pred_tracks,
    visibility=pred_visibility,
    filename=f"demo_cotracker_{seq_dir.split('/')[-1]}") 
    logger.info(f"Cotracker demo vido saved to: temp/demo_cotracker_{seq_dir.split('/')[-1]}.mp4")

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

    if cfg.DBG:
        data_io.save_ply("model/detections_pointcloud", detection_counter)
    logger.info("Done")


if __name__ == "__main__":
    global cfg
    cfg: CONFIG = CONFIG()
    os.system(f"rm -rf temp/*")
    main()
