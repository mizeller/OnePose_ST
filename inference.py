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
from submodules.CoTracker.cotracker.utils.visualizer import Visualizer, read_video_from_path
from submodules.CoTracker.cotracker.predictor import CoTrackerOnlinePredictor
from PIL import Image, ImageDraw

os.environ[
    "TORCH_USE_RTLD_GLOBAL"
] = "TRUE"  # important for DeepLM module, this line should before import torch
import numpy as np
import torch
from torchvision.transforms import ToPILImage

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
        self.data_dirs: List[str] = ["cotrack-test"]
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
##########################TODO: refactor helper methods at a later date ############################
####################################################################################################


def _process_step(model, window_frames, is_first_step, queries):
    """Process a step of the tracker.
    Args:
        model: the tracker model
        window_frames: a list of frames to process
        is_first_step: whether this is the first step
        queries: the queries tensor
    """
    video_chunk = (
        torch.tensor(np.stack(window_frames[-model.step * 2 :]), device="cuda")
        .float()
        .permute(0, 3, 1, 2)[None]
    )  # (1, T, 3, H, W)
    return model(video_chunk, is_first_step=is_first_step, queries=queries[None])


def transform_kpts(transform, mkpts_query):
    inliers_query_homo = np.c_[mkpts_query, np.ones(mkpts_query.shape[0])]
    inliers_orig_homo = np.dot(inliers_query_homo, transform.T)
    inliers_orig = inliers_orig_homo[:, :2] / inliers_orig_homo[:, 2, np.newaxis]
    return inliers_orig


def _visualize_image_tensor(image, name):
    img = ToPILImage()(image.squeeze(0))
    img.save(f"temp/debug/{name}.jpg")


def get_query_array(query_image, id):
    query_array = query_image.repeat(1, 3, 1, 1)
    query_array = query_array.numpy()
    query_array = np.squeeze(query_array, axis=0)
    query_array = np.transpose(query_array, (1, 2, 0))  # (H, W, C)

    # DBG - save the query image and the query array variables as images
    # img = Image.fromarray((query_array * 255).astype(np.uint8))
    # img.save(f"temp/debug/00_query_array_{id}.jpg")
    if cfg.DBG:
        _visualize_image_tensor(query_image, f"00_query_tensor_{id}")
    return query_array


def visualize_mkpts(
    img_tensor, mkpts_query, name, img_path: str = "", fill=(0, 255, 0), r=3
):
    if img_path:
        query_image = Image.open(img_path)
    else:
        query_image = img_tensor.squeeze().cpu().detach().numpy()
        query_image = query_image * 255
        query_image = query_image.astype(np.uint8)
        query_image = Image.fromarray(query_image).convert("RGB")
    draw = ImageDraw.Draw(query_image)
    for point in mkpts_query:
        x, y = point.astype(int)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=fill)
    query_image.save(f"temp/debug/{name}.jpg")


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

    ####### COTRACKING STUFF ########
    tracker_queries = torch.empty(0, 3, device="cuda")
    pred_tracks_orig = None
    inlier_mkpts = {}
    window_frames = []
    tracker_model = CoTrackerOnlinePredictor(checkpoint="submodules/CoTracker/checkpoints/cotracker2.pth")
    tracker_model = tracker_model.cuda()
    is_first_step = True
    do_tracking = False
    ####### COTRACKING STUFF ########

    for id in tqdm(range(len(dataset))):
        data = dataset[id]
        query_image = data["query_image"]
        query_image_path = data["query_image_path"]

        ####### COTRACKING STUFF ########
        query_array = get_query_array(query_image, id)

        # TODO: tracker should run constantly after the first time,
        # this is a missing feature in the online tracker...
        if id % tracker_model.step == 0 and id != 0:
            logger.error(f"Running CoTracking for frame {id}")
            pred_tracks_orig, pred_visibility = _process_step(
                model=tracker_model,
                window_frames=window_frames,
                is_first_step=is_first_step,
                queries=tracker_queries,
            )
            is_first_step = False
            do_tracking = True
        window_frames.append(query_array)
        ####### COTRACKING STUFF ########
        if K is None:
            K = data_utils.infer_K(img_path=query_image_path)

        # Detect object:
        if id == 0:
            logger.warning(f"Running local feature object detector for frame {id}")
            # Detect object by 2D local feature matching for the first frame:
            _, inp_crop, K_crop, transform = local_feature_obj_detector.detect(
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
                _, inp_crop, K_crop, transform = local_feature_obj_detector.detect(
                    query_image, query_image_path, K
                )
            else:
                (
                    _,
                    inp_crop,
                    K_crop,
                    transform,
                ) = local_feature_obj_detector.previous_pose_detect(
                    query_image_path, K, previous_frame_pose, bbox3d
                )
        if cfg.DBG:
            _visualize_image_tensor(inp_crop, f"01_inp_crop_{id}")  # DBG
        data.update({"query_image": inp_crop.cuda()})

        # Perform keypoint-free 2D-3D matching and then estimate object pose of query image by PnP:
        with torch.no_grad():
            match_2D_3D_model(data)
        mkpts_3d = data["mkpts_3d_db"].cpu().numpy()  # N*3
        mkpts_query = data["mkpts_query_f"].cpu().numpy()  # N*2

        # DBG
        if cfg.DBG:
            visualize_mkpts(inp_crop, mkpts_query, f"02_mkpts_query_{id}")

        if do_tracking and id != 4 and pred_tracks_orig is not None:
            # map predicted tracks from coords wrt original image to current cropped img
            transform_tensor = torch.from_numpy(transform).float().cuda()
            ones = torch.ones(*pred_tracks_orig.shape[:-1], 1, device="cuda")
            pred_tracks_orig_homo = torch.cat([pred_tracks_orig, ones], dim=-1).float()
            pred_tracks_query_homo = torch.einsum(
                "...ij,...j->...i", transform_tensor, pred_tracks_orig_homo
            )
            pred_tracks_query = (
                pred_tracks_query_homo[..., :2] / pred_tracks_query_homo[..., 2:3]
            )

            mkpts_query_tracked = pred_tracks_query[0][id - 1].cpu().numpy()
            # DBG: visualize the keypoints provided by the tracker, mapped onto the current cropped frame
            if cfg.DBG:
                visualize_mkpts(
                    _,
                    mkpts_query_tracked,
                    f"02_mkpts_injected_{id}",
                    img_path=f"temp/debug/02_mkpts_query_{id}.jpg",
                    fill=(255, 0, 255),
                    r=5,
                )

            mkpts_3d_tracked = inlier_mkpts[0]["mkpts_3d"]

            # inject these keypoints before the ransac PnP step
            mkpts_query = np.concatenate((mkpts_query, mkpts_query_tracked), axis=0)
            mkpts_3d = np.concatenate((mkpts_3d, mkpts_3d_tracked), axis=0)
            do_tracking = False

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

        # mkpts_query[inliers] = inliers in cropped image
        # inliers_orig = inliers in original image
        inliers_orig = transform_kpts(np.linalg.inv(transform), mkpts_query[inliers])

        if cfg.DBG:
            visualize_mkpts(
                _,
                mkpts_query[inliers],
                f"03_inliers_query_{id}",
                img_path=f"temp/debug/02_mkpts_query_{id}.jpg",
                fill=(255, 0, 0),
            )
            visualize_mkpts(
            query_image, inliers_orig, f"04_inliers_orig_{id}", fill=(255, 0, 0), r=1
        )

        inlier_mkpts[id] = {
            "mkpts_3d": mkpts_3d[inliers],
            "mkpts_query": mkpts_query[inliers],
            "mkpts_orig": inliers_orig,
        }  # 2D-3D correspondences cache

        # update detection_counter & tracker queries w/ inliers for current frame
        for inlier_idx, inlier_orig in zip(inliers, inliers_orig):
            if id == 0:
                x, y = inlier_orig
                tracker_queries = torch.cat(
                    [
                        tracker_queries,
                        torch.tensor([[int(id), int(x), int(y)]], device="cuda"),
                    ],
                    dim=0,
                )
            detection_counter[tuple(mkpts_3d[inlier_idx])] += 1

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

    # Processing the final video frames in case video length is not a multiple of model.step
    pred_tracks_orig, pred_visibility = _process_step(
        tracker_model,
        window_frames[-(id % tracker_model.step) - tracker_model.step - 1 :],
        is_first_step,
        tracker_queries,
    )

    # save a video with estimated poses
    logger.info(
        f"POSE ESTIMATION DEMO VIDEO SAVED TO: temp/demo_pose_{seq_dir.split('/')[-1]}.mp4"
    )
    vis_utils.make_video(
        paths["vis_box_dir"], f"temp/demo_pose_{seq_dir.split('/')[-1]}.mp4"
    )

    # save a video with predicted tracks
    logger.info(
        f"Cotracker demo vido saved to: temp/demo_track_{seq_dir.split('/')[-1]}.mp4"
    )

    torch.cuda.empty_cache()
    # Output video to visualize the co-tracking on the detected keypoints
    video = read_video_from_path(
        f"{seq_dir}/clip.mp4"
    )  # NOTE: could also use window_frames here...
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    vis = Visualizer(save_dir="temp/", mode="cool", linewidth=2, tracks_leave_trace=-1)
    vis.visualize(
        video=video,
        tracks=pred_tracks_orig,
        visibility=pred_visibility,
        filename=f"demo_track_{seq_dir.split('/')[-1]}",
    )

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
    if cfg.DBG:
        Path("temp/debug").mkdir(exist_ok=True, parents=True)
    main()
