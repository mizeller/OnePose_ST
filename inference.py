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


os.environ[
    "TORCH_USE_RTLD_GLOBAL"
] = "TRUE"  # important for DeepLM module, this line should before import torch
import numpy as np
import torch

from src.utils import data_utils
from src.utils import vis_utils
from src.utils import metric_utils
from src.datasets.OnePosePlus_inference_dataset import OnePosePlusInferenceDataset
from src.inference import inference_OnePosePlus
from src.local_feature_object_detector.local_feature_2D_detector import (
    LocalFeatureObjectDetector,
)
from src.models.OnePosePlus.OnePosePlusModel import OnePosePlus_model

from submodules.CoTracker.cotracker.predictor import CoTrackerPredictor
from submodules.CoTracker.cotracker.utils.visualizer import (
    read_video_from_path,
    Visualizer,
)


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
##########################TODO: refactor helper methods later ######################################
####################################################################################################
class MKPT:
    def __init__(self, K_crop, transform, mkpts_3d, mkpts_crop, inliers_crop):
        """
        K_crop          (3,3)   camera intrinsics matrix
        transform       (3,3)   transform matrix from original image to cropped image
        mkpts_3d        (N,3)   3D key points in pointcloud
        mkpts_crop      (N,2)   2D key points in cropped image
        inliers_crop    (M,1)   list of indices corresponding to inliers in mkpts_crop and mkpts_3d
        queries         (M,3)   mkpts_crop[inliers_crop] mapped to original image w/ inv(transform)
        """
        self.inliers_crop = inliers_crop
        self.K_crop = K_crop
        self.mkpts_3d = mkpts_3d
        self.mkpts_crop = mkpts_crop
        self.transform = transform
        self.queries = self._get_queries()

    def _get_queries(self):
        inliers_orig = transform_mkpts(
            np.linalg.inv(self.transform), self.mkpts_crop[self.inliers_crop]
        )
        queries = (
            torch.from_numpy(np.c_[np.ones(inliers_orig.shape[0]), inliers_orig])
            .float()
            .cuda()
        )
        return queries


def transform_mkpts(transform, mkpts_query):
    inliers_query_homo = np.c_[mkpts_query, np.ones(mkpts_query.shape[0])]
    inliers_orig_homo = np.dot(inliers_query_homo, transform.T)
    inliers_orig = inliers_orig_homo[:, :2] / inliers_orig_homo[:, 2, np.newaxis]
    return inliers_orig


####################################################################################################
####################################################################################################


def inference_core(seq_dir):
    """Inference core function for OnePosePlus."""
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
    mkpts_cache = []

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
            _, query_crop, K_crop, transform = local_feature_obj_detector.detect(
                query_image, query_image_path, K
            )
        else:
            # Use 3D bbox and previous frame's pose to yield current frame 2D bbox:
            previous_frame_pose, inliers_crop = pred_poses[id - 1]

            if len(inliers_crop) < 20:
                logger.warning(
                    f"Re-Running local feature object detector for frame {id}"
                )
                # Consider previous pose estimation failed, reuse local feature object detector:
                (
                    _,
                    query_crop,
                    K_crop,
                    transform,
                ) = local_feature_obj_detector.detect(query_image, query_image_path, K)
            else:
                (
                    _,
                    query_crop,
                    K_crop,
                    transform,
                ) = local_feature_obj_detector.previous_pose_detect(
                    query_image_path, K, previous_frame_pose, bbox3d
                )
        data.update({"query_image": query_crop.cuda()})

        # Perform keypoint-free 2D-3D matching and then estimate object pose of query image by PnP:
        with torch.no_grad():
            match_2D_3D_model(data)
        mkpts_3d = data["mkpts_3d_db"].cpu().numpy()  # N*3
        mkpts_crop = data["mkpts_query_f"].cpu().numpy()  # N*2
        pose_pred, _, inliers_crop, _ = metric_utils.ransac_PnP(
            K_crop,
            mkpts_crop,
            mkpts_3d,
            scale=1000,
            pnp_reprojection_error=7,
            img_hw=[512, 512],
            use_pycolmap_ransac=True,
        )
        logger.debug(f"Pose estimation inliers: {len(inliers_crop)} for frame {id}")
        pred_poses[id] = [pose_pred, inliers_crop]
        mkpts_cache.append(
            MKPT(
                K_crop=K_crop,
                transform=transform,
                mkpts_3d=mkpts_3d,  # load all 3D keypoints
                mkpts_crop=mkpts_crop,  # load all 2D keypoints
                inliers_crop=inliers_crop,
            )
        )

        # visualise the detected keypoints on the 3D pointcloud and the query image.
        # ransac_PnP operates on these sets of keypoints
        if cfg.DBG:
            vis_utils.visualize_2D_3D_keypoints(
                path=Path(f"temp/{local_feature_obj_detector.query_id}"),
                data=data,
                inp_crop=query_crop,
                inliers=inliers_crop,
                mkpts_3d=mkpts_3d,
                mkpts_query=mkpts_crop,
            )

        # Visualize:
        vis_utils.save_demo_image(
            pose_pred,
            K,
            image_path=query_image_path,
            box3d=bbox3d,
            draw_box=len(inliers_crop) > 20,
            save_path=osp.join(paths["vis_box_dir"], f"{id}.jpg"),
        )

    # Output video to visualize estimated poses:
    pose_estimation_demo_video: str = f"temp/demo_pose_{seq_dir.split('/')[-1]}.mp4"
    logger.info(f"POSE ESTIMATION DEMO VIDEO SAVED TO: {pose_estimation_demo_video}")
    vis_utils.make_video(paths["vis_box_dir"], pose_estimation_demo_video)

    ####################################################################################################
    ####################################################################################################

    logger.error("Running CoTracker")
    # save/load mkpts_cache in case of debugging the tracker exclusively
    # import pickle
    # with open("mkpts.pkl", "wb") as f:
    #     pickle.dump(mkpts_cache, f)
    # with open("mkpts.pkl", "rb") as f:
    #     mkpts_cache = pickle.load(f)

    temp_thresh: int = 5  # time horizon for tracking & initialization phase
    tracker: CoTrackerPredictor = CoTrackerPredictor(
        checkpoint="submodules/CoTracker/checkpoints/cotracker2.pth"
    )
    tracker = tracker.cuda()
    video = read_video_from_path(f"{seq_dir}/clip.mp4")
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    queries: torch.Tensor = torch.empty(0, 3).cuda()

    for frame_id in tqdm(range(len(mkpts_cache))):
        query_image_path = dataset[frame_id]["query_image_path"]
        # don't optimize frames in initialization phase, use previous pose estimation
        if frame_id < temp_thresh:
            vis_utils.save_demo_image(
                pred_poses[frame_id][0],
                K,  # re-use K from previous pose estimation loop
                image_path=query_image_path,
                box3d=bbox3d,
                draw_box=len(pred_poses[frame_id][1]) > 20,
                save_path=osp.join("temp/optimized", f"{frame_id}.jpg"),
                color="r",
            )
            continue

        mkpt: MKPT = mkpts_cache[frame_id]
        mkpts_3d_previous = np.empty((0, 3))
        i = 0
        for _mkpt in mkpts_cache[frame_id - temp_thresh : frame_id]:
            # combine previous inliers into query tensor
            _queries = _mkpt.queries
            _queries[:, 0] = i
            queries = torch.cat((queries, _mkpt.queries), dim=0)

            # combine previous 3D keypoints into 3D keypoints array
            _mkpts_3d = _mkpt.mkpts_3d[_mkpt.inliers_crop]
            mkpts_3d_previous = np.append(mkpts_3d_previous, _mkpts_3d, axis=0)

            i += 1

        assert (
            queries.shape[0] == mkpts_3d_previous.shape[0]
        ), "The number of query points (= 2D kpts) and 3D kpts should match!"

        # track the queries a.k.a previous inliers to current frame
        _video = video[:, frame_id - temp_thresh : frame_id]
        _video = _video.cuda()  # load sliced video into cuda memory
        _tracks, _visibility = tracker(
            _video, queries=queries[None], backward_tracking=True
        )

        # map predicted tracks from coords wrt original image to current cropped img
        transform_tensor = torch.from_numpy(mkpt.transform).float().cuda()
        ones = torch.ones(*_tracks.shape[:-1], 1, device="cuda")
        pred_tracks_orig_homo = torch.cat([_tracks, ones], dim=-1).float()
        pred_tracks_query_homo = torch.einsum(
            "...ij,...j->...i", transform_tensor, pred_tracks_orig_homo
        )
        pred_tracks_query = (
            pred_tracks_query_homo[..., :2] / pred_tracks_query_homo[..., 2:3]
        )

        # inject tracked these key points before the ransac PnP step
        mkpts_crop_tracked = pred_tracks_query[0][-1].cpu().numpy()
        mkpts_crop_injected = np.concatenate(
            (mkpt.mkpts_crop, mkpts_crop_tracked), axis=0
        )
        mkpts_3d_injected = np.concatenate((mkpt.mkpts_3d, mkpts_3d_previous), axis=0)

        # now, re-run ransac_PnP w/ additional key-points
        pose_pred_optimized, _, inliers_optimized, _ = metric_utils.ransac_PnP(
            mkpt.K_crop,
            mkpts_crop_injected,
            mkpts_3d_injected,
            scale=1000,
            pnp_reprojection_error=7,
            img_hw=[512, 512],
            use_pycolmap_ransac=True,
        )

        logger.debug(
            f"Pose estimation inliers: {len(inliers_optimized)} for frame {frame_id}"
        )
        
        # TODO: some numerical analysis if the optimized pose is better than before.. 
        
        
        # Visualize:
        vis_utils.save_demo_image(
            pose_pred_optimized,
            K,  # re-use K from previous pose estimation loop
            image_path=query_image_path,
            box3d=bbox3d,
            draw_box=len(inliers_optimized) > 20,
            save_path=osp.join("temp/optimized", f"{frame_id}.jpg"),
            color="r",
        )

        if False:  # skip tracking visualisation for now
            vis = Visualizer(
                save_dir="temp/", linewidth=1, mode="cool", tracks_leave_trace=-1
            )
            vis.visualize(
                video=video[:, frame_id - temp_thresh : frame_id],
                tracks=_tracks,
                visibility=_visibility,
                filename=f"demo_cotracker_{seq_dir.split('/')[-1]}_{frame_id}",
            )
            logger.info(
                f"Cotracker demo vido saved to: temp/demo_cotracker_{seq_dir.split('/')[-1]}_{frame_id}.mp4"
            )

        # reset queries for next frame
        queries = torch.empty(0, 3).cuda()

        # Output video to visualize estimated poses:
    optimized_pose_estimation_demo_video: str = (
        f"temp/demo_optimized_pose_{seq_dir.split('/')[-1]}.mp4"
    )
    logger.info(
        f"POSE ESTIMATION DEMO VIDEO SAVED TO: {optimized_pose_estimation_demo_video}"
    )
    vis_utils.make_video("temp/optimized", optimized_pose_estimation_demo_video)

    ####################################################################################################
    ####################################################################################################
    return


def main() -> None:
    # loop over all data_dirs specified in the CONFIG class
    for test_dir in tqdm(cfg.data_dirs, total=len(cfg.data_dirs)):
        seq_dir = osp.join(cfg.data_root, test_dir)
        logger.info(f"Eval {seq_dir}")
        inference_core(seq_dir)

    logger.info("Done")
    return

if __name__ == "__main__":
    global cfg
    cfg: CONFIG = CONFIG()
    os.system(f"rm -rf temp/*")
    Path("temp/optimized").mkdir(exist_ok=True, parents=True)
    if cfg.DBG:
        Path("temp/debug").mkdir(exist_ok=True, parents=True)
    main()
    # remove the temp/optimized directory, just keep the video (more efficienc scp to loacl machine)
    os.system(f"rm -rf temp/optimized")
