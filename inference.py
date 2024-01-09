from typing import List
import os.path as osp
from tqdm import tqdm
from loguru import logger
import yaml
import demo
import os
from pathlib import Path
import pickle
import numpy as np
import torch

from src.utils import data_utils, vis_utils, metric_utils, data_io
from src.utils.optimization_utils import MKPT
from src.datasets.OnePosePlus_inference_dataset import OnePosePlusInferenceDataset
from src.inference import inference_OnePosePlus
from src.local_feature_object_detector.local_feature_2D_detector import (
    LocalFeatureObjectDetector,
)
from src.models.OnePosePlus.OnePosePlusModel import OnePosePlus_model

# CoTracker
from submodules.CoTracker.cotracker.predictor import CoTrackerPredictor
from submodules.CoTracker.cotracker.utils.visualizer import Visualizer


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
        self.data_dirs: List[str] = ["asus_03-test"]
        self.sfm_model_dir: str = (
            f"{self.data_root}/sfm_model/outputs_softmax_loftr_loftr/{self.obj_name}"
        )
        self.datamodule = CONFIG.DATAMODULE()
        self.model: dict = self._get_model()
        self.DBG: bool = False

        # pose estimation optimization meta parameters
        self.temp_thresh: int = 5  # time horizon for tracking & initialization phase
        self.inliers_only: bool = (
            False  # use only the inliers for tracking OR all previous key points
        )

    def _get_model(self) -> dict:
        with open("configs/experiment/inference_demo.yaml", "r") as f:
            onepose_config = yaml.load(f, Loader=yaml.FullLoader)
        return onepose_config["model"]


####################################################################################################
##########################TODO: refactor helper methods later ######################################
####################################################################################################


####################################################################################################
####################################################################################################


def inference_core(seq_dir):
    """Inference core function for OnePosePlus."""
    logger.warning(f"Running inference on {seq_dir}")
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
    test_dir: str = seq_dir.split("/")[-1]
    Path("temp/original_pose_predictions").mkdir(exist_ok=True, parents=True)
    # only run pose estimation if cache does not exist
    if not Path(f"cache_{test_dir}.pkl").exists():
        logger.warning("running pose estimation...")
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
                    ) = local_feature_obj_detector.detect(
                        query_image, query_image_path, K
                    )
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
                    K=K,
                    transform=transform,
                    mkpts_3d=mkpts_3d,  # load all 3D key points
                    mkpts_crop=mkpts_crop,  # load all 2D key points
                    inliers=inliers_crop,
                )
            )

            # visualise the detected key points on the 3D pointcloud and the query image.
            # ransac_PnP operates on these sets of key points
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
                save_path=f"temp/original_pose_predictions/{id}.jpg",
            )

        # Output video to visualize estimated poses:
        vis_utils.make_video(
            "temp/original_pose_predictions", f"temp/demo_pose_{test_dir}.mp4"
        )

        # store variables to disk to debug cotracker separately
        cache = {"mkpts_cache": mkpts_cache, "pred_poses": pred_poses}
        with open(f"cache_{test_dir}.pkl", "wb") as f:
            pickle.dump(cache, f)

    os.system(f"rm -rf temp/original_pose_predictions")
    ####################################################################################################
    ####################################################################################################

    logger.warning("optimizing pose estimation...")
    Path("temp/optimized_pose_predictions").mkdir(exist_ok=True, parents=True)
    # load required variables from disk...
    with open(f"cache_{test_dir}.pkl", "rb") as f:
        cache = pickle.load(f)
    mkpts_cache = cache["mkpts_cache"]
    pred_poses = cache["pred_poses"]
    tracker: CoTrackerPredictor = CoTrackerPredictor(
        checkpoint="submodules/CoTracker/checkpoints/cotracker2.pth"
    )
    tracker = tracker.cuda()
    video = data_io.read_video_from_path(Path(f"{seq_dir}/color_full"))
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    queries: torch.Tensor = torch.empty(0, 3).cuda()

    for frame_id in tqdm(range(len(mkpts_cache))):
        mkpt: MKPT = mkpts_cache[frame_id]
        query_image_path = dataset[frame_id]["query_image_path"]
        # don't optimize frames in initialization phase, use previous pose estimation
        if frame_id < cfg.temp_thresh:
            vis_utils.save_demo_image(
                pred_poses[frame_id][0],
                mkpt.K_orig,  # re-use K from previous pose estimation loop
                image_path=query_image_path,
                box3d=bbox3d,
                draw_box=len(pred_poses[frame_id][1]) > 20,
                save_path=f"temp/optimized_pose_predictions/{frame_id}.jpg",
                color="r",
            )
            continue
        mkpt.set_images(img_path=query_image_path)
        mkpts_3d_previous = np.empty((0, 3))
        i = 0
        for _mkpt in mkpts_cache[frame_id - cfg.temp_thresh : frame_id]:
            # combine previous inliers into query tensor
            _queries = _mkpt.get_queries(cfg.inliers_only)
            _queries[
                :, 0
            ] = i  # replace frame id wrt original video to frame id wrt sliced video)
            queries = torch.cat((queries, _queries), dim=0)

            # combine previous 3D key points into 3D key points array
            _mkpts_3d = (
                _mkpt.mkpts_3d[_mkpt.inliers] if cfg.inliers_only else _mkpt.mkpts_3d
            )
            mkpts_3d_previous = np.append(mkpts_3d_previous, _mkpts_3d, axis=0)

            i += 1

        assert (
            queries.shape[0] == mkpts_3d_previous.shape[0]
        ), "The number of query points (= 2D kpts) and 3D kpts should match!"

        # track the queries a.k.a previous inliers to current frame
        sliced_video = video[:, frame_id - cfg.temp_thresh : frame_id + 1]
        sliced_video = sliced_video.cuda()  # load sliced video into cuda memory
        assert (
            sliced_video.shape[1] == cfg.temp_thresh + 1
        ), "Sliced video should temp_thresh frames +1 (current) frame"
        assert np.array_equal(
            mkpt.img_orig, sliced_video[0, -1].permute(1, 2, 0).cpu().numpy()
        ), "The last frame of sliced video should be the same as the original image of the current frame"

        _tracks, _visibility = tracker(
            sliced_video, queries=queries[None], backward_tracking=True
        )

        # extract the tracked key points from the last frame
        mkpt.set_new_mkpts(_tracks)

        # inject the tracked key points before the ransac PnP step
        mkpts_2d_injected = np.concatenate(
            (mkpt.mkpts_crop, mkpt.mkpts_crop_tracked), axis=0
        )
        mkpts_3d_injected = np.concatenate((mkpt.mkpts_3d, mkpts_3d_previous), axis=0)

        # now, re-run ransac_PnP w/ additional key-points
        pose_pred_optimized, _, inliers_optimized, _ = metric_utils.ransac_PnP(
            mkpt.K_crop,
            mkpts_2d_injected,
            mkpts_3d_injected,
            scale=1000,
            pnp_reprojection_error=7,
            img_hw=[512, 512],
            use_pycolmap_ransac=True,
        )

        mkpt.set_new_inliers(inliers_optimized)

        # TODO: some numerical analysis if the optimized pose is better than before..

        # Visualize:
        vis_utils.save_demo_image(
            pose_pred_optimized,
            mkpt.K_orig,
            image_path=query_image_path,
            box3d=bbox3d,
            draw_box=len(inliers_optimized) > 20,
            save_path=f"temp/optimized_pose_predictions/{frame_id}.jpg",
            color="r",
        )
        # save some debug frames and short tracking videos if DBG is True
        if cfg.DBG and frame_id % 10 == 0:
            tmp_path: Path = Path(f"temp/frame_{frame_id}")
            tmp_path.mkdir(exist_ok=True, parents=True)
            Path(tmp_path / "orig").mkdir(exist_ok=True, parents=True)
            Path(tmp_path / "crop").mkdir(exist_ok=True, parents=True)

            mkpt.debug(out_path=tmp_path)

            vis = Visualizer(
                save_dir=f"temp/frame_{frame_id}",
                linewidth=1,
                mode="cool",
                tracks_leave_trace=-1,
            )
            vis.visualize(
                video=sliced_video[:, :, [2, 1, 0], :, :],  # BGR -> RGB
                tracks=_tracks,
                visibility=_visibility,
                filename=f"tracked_seq",
            )

        # reset queries for next frame
        queries = torch.empty(0, 3).cuda()

    vis_utils.make_video(
        "temp/optimized_pose_predictions", f"temp/demo_optimized_pose_{test_dir}.mp4"
    )
    os.system(f"rm -rf temp/optimized_pose_predictions")
    return


def main() -> None:
    # loop over all data_dirs specified in the CONFIG class
    for test_dir in cfg.data_dirs:
        seq_dir = osp.join(cfg.data_root, test_dir)
        inference_core(seq_dir)

    logger.warning("Done")
    return


if __name__ == "__main__":
    global cfg
    cfg: CONFIG = CONFIG()
    os.system(f"rm -rf temp/*")
    if cfg.DBG:
        Path("temp/debug").mkdir(exist_ok=True, parents=True)
    main()
