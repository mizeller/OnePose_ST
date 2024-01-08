from typing import List
import os.path as osp
from tqdm import tqdm
from loguru import logger
import yaml
import demo
import os
from pathlib import Path
import pickle
import cv2
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
    def __init__(self, K_crop, K, transform, mkpts_3d, mkpts_crop, inliers):
        """
        K_orig          (3,3)   camera intrinsics matrix of original image
        K_crop          (3,3)   camera intrinsics matrix of cropped image

        transform       (3,3)   transform matrix from original image to cropped image

        mkpts_3d        (N,3)   3D key points in pointcloud

        mkpts_crop      (N,2)   2D key points in cropped image
        mkpts_orig      (N,2)   2D key points in original image

        inliers         (M,1)   list of indices corresponding to inliers in mkpts_crop and mkpts_3d

        inliers_crop    (M,2)   inliers in cropped image
        inliers_orig    (M,2)   inliers in original image

        queries         (M,3)   self.inliers_orig in format expected by CoTracker
        """
        # set initially
        self.K_orig = K
        self.K_crop = K_crop
        self.transform = transform
        self.inliers = inliers
        self.mkpts_crop = mkpts_crop
        self.mkpts_orig = self.map_to_original(self.mkpts_crop)
        self.mkpts_3d = mkpts_3d
        self.inliers_crop = mkpts_crop[inliers]
        self.inliers_orig = self.map_to_original(self.inliers_crop)

        # set later
        self.img_orig = None
        self.img_crop = None
        self.mkpts_orig_tracked = None
        self.mkpts_crop_tracked = None
        self.inliers_crop_optimised = None
        self.inliers_orig_optimised = None

    def get_queries(self):
        """Convert the inliers to the format expected by CoTracker."""
        queries = (
            torch.from_numpy(np.c_[np.ones(self.mkpts_orig.shape[0]), self.mkpts_orig])
            .float()
            .cuda()
        )
        return queries

    def set_images(self, img_path: str = "") -> None:
        img_orig = cv2.imread(img_path)
        img_crop = cv2.warpAffine(
            img_orig, self.transform[:2, :], (512, 512), flags=cv2.INTER_LINEAR
        )
        self.img_orig = img_orig
        self.img_crop = img_crop
        return

    def set_new_mkpts(self, predicted_tracks: torch.Tensor) -> None:
        """Extract the new key points in the current frame from the predicted tracks."""
        self.mkpts_orig_tracked = np.squeeze(
            predicted_tracks[:, -1].cpu().numpy(), axis=0
        )
        self.mkpts_crop_tracked = self.map_to_crop(self.mkpts_orig_tracked)

    def set_new_inliers(self, inliers_optimised: List[int]) -> None:
        # concat the old mkpts with the tracked mkpts
        mkpts_injected = np.concatenate(
            (self.mkpts_crop, self.mkpts_crop_tracked), axis=0
        )
        self.inliers_crop_optimised = mkpts_injected[inliers_optimised]
        self.inliers_orig_optimised = self.map_to_original(self.inliers_crop_optimised)

    def _get_img_w_mkpts(
        self, img_path: str, mkpts: np.ndarray, color: tuple = (0, 255, 0), r: int = 3
    ):
        query_image = cv2.imread(img_path)
        for point in mkpts:
            x, y = point.astype(int)
            cv2.circle(query_image, (x, y), r, color, -1)
        return query_image

    def map_to_original(self, mkpts_crop):
        """Map points on the cropped image to the original image."""
        inv_transform = np.linalg.inv(self.transform)
        mkpts_crop_homogeneous = np.c_[mkpts_crop, np.ones(mkpts_crop.shape[0])]
        mkpts_orig_homogeneous = np.dot(mkpts_crop_homogeneous, inv_transform.T)
        mkpts_orig = (
            mkpts_orig_homogeneous[:, :2] / mkpts_orig_homogeneous[:, 2, np.newaxis]
        )
        return mkpts_orig

    def map_to_crop(self, mkpts_orig):
        """Map points on the original image to the cropped image."""
        mkpts_orig_homogeneous = np.c_[mkpts_orig, np.ones(mkpts_orig.shape[0])]
        mkpts_crop_homogeneous = np.dot(mkpts_orig_homogeneous, self.transform.T)
        mkpts_crop = (
            mkpts_crop_homogeneous[:, :2] / mkpts_crop_homogeneous[:, 2, np.newaxis]
        )
        return mkpts_crop

    def debug(self, out_path: Path) -> None:
        blue, green, red = (255, 0, 0), (0, 255, 0), (0, 0, 255)

        # img paths, that are save in this method
        query_orig_path: str = str(out_path / "orig" / "00_query_orig.png")
        query_crop_path: str = str(out_path / "crop" / "00_query_crop.png")
        query_orig_mkpts_old_path: str = str(
            out_path / "orig" / "01_query_orig_mkpts_old.png"
        )
        query_crop_mkpts_old_path: str = str(
            out_path / "crop" / "01_query_crop_mkpts_old.png"
        )
        query_orig_mkpts_inliers_path: str = str(
            out_path / "orig" / "02_query_orig_mkpts_inliers_old.png"
        )
        query_crop_mkpts_inliers_path: str = str(
            out_path / "orig" / "02_query_crop_mkpts_inliers_old.png"
        )
        query_orig_inliers_path: str = str(
            out_path / "orig" / "02_query_orig_inliers_old.png"
        )
        query_crop_inliers_path: str = str(
            out_path / "crop" / "02_query_crop_inliers_old.png"
        )
        query_orig_mkpts_new_path: str = str(
            out_path / "orig" / "03_query_orig_mkpts_old.png"
        )
        query_crop_mkpts_new_path: str = str(
            out_path / "crop" / "03_query_crop_mkpts_old.png"
        )
        query_orig_inliers_new_path: str = str(
            out_path / "orig" / "04_query_orig_inliers_old.png"
        )
        query_crop_inliers_new_path: str = str(
            out_path / "crop" / "04_query_crop_inliers_old.png"
        )

        # save original and cropped image
        cv2.imwrite(query_orig_path, self.img_orig)
        cv2.imwrite(query_crop_path, self.img_crop)

        # plot old key points in blue
        query_orig_mkpts_old = self._get_img_w_mkpts(
            img_path=query_orig_path, mkpts=self.mkpts_orig, color=blue
        )
        query_crop_mkpts_old = self._get_img_w_mkpts(
            img_path=query_crop_path, mkpts=self.mkpts_crop, color=blue
        )
        cv2.imwrite(query_orig_mkpts_old_path, query_orig_mkpts_old)
        cv2.imwrite(query_crop_mkpts_old_path, query_crop_mkpts_old)

        # plot old inliers in green
        query_orig_inliers_old = self._get_img_w_mkpts(
            img_path=query_orig_path,
            mkpts=self.map_to_original(self.mkpts_crop[self.inliers]),
            color=green,
        )
        query_orig_mkpts_inliers_old = self._get_img_w_mkpts(
            img_path=query_orig_mkpts_old_path,
            mkpts=self.map_to_original(self.mkpts_crop[self.inliers]),
            color=green,
        )
        query_crop_inliers_old = self._get_img_w_mkpts(
            img_path=query_crop_path, mkpts=self.mkpts_crop[self.inliers], color=green
        )
        query_crop_mkpts_inliers_old = self._get_img_w_mkpts(
            img_path=query_crop_mkpts_old_path,
            mkpts=self.mkpts_crop[self.inliers],
            color=green,
        )
        # inliers only
        cv2.imwrite(query_orig_inliers_path, query_orig_inliers_old)
        cv2.imwrite(query_crop_inliers_path, query_crop_inliers_old)
        # inliers + old key points
        cv2.imwrite(
            query_orig_mkpts_inliers_path,
            query_orig_mkpts_inliers_old,
        )
        cv2.imwrite(
            query_crop_mkpts_inliers_path,
            query_crop_mkpts_inliers_old,
        )

        # plot the injected key points in red, onto the image w/ the original key points in blue
        query_orig_mkpts_new = self._get_img_w_mkpts(
            img_path=query_orig_mkpts_old_path,
            mkpts=self.mkpts_orig_tracked,
            color=red,
        )
        cv2.imwrite(query_orig_mkpts_new_path, query_orig_mkpts_new)
        query_crop_mkpts_new = self._get_img_w_mkpts(
            img_path=query_crop_mkpts_old_path,
            mkpts=self.mkpts_crop_tracked,
            color=red,
        )
        cv2.imwrite(query_crop_mkpts_new_path, query_crop_mkpts_new)

        # plot the optimized inliers in red, onto the image w/ the original inliers in green
        query_orig_inliers_new = self._get_img_w_mkpts(
            img_path=query_orig_inliers_path,
            mkpts=self.inliers_orig_optimised,
            color=red,
        )
        cv2.imwrite(query_orig_inliers_new_path, query_orig_inliers_new)
        query_crop_inliers_new = self._get_img_w_mkpts(
            img_path=query_crop_inliers_path,
            mkpts=self.inliers_crop_optimised,
            color=red,
        )
        cv2.imwrite(query_crop_inliers_new_path, query_crop_inliers_new)


def read_video_from_path(directory):
    """
    Reads all images from a directory and returns a numpy array of shape (n_frames, height, width, 3)
    NOTE: the directory must contain ONLY images.
    """
    frames = []
    for filename in sorted(Path(directory).iterdir()):
        img = cv2.imread(str(filename))
        frames.append(img)

    return np.stack(frames)


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
    if False:
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
                save_path=osp.join(paths["vis_box_dir"], f"{id}.jpg"),
            )

        # Output video to visualize estimated poses:
        pose_estimation_demo_video: str = f"temp/demo_pose_{seq_dir.split('/')[-1]}.mp4"
        logger.info(
            f"POSE ESTIMATION DEMO VIDEO SAVED TO: {pose_estimation_demo_video}"
        )
        vis_utils.make_video(paths["vis_box_dir"], pose_estimation_demo_video)

        # store variables to disk to debug cotracker separately
        cache = {"mkpts_cache": mkpts_cache, "pred_poses": pred_poses}
        with open("cache.pkl", "wb") as f:
            pickle.dump(cache, f)

    ####################################################################################################
    ####################################################################################################

    logger.error("Running CoTracker")
    # load required variables from disk...
    with open("cache.pkl", "rb") as f:
        cache = pickle.load(f)
    mkpts_cache = cache["mkpts_cache"]
    pred_poses = cache["pred_poses"]

    temp_thresh: int = 5  # time horizon for tracking & initialization phase
    tracker: CoTrackerPredictor = CoTrackerPredictor(
        checkpoint="submodules/CoTracker/checkpoints/cotracker2.pth"
    )
    tracker = tracker.cuda()
    # NOTE: this assumes there is a clip.mp4 in the sequence directory, which corresponds to the frames
    # in the color_full directory; maybe change code, to use the images from the color_full directory instead...
    video = read_video_from_path(Path(f"{seq_dir}/color_full"))
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    queries: torch.Tensor = torch.empty(0, 3).cuda()

    for frame_id in tqdm(range(len(mkpts_cache))):
        mkpt: MKPT = mkpts_cache[frame_id]
        query_image_path = dataset[frame_id]["query_image_path"]
        # don't optimize frames in initialization phase, use previous pose estimation
        if frame_id < temp_thresh:
            vis_utils.save_demo_image(
                pred_poses[frame_id][0],
                mkpt.K_orig,  # re-use K from previous pose estimation loop
                image_path=query_image_path,
                box3d=bbox3d,
                draw_box=len(pred_poses[frame_id][1]) > 20,
                save_path=f"temp/optimized/{frame_id}.jpg",
                color="r",
            )
            continue
        mkpt.set_images(img_path=query_image_path)
        mkpts_3d_previous = np.empty((0, 3))
        i = 0
        for _mkpt in mkpts_cache[frame_id - temp_thresh : frame_id]:
            # combine previous inliers into query tensor
            _queries = _mkpt.get_queries()
            _queries[
                :, 0
            ] = i  # replace frame id wrt original video to frame id wrt sliced video)
            queries = torch.cat((queries, _queries), dim=0)

            # combine previous 3D key points into 3D key points array
            _mkpts_3d = _mkpt.mkpts_3d
            mkpts_3d_previous = np.append(mkpts_3d_previous, _mkpts_3d, axis=0)

            i += 1

        assert (
            queries.shape[0] == mkpts_3d_previous.shape[0]
        ), "The number of query points (= 2D kpts) and 3D kpts should match!"

        # track the queries a.k.a previous inliers to current frame
        sliced_video = video[:, frame_id - temp_thresh : frame_id + 1]
        sliced_video = sliced_video.cuda()  # load sliced video into cuda memory
        assert (
            sliced_video.shape[1] == temp_thresh + 1
        ), "Sliced video should temp_thresh frames +1 (current) frame"
        assert np.array_equal(
            mkpt.img_orig, sliced_video[0, -1].permute(1, 2, 0).cpu().numpy()
        ), "The first frame of sliced video should be the same as the original image"

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

        logger.debug(
            f"Pose estimation inliers: {len(inliers_optimized)} for frame {frame_id}"
        )

        # TODO: some numerical analysis if the optimized pose is better than before..

        # Visualize:
        vis_utils.save_demo_image(
            pose_pred_optimized,
            mkpt.K_orig,
            image_path=query_image_path,
            box3d=bbox3d,
            draw_box=len(inliers_optimized) > 20,
            save_path=f"temp/optimized/{frame_id}.jpg",
            color="r",
        )

        if frame_id % 10 == 0:
            tmp_path: Path = Path(f"temp/frame_{frame_id}")
            tmp_path.mkdir(exist_ok=True, parents=True)
            Path(tmp_path / "orig").mkdir(exist_ok=True, parents=True)
            Path(tmp_path / "crop").mkdir(exist_ok=True, parents=True)  

            mkpt.debug(out_path=tmp_path)
            # TODO: somehow the video is BGR instead of RGB - fix this!
            vis = Visualizer(
                save_dir=f"temp/frame_{frame_id}",
                linewidth=1,
                mode="cool",
                tracks_leave_trace=-1,
            )
            vis.visualize(
                video=sliced_video,
                tracks=_tracks,
                visibility=_visibility,
                filename=f"tracked_seq",
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
