from pathlib import Path
import cv2
import torch
import numpy as np
from typing import List, Tuple


class MKPT:
    def __init__(self, K_crop, K, transform, mkpts_3d, mkpts_crop, inliers):
        """
        K_orig          (3,3)   camera intrinsics matrix of original image
        K_crop          (3,3)   camera intrinsics matrix of cropped image

        transform       (3,3)   transform matrix from original image to cropped image
        inliers         (M,1)   list of indices corresponding to inliers in mkpts_crop and mkpts_3d

        mkpts_crop      (N,2)   2D key points in cropped image
        mkpts_orig      (N,2)   2D key points in original image
        mkpts_3d        (N,3)   3D key points in pointcloud

        inliers_crop    (M,2)   inliers in cropped image
        inliers_orig    (M,2)   inliers in original image

        img_orig        (H,W,3) original image
        img_crop        (H,W,3) cropped image

        mkpts_orig_tracked  (N,2)   tracked key points in original image
        mkpts_crop_tracked  (N,2)   tracked key points in cropped image
        inliers_crop_optimised (M,2)   inliers in cropped image after using mkpts_crop_tracked
        inliers_orig_optimised (M,2)   inliers in original image after using mkpts_crop_tracked
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

    def get_queries(self, inliers_only: bool = False) -> torch.Tensor:
        """Convert the inliers to the format expected by CoTracker."""
        if inliers_only:
            queries = torch.from_numpy(
                np.c_[np.ones(self.inliers_orig.shape[0]), self.inliers_orig]
            )
        else:
            queries = torch.from_numpy(
                np.c_[np.ones(self.mkpts_orig.shape[0]), self.mkpts_orig]
            )
        return queries.float().cuda()

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
        self, img_path: str, mkpts: np.ndarray, color: tuple = (0, 255, 0), r: int = 1
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

    def _save_img_w_mkpts(
        self,
        img_path: Path,
        mkpts,
        save_path: Path,
        color: Tuple[int, int, int],
        r: int = 1,
    ):
        """
        img_path    Path        path to image onto which the key points are plotted
        mkpts       np.ndarray  key points to plot
        save_path   Path        new image path
        color       tuple       color of the key points
        r           int         radius of the key points
        """
        img_cv2 = cv2.imread(str(img_path))

        for point in mkpts:
            x, y = point.astype(int)
            cv2.circle(img_cv2, (x, y), r, color, -1)

        cv2.imwrite(str(save_path), img_cv2)

    def debug(self, out_path: Path) -> None:
        blue, green, red = (255, 0, 0), (0, 255, 0), (0, 0, 255) # BGR

        query_orig_path: Path = out_path / "orig" / "00_query_orig.png"
        query_crop_path: Path = out_path / "crop" / "00_query_crop.png"
        query_orig_mkpts_old_path: Path = out_path / "orig" / "01_mkpts_old.png"
        query_crop_mkpts_old_path: Path = out_path / "crop" / "01_mkpts_old.png"
        query_orig_inliers_path: Path = out_path / "orig" / "02_inliers_old.png"
        query_crop_inliers_path: Path = out_path / "crop" / "02_inliers_old.png"

        # save original and cropped image for reference
        cv2.imwrite(str(query_orig_path), self.img_orig)
        cv2.imwrite(str(query_crop_path), self.img_crop)

        self._save_img_w_mkpts(
            img_path=query_crop_path,
            mkpts=self.mkpts_crop,
            color=blue,
            save_path=query_crop_mkpts_old_path,
        )

        self._save_img_w_mkpts(
            img_path=query_orig_path,
            mkpts=self.mkpts_orig,
            color=blue,
            save_path=query_orig_mkpts_old_path,
        )

        self._save_img_w_mkpts(
            img_path=query_orig_path,
            mkpts=self.map_to_original(self.mkpts_crop[self.inliers]),
            color=green,
            save_path=query_orig_inliers_path,
        )

        self._save_img_w_mkpts(
            img_path=query_crop_path,
            mkpts=self.mkpts_crop[self.inliers],
            color=green,
            save_path=query_crop_inliers_path,
        )

        self._save_img_w_mkpts(
            img_path=query_crop_mkpts_old_path,
            mkpts=self.mkpts_crop[self.inliers],
            color=green,
            save_path=out_path / "crop" / "02_mkpts_inliers_old.png",
        )

        self._save_img_w_mkpts(
            img_path=query_orig_mkpts_old_path,
            mkpts=self.mkpts_orig_tracked,
            color=red,
            save_path=out_path / "orig" / "03_mkpts_new.png",
        )

        self._save_img_w_mkpts(
            img_path=query_orig_mkpts_old_path,
            mkpts=self.map_to_original(self.mkpts_crop[self.inliers]),
            color=green,
            save_path=out_path / "orig" / "02_mkpts_inliers_old.png",
        )

        self._save_img_w_mkpts(
            img_path=query_crop_mkpts_old_path,
            mkpts=self.mkpts_crop_tracked,
            color=red,
            save_path=out_path / "crop" / "03_mkpts_new.png",
        )

        self._save_img_w_mkpts(
            img_path=query_orig_inliers_path,
            mkpts=self.inliers_orig_optimised,
            color=red,
            save_path=out_path / "orig" / "04_inliers_new.png",
        )

        self._save_img_w_mkpts(
            img_path=query_crop_inliers_path,
            mkpts=self.inliers_crop_optimised,
            color=red,
            save_path=out_path / "crop" / "04_inliers_new.png",
        )
