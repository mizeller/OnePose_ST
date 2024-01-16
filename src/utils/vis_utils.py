import cv2
import numpy as np
import torch
from pathlib import Path
import natsort
import os
from loguru import logger
from wis3d import Wis3D as Vis3D
import open3d as o3d


def reproj(K, pose, pts_3d):
    """
    Reproj 3d points to 2d points
    @param K: [3, 3] or [3, 4]
    @param pose: [3, 4] or [4, 4]
    @param pts_3d: [n, 3]
    """
    assert K.shape == (3, 3) or K.shape == (3, 4)
    assert pose.shape == (3, 4) or pose.shape == (4, 4)

    if K.shape == (3, 3):
        K_homo = np.concatenate([K, np.zeros((3, 1))], axis=1)
    else:
        K_homo = K

    if pose.shape == (3, 4):
        pose_homo = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)
    else:
        pose_homo = pose

    pts_3d = pts_3d.reshape(-1, 3)
    pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1))], axis=1)
    pts_3d_homo = pts_3d_homo.T

    reproj_points = K_homo @ pose_homo @ pts_3d_homo
    reproj_points = reproj_points[:] / reproj_points[2:]
    reproj_points = reproj_points[:2, :].T
    return reproj_points  # [n, 2]


def draw_3d_box(image, corners_2d, linewidth=3, color="g"):
    """Draw 3d box corners
    @param corners_2d: [8, 2]
    """
    lines = np.array(
        [[0, 1, 5, 4, 2, 3, 7, 6, 0, 1, 5, 4], [1, 5, 4, 0, 3, 7, 6, 2, 3, 2, 6, 7]]
    ).T

    colors = {"g": (0, 255, 0), "r": (0, 0, 255), "b": (255, 0, 0)}
    if color not in colors.keys():
        color = (42, 97, 247)
    else:
        color = colors[color]

    for id, line in enumerate(lines):
        pt1 = corners_2d[line[0]].astype(int)
        pt2 = corners_2d[line[1]].astype(int)
        image = cv2.line(image, tuple(pt1), tuple(pt2), color, linewidth)

    return image


def draw_2d_box(image, corners_2d, linewidth=3):
    """Draw 2d box corners
    @param corners_2d: [x_left, y_top, x_right, y_bottom]
    """
    x1, y1, x2, y2 = corners_2d.astype(int)
    box_pts = [
        [(x1, y1), (x1, y2)],
        [(x1, y2), (x2, y2)],
        [(x2, y2), (x2, y1)],
        [(x2, y1), (x1, y1)],
    ]

    for pts in box_pts:
        pt1, pt2 = pts
        cv2.line(image, pt1, pt2, (0, 0, 255), linewidth)


def visualize_pointcloud(
    npz_file: str, tensor: torch.Tensor = None, v: str = ""
) -> None:
    """Given an npz file extract and save the pointcloud as a ply file in temp/
    Alternatively: provide the list of 3d points as a torch tensor [m, 3] and add a version/uuid to the filename
    """
    if tensor is not None:
        keypoints3d = tensor
    else:
        assert Path(npz_file).exists(), f"{npz_file} does not exist"
        avg_data = np.load(npz_file)
        keypoints3d = torch.Tensor(avg_data["keypoints3d"])  # [m, 3]
    pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(keypoints3d.numpy())
    keypoints3d_cpu = keypoints3d.detach().clone().cpu().numpy().reshape(-1, 3)
    pcd.points = o3d.utility.Vector3dVector(keypoints3d_cpu)
    Path(f"temp/model").mkdir(parents=True, exist_ok=True)
    path: str = (
        f"temp/model/3d_keypoints_{v}.ply" if v else f"temp/model/3d_keypoints.ply"
    )
    logger.info(f"Saving pointcloud to {path}")
    o3d.io.write_point_cloud(path, pcd)
    return


def add_pointcloud_to_vis3d(pointcloud_pth, dump_dir, save_name):
    vis3d = Vis3D(dump_dir, save_name)
    vis3d.add_point_cloud(pointcloud_pth, name="filtered_pointcloud")


def save_demo_image(
    pose_pred,
    K,
    image_path,
    box3d,
    draw_box=True,
    save_path=None,
    color="b",
    comment: str = "",
):
    """
    Project 3D bbox by predicted pose and visualize
    """
    if isinstance(box3d, str):
        box3d = np.loadtxt(box3d)

    image_full = cv2.imread(image_path)

    if draw_box:
        reproj_box_2d = reproj(K, pose_pred, box3d)
        draw_3d_box(image_full, reproj_box_2d, color=color, linewidth=5)

    if comment:
        image_full = cv2.putText(
            image_full,
            comment,
            (10, image_full.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    if save_path is not None:
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)

        cv2.imwrite(save_path, image_full)
    return image_full


def save_comparison_image(
    pose_pred, pose_pred_optimized, K, image_path, box3d, draw_box=True, save_path=None
):
    """
    Save image w/ the inital & optimized pose predictions.
    """
    if isinstance(box3d, str):
        box3d = np.loadtxt(box3d)

    image_full = cv2.imread(image_path)

    if draw_box:
        # original pose prediction
        reproj_box_2d = reproj(K, pose_pred, box3d)
        draw_3d_box(image_full, reproj_box_2d, color="b", linewidth=2)
        # optimised
        reproj_box_2d = reproj(K, pose_pred_optimized, box3d)
        draw_3d_box(image_full, reproj_box_2d, color="r", linewidth=2)

    if save_path is not None:
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)

        cv2.imwrite(save_path, image_full)
    return image_full


def make_video(image_path, output_video_path):
    # Generate video:
    images = natsort.natsorted(os.listdir(image_path))
    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
    H, W, C = cv2.imread(str(Path(image_path) / images[0])).shape
    if Path(output_video_path).exists():
        Path(output_video_path).unlink()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_video_path, fourcc, 10, (W, H))
    for id, image_name in enumerate(images):
        image = cv2.imread(str(Path(image_path) / image_name))
        video.write(image)
    video.release()


def visualize_2D_3D_keypoints(
    path: Path, data, inp_crop, inliers, mkpts_3d, mkpts_query
):
    ## Visualize 3D keypoints ##
    pcd = o3d.geometry.PointCloud()
    keypoints3d_cpu = data["mkpts_3d_db"].detach().clone().cpu().numpy()
    pcd.points = o3d.utility.Vector3dVector(keypoints3d_cpu)
    o3d.io.write_point_cloud(str(path / "mkpts_3d_db.ply"), pcd)

    ## Visualize 2D keypoints ##
    query_image = inp_crop.squeeze().cpu().detach().numpy()
    query_image = query_image * 255
    query_image = query_image.astype(np.uint8)
    query_image = cv2.cvtColor(query_image, cv2.COLOR_GRAY2RGB)
    for point in mkpts_query:
        x, y = point.astype(int)
        cv2.circle(query_image, (x, y), 3, (0, 255, 0), -1)
    cv2.imwrite(str(path / "mkpts_query.png"), query_image)

    ## Visualize 2D inliers ##
    inliers_3d = np.zeros((len(inliers), 3))
    for idx, inlier in enumerate(inliers):
        inliers_3d[idx] = mkpts_3d[inlier]
        x, y = mkpts_query[inlier].astype(int)
        cv2.circle(query_image, (x, y), 3, (0, 0, 255), -1)
    cv2.imwrite(str(path / "mkpts_inliers.png"), query_image)

    ## Visualize 3D inliers ##
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(inliers_3d)
    o3d.io.write_point_cloud(str(path / "mkpts_3d_inliers.ply"), pcd)
