import os
import os.path as osp
import glob
import natsort


"""
For each object, we store in the following directory format:

data_root:
    - box3d_corners.txt
    - seq1_root
        - intrinsics.txt
        - color/
        - poses_ba/
        - intrin_ba/
        - ......
    - seq2_root
    - ......
"""


def get_gt_pose_path_by_color(color_path, det_type="GT_box"):
    ext = osp.splitext(color_path)[1]
    if det_type == "GT_box":
        return color_path.replace("/color/", "/poses_ba/").replace(ext, ".txt")
    elif det_type == "feature_matching":
        return color_path.replace("/color_det/", "/poses_ba/").replace(ext, ".txt")
    else:
        raise NotImplementedError


def get_img_full_path_by_color(color_path, det_type="GT_box"):
    if det_type == "GT_box":
        return color_path.replace("/color/", "/color_full/")
    elif det_type == "feature_matching":
        return color_path.replace("/color_det/", "/color_full/")
    else:
        raise NotImplementedError


def get_intrin_path_by_color(color_path, det_type="GT_box"):
    if det_type == "GT_box":
        return color_path.replace("/color/", "/intrin_ba/").replace(".png", ".txt")
    elif det_type == "feature_matching":
        return color_path.replace("/color_det/", "/intrin_det/").replace(".png", ".txt")
    else:
        raise NotImplementedError


def get_intrin_dir(seq_root):
    return osp.join(seq_root, "intrin_ba")


def get_gt_pose_dir(seq_root):
    return osp.join(seq_root, "poses_ba")


def get_intrin_full_path(seq_root):
    return osp.join(seq_root, "intrinsics.txt")


def get_3d_box_path(data_root):
    return osp.join(data_root, "box3d_corners.txt")


def get_test_seq_path(obj_root, last_n_seq_as_test=1):
    seq_names = os.listdir(obj_root)
    seq_names = [seq_name for seq_name in seq_names if "-" in seq_name]
    seq_ids = [
        int(seq_name.split("-")[-1]) for seq_name in seq_names if "-" in seq_name
    ]

    test_obj_name = seq_names[0].split("-")[0]
    test_seq_ids = sorted(seq_ids)[(-1 * last_n_seq_as_test) :]
    test_seq_paths = [
        osp.join(obj_root, test_obj_name + "-" + str(test_seq_id))
        for test_seq_id in test_seq_ids
    ]
    return test_seq_paths


def get_default_paths(data_root, data_dir, sfm_model_dir):
    sfm_ws_dir = osp.join(
        sfm_model_dir,
        "sfm_ws",
        "model",
    )

    img_lists = []
    color_dir = osp.join(data_dir, "color_full")
    img_lists += glob.glob(color_dir + "/*.png", recursive=True)
    img_lists = natsort.natsorted(img_lists)
    det_box_vis_video_path = osp.join(data_dir, "det_box.mp4")
    demo_video_path = osp.join(data_dir, "demo_video.mp4")
    intrin_full_path = osp.join(data_dir, "intrinsics.txt")
    intrin_full_dir = osp.join(data_dir, "intrin_full")
    bbox3d_path = osp.join(data_root, "box3d_corners.txt")
    paths = {
        "data_root": data_root,
        "data_dir": data_dir,
        "sfm_dir": sfm_model_dir,
        "sfm_ws_dir": sfm_ws_dir,
        "bbox3d_path": bbox3d_path,
        "intrin_full_path": intrin_full_path,
        "intrin_full_dir": intrin_full_dir,
        "det_box_vis_video_path": det_box_vis_video_path,
        "demo_video_path": demo_video_path,
    }
    return img_lists, paths
