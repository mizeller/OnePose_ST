import torch


def normalize_2d_keypoints(kpts, image_shape):
    """Normalize 2d keypoints locations based on image shape
    kpts: [b, n, 2]
    image_shape: [b, 2]
    """
    height, width = image_shape[0]
    one = kpts.new_tensor(1)
    size = torch.stack([one * width, one * height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


def normalize_3d_keypoints(kpts):
    """Normalize 3d keypoints locations based on the tight box
    kpts: [b, n, 3]
    """
    width, height, length = kpts[0].max(dim=0).values - kpts[0].min(dim=0).values
    center = torch.mean(kpts, dim=-2)  # B*3
    one = kpts.new_tensor(1)
    size = torch.stack([one * width, one * height, one * length])[None]
    scaling = size.max(1, keepdim=True).values * 0.6
    kpts_rescaled = (kpts - center[:, None, :]) / scaling[:, None, :]

    # DBG: visualize normalized keypoints
    # import open3d as o3d
    # pcd = o3d.geometry.PointCloud()
    # keypoints3d_cpu = kpts_rescaled.detach().clone().cpu().numpy().reshape(-1,3)
    # pcd.points = o3d.utility.Vector3dVector(keypoints3d_cpu)
    # path: str = f"temp/3d_keypoints_normalized.ply"
    # o3d.io.write_point_cloud(path, pcd)

    return kpts_rescaled
