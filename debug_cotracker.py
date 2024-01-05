# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# *slightly modified*

import torch
import imageio.v3 as iio
import numpy as np
from cotracker.cotracker.utils.visualizer import Visualizer
from cotracker.cotracker.predictor import CoTrackerOnlinePredictor


def _process_step(model, window_frames, is_first_step, queries):
    # video_chunck = last 8 frames
    video_chunk = (
        torch.tensor(np.stack(window_frames[-model.step * 2 :]), device="cuda")
        .float()
        .permute(0, 3, 1, 2)[None]
    )  # (1, T, 3, H, W)
    # print(video_chunk.shape)
    return model(video_chunk, is_first_step=is_first_step, queries=queries[None])


if __name__ == "__main__":
    model = CoTrackerOnlinePredictor()
    model = model.to("cuda")
    window_frames = []

    # these queries (58) were provided by OnePose++ for the
    # initial frame of the cotrack-test video
    queries = torch.tensor(
        [
            [0.0, 200.0, 257.0],
            [0.0, 219.0, 247.0],
            [0.0, 259.0, 267.0],
            [0.0, 227.0, 255.0],
            [0.0, 214.0, 242.0],
            [0.0, 233.0, 264.0],
            [0.0, 227.0, 268.0],
            [0.0, 227.0, 272.0],
            [0.0, 213.0, 245.0],
            [0.0, 171.0, 252.0],
            [0.0, 260.0, 280.0],
            [0.0, 259.0, 276.0],
            [0.0, 268.0, 280.0],
            [0.0, 256.0, 279.0],
            [0.0, 263.0, 279.0],
            [0.0, 262.0, 271.0],
            [0.0, 211.0, 258.0],
            [0.0, 270.0, 278.0],
            [0.0, 219.0, 254.0],
            [0.0, 216.0, 246.0],
            [0.0, 276.0, 281.0],
            [0.0, 231.0, 235.0],
            [0.0, 235.0, 296.0],
            [0.0, 227.0, 251.0],
            [0.0, 279.0, 281.0],
            [0.0, 270.0, 275.0],
            [0.0, 208.0, 243.0],
            [0.0, 215.0, 255.0],
            [0.0, 235.0, 267.0],
            [0.0, 227.0, 243.0],
            [0.0, 219.0, 238.0],
            [0.0, 230.0, 251.0],
            [0.0, 171.0, 259.0],
            [0.0, 219.0, 251.0],
            [0.0, 219.0, 235.0],
            [0.0, 239.0, 267.0],
            [0.0, 220.0, 316.0],
            [0.0, 232.0, 266.0],
            [0.0, 230.0, 246.0],
            [0.0, 263.0, 283.0],
            [0.0, 235.0, 243.0],
            [0.0, 216.0, 250.0],
            [0.0, 203.0, 263.0],
            [0.0, 228.0, 258.0],
            [0.0, 276.0, 275.0],
            [0.0, 279.0, 270.0],
            [0.0, 231.0, 238.0],
            [0.0, 213.0, 251.0],
            [0.0, 211.0, 254.0],
            [0.0, 235.0, 235.0],
            [0.0, 223.0, 238.0],
            [0.0, 264.0, 275.0],
            [0.0, 235.0, 238.0],
            [0.0, 225.0, 259.0],
            [0.0, 223.0, 316.0],
            [0.0, 243.0, 275.0],
            [0.0, 169.0, 259.0],
            [0.0, 243.0, 259.0],
        ],
        device="cuda:0",
    )

    # Iterating over video frames, processing one window at a time:
    is_first_step = True
    for i, frame in enumerate(
        iio.imiter(
            "data/spot_rgb/cotrack-test/clip.mp4",  # TODO: fix path here if needed...
            plugin="FFMPEG",
        )
    ):
        run_tracker: bool = i % model.step == 0 and i != 0
        if run_tracker:
            print(f"tracking frame {i}...")
            pred_tracks, pred_visibility = _process_step(
                model, window_frames, is_first_step, queries
            )
            is_first_step = False
        window_frames.append(frame)

    # Processing the final video frames in case video length is not a multiple of model.step
    pred_tracks, pred_visibility = _process_step(
        model,
        window_frames[-(i % model.step) - model.step - 1 :],
        is_first_step,
        queries,
    )

    print("Tracks are computed")

    # save a video with predicted tracks
    seq_name = "test"
    video = torch.tensor(np.stack(window_frames), device="cuda").permute(0, 3, 1, 2)[
        None
    ]

    vis = Visualizer(save_dir="temp/", mode="cool", linewidth=2, tracks_leave_trace=-1)
    vis.visualize(
        video=video,
        tracks=pred_tracks,
        visibility=pred_visibility,
        filename="asdf",
    )
