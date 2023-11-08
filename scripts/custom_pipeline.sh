#!/bin/bash
PROJECT_DIR="$(pwd)"
echo "Current work dir: $PROJECT_DIR"

echo '-------------------'
echo 'Parse scanned data:'
echo '-------------------'
# Parse scanned annotated & test sequence:
python3 $PROJECT_DIR/parse_scanned_data.py \
    --scanned_object_path \
    "$PROJECT_DIR/data/demo/demo_cam"

echo '--------------------------------------------------------------'
echo 'Run Keypoint-Free SfM to reconstruct object point cloud for pose estimation:'
echo '--------------------------------------------------------------'
# Run SfM to reconstruct object sparse point cloud from demo_cam-annotate sequence:
python3 $PROJECT_DIR/run.py \
    +preprocess="sfm_demo" \
    dataset.data_dir="[$PROJECT_DIR/data/demo/demo_cam demo_cam-annotate]" \
    dataset.outputs_dir="$PROJECT_DIR/data/demo/sfm_model" \

echo "-----------------------------------"
echo "Run inference and output demo video:"
echo "-----------------------------------"
# Run inference on demo_cam-test and output demo video:
python3 $PROJECT_DIR/demo.py +experiment="inference_demo" data_base_dir="$PROJECT_DIR/data/demo/demo_cam demo_cam-test" sfm_base_dir="$PROJECT_DIR/data/demo/sfm_model/outputs_softmax_loftr_loftr/demo_cam"







#!/bin/bash

echo '-------------------'
echo 'Parse scanned data:'
echo '-------------------'
python3 /workspaces/OnePose_ST/parse_scanned_data.py --scanned_object_path "/workspaces/OnePose_ST/data/spot"
echo '--------------------------------------------------------------'
echo 'Run Keypoint-Free SfM to reconstruct object point cloud for pose estimation:'
echo '--------------------------------------------------------------'
python3 /workspaces/OnePose_ST/run.py \
    +preprocess="sfm_demo" \
    dataset.data_dir="[/workspaces/OnePose_ST/data/spot spot_01-annotate spot_02-annotate spot_03-annotate spot_04-annotate spot_05-annotate spot_06-annotate spot_07-annotate spot_08-annotate spot_09-annotate spot_10-annotate spot_11-annotate spot_12-annotate spot_13-annotate spot_14-annotate spot_15-annotate spot_16-annotate spot_17-annotate spot_18-annotate spot_19-annotate spot_20-annotate spot_21-annotate spot_22-annotate spot_23-annotate spot_24-annotate spot_25-annotate spot_26-annotate spot_27-annotate spot_28-annotate spot_29-annotate]" \
    dataset.outputs_dir="/workspaces/OnePose_ST/data/spot/sfm_model"
# TODO: incl inference as a next step