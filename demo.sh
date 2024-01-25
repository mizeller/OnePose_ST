#!/bin/bash
PROJECT_DIR="$(pwd)"
echo "Current work dir: $PROJECT_DIR"

echo '-------------------'
echo 'Parse scanned data:'
echo '-------------------'
# Parse scanned annotated sequences:
python3 $PROJECT_DIR/parse_scanned_data.py --scanned_object_path "$PROJECT_DIR/data/spot_demo"

echo '--------------------------------------------------------------'
echo 'Run Keypoint-Free SfM to reconstruct object point cloud for pose estimation:'
echo '--------------------------------------------------------------'
# Run SfM to reconstruct object sparse point cloud from demo_cam-annotate sequence:
python3 run.py \
    +preprocess="sfm_demo" \
    dataset.data_dir="[$PROJECT_DIR/data/spot_demo spot_00-annotate spot_01-annotate spot_02-annotate spot_03-annotate spot_04-annotate]" \
    dataset.outputs_dir="$PROJECT_DIR/data/spot_demo/sfm_model" \

echo "-----------------------------------"
echo "Run inference and output demo video:"
echo "-----------------------------------"
# Run inference on demo_cam-test and output demo video:
python3 $PROJECT_DIR/inference.py --obj_name spot_demo --test_dirs asus_00-test