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