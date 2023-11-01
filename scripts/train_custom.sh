#!/bin/bash
PROJECT_DIR="$(pwd)"

DATA_PATH="data/synthetic_data_spot/synthetic_data_nerf"

OBJ_NAME=$1

DATA_DIR="$PROJECT_DIR/$DATA_PATH"


echo -e "Current work dir:\t $PROJECT_DIR" > output.txt 
echo -e "Custom data dir:\t $DATA_DIR" > output.txt 

: '
test
The DATA_PATH points to the file structure containing required files in the naming convention
defined by OnePose++. This data was generated during the synthetic data generation pipeline 
and manually re-structured to conform with the required conventions (*).

(*) This will be resolved in the future by adding a script to the synthetic data 
generation pipeline to
- automatically convert the rgb frames to a (black-and-white) Frames.m4v video in the onepose_data dir
- automatically copy the onepose_data dir content to a scene_<idx>-annotate directory
- automatically remove the old scene_<idx> directory

In the end, DATA_DIR should contain the following structure:
.
├── scene_00-annotate
│   ├── ARposes.txt
│   ├── Box.txt
│   ├── Frames.m4v
│   ├── Frames.txt
│   ├── intrinsics.txt
│   └── synthetic_data_annotated_00.mp4
...
└── scene_29-annotate
    ├── ARposes.txt
    ├── Box.txt
    ├── Frames.m4v
    ├── Frames.txt
    └── intrinsics.txt
'


: '
# Parsing worked nicely :-)


echo '-------------------'
echo 'Parse scanned data:'
echo '-------------------'
# Parse scanned annotated & test sequence:

python $PROJECT_DIR/parse_scanned_data.py --scanned_object_path "$DATA_DIR" >> output.txt 
'

# annotate_directories=()
# for dir in "$DATA_DIR"/*-annotate; do
#     if [ -d "$dir" ]; then
#         annotate_directories+=("$(basename $dir)")
#     fi
# done
# echo ${annotate_directories[@]}

# annotated_directories_string=""
# for dir in "${annotate_directories[@]}"
# do
#   annotated_directories_string+="$dir "
# done

# echo $annotated_directories_string


echo '--------------------------------------------------------------'
echo 'Run Keypoint-Free SfM to reconstruct object point cloud for pose estimation:'
echo '--------------------------------------------------------------'
# Run SfM to reconstruct object sparse point cloud from $OBJ_NAME-annotate sequence:
python $PROJECT_DIR/run.py \
    +preprocess="sfm_demo" \
    dataset.data_dir="[$DATA_DIR scene_00-annotate]" \
    dataset.outputs_dir="$PROJECT_DIR/data/sfm_model"
    
#   dataset.data_dir="[$DATA_DIR scene_00-annotate scene_01-annotate scene_02-annotate scene_03-annotate scene_04-annotate scene_05-annotate scene_06-annotate scene_07-annotate scene_08-annotate scene_09-annotate scene_10-annotate scene_11-annotate scene_12-annotate scene_13-annotate scene_14-annotate scene_15-annotate scene_16-annotate scene_17-annotate scene_18-annotate scene_19-annotate scene_20-annotate scene_21-annotate scene_22-annotate scene_23-annotate scene_24-annotate scene_25-annotate scene_26-annotate scene_27-annotate scene_28-annotate scene_29-annotate]" \
# echo "-----------------------------------"
# echo "Run inference and output demo video:"
# echo "-----------------------------------"

# # Run inference on $OBJ_NAME-test and output demo video:
# python $PROJECT_DIR/demo.py +experiment="inference_demo" data_base_dir="$PROJECT_DIR/data/demo/$OBJ_NAME $OBJ_NAME-test" sfm_base_dir="$PROJECT_DIR/data/demo/sfm_model/outputs_softmax_loftr_loftr/$OBJ_NAME"