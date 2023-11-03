Taken from https://github.com/JanaldoChen/colmap-docker/tree/main#f3

Build via: 

 
```bash
bash build.sh ~/projects/OnePose_ST/
```

This build and ran a working container, where COLMAP 3.8 -- (Commit ea40ef9a on 2022-01-26 with CUDA) is installed. 

both 

```bash
colmap matches_importer --database_path /workspaces/OnePose_ST/data/demo/sfm_model/outputs_softmax_loftr_loftr/demo_cam/sfm_ws/database.db --match_list_path /workspaces/OnePose_ST/data/demo/sfm_model/outputs_softmax_loftr_loftr/demo_cam/pairs-covis10.txt --match_type pairs
```

and 

```bash
colmap point_triangulator --database_path /workspaces/OnePose_ST/data/demo/sfm_model/outputs_softmax_loftr_loftr/demo_cam/sfm_ws/database.db --image_path / --input_path /workspaces/OnePose_ST/data/demo/sfm_model/outputs_softmax_loftr_loftr/demo_cam/sfm_empty --output_path /workspaces/OnePose_ST/data/demo/sfm_model/outputs_softmax_loftr_loftr/demo_cam/sfm_ws/model --Mapper.ba_refine_focal_length 0 --Mapper.ba_refine_principal_point 0 --Mapper.ba_refine_extra_params 0
```

ran through succesfully. (See stacktrace.log)
