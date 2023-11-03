This folder contains all the neccessary files to test the colmap installation w/o having to run
the whole OnePose++ pipeline OR having these dependencies installed in the current environment. 

It should make the whole testing/debugging process easier and more straight-forward (hopefully).

/workspaces/OnePose_ST/colmap_test/database.db points to the database
/workspaces/OnePose_ST/colmap_test/pairs-covis10.txt points to the match_list

Of course, the prepared data needs to be at the path specified in the paris-covis10.txt file. 
Therefore, the /data dir in this repo mustn't be changed!

For the geometric verification call use this one:

```bash
colmap matches_importer --database_path /workspaces/OnePose_ST/colmap_test/database.db --match_list_path /workspaces/OnePose_ST/colmap_test/pairs-covis10.txt --match_type pairs
```

Currently this error/issue arises:

```bash 
# Shader not supported by your hardware!
# ERROR: SiftGPU not fully supported
```




