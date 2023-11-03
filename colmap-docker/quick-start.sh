docker pull colmap/colmap:latest
docker run --gpus all -w /workspaces/OnePose_ST -v $1:/workspaces/OnePose_ST -it colmap/colmap:latest;
