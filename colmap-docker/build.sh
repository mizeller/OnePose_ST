docker build -t="colmap:latest" .;
docker run --gpus all -w /workspaces/OnePose_ST -v $1:/workspaces/OnePose_ST -it colmap:latest;
