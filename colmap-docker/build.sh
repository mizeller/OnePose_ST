docker build -t="colmap:cuda113" .;
docker run --gpus all -w /workspaces/OnePose_ST -v ~/projects/OnePose_ST:/workspaces/OnePose_ST -it colmap:cuda113;
