docker build -t="mizeller:00" .;
docker run --gpus all -w /workspaces/OnePose_ST -v ~/projects/OnePose_ST:/workspaces/OnePose_ST -it mizeller:00;