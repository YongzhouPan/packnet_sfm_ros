# /usr/bin/python -m pip install --upgrade pip
# sudo pip install --upgrade wandb
wandb login c7a16e7705748cd60106adbb57f03a601ef3f828

# export WANDB_NAME="Let's graduate"
export WANDB_ENTITY="surfii3z"
export WANDB_PROJECT="thesis"
export WANDB_API_KEY="c7a16e7705748cd60106adbb57f03a601ef3f828"
# export WANDB_RESUME="allow"
# export WANDB_RUN_ID="c038edb911820e40b927968ca8e89ae2ad6cb6da"
CUDA_VISIBLE_DEVICES=0 python scripts/train.py configs/tello.yaml
# python scripts/train.py /workspace/packnet-sfm/results/chept/eager-frost-3/epoch=19_optitrack_1-optitrack_1_part1-loss=0.000.ckpt