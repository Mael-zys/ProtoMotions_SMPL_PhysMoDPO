#!/bin/sh

export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/
export CUDA_VISIBLE_DEVICES=0

model_path=$1
success_threshold=0.5

## compute metrics


# evaluate the first stage model on testing data
python protomotions/eval_agent.py +robot=smpl +simulator=isaacgym \
+motion_file=data/Dataset/amass_data/amass_test_smpl.pt \
+checkpoint=${model_path} \
+force_flat_terrain=True +ngpu=1 +eval_overrides.ngpu=1 +eval_compute_metrics=True +headless=True +terrain=flat +save_motion=True \
hydra.run.dir=outputs/full_body_${success_threshold}_test \
+agent.config.eval_metric_keys=["cartesian_err","gt_err","dv_rew","kb_rew","lr_rew","rv_rew","rav_rew","gr_err","gr_err_degrees"] \
+eval_overrides.num_envs=256 +env.config.num_envs=256 +num_envs=256 +agent.config.success_threshold=${success_threshold}
