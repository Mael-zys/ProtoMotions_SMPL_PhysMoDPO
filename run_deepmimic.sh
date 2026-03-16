#!/bin/bash  -i

source ~/miniconda3/etc/profile.d/conda.sh

conda activate protomotions


input_folder=$1
cuda_num=$2

export CUDA_VISIBLE_DEVICES=$cuda_num
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/

# convert amass to isaac format
python data/scripts/convert_amass_to_isaac.py ${input_folder}/amass_format --humanoid-type=smpl --force-remake 

python data/scripts/process_my_data.py --body_format smpl \
    --root_path ${input_folder}/amass_format \
    --output_path ${input_folder}/amass_format/data_list.yaml 

python data/scripts/package_motion_lib.py ${input_folder}/amass_format/data_list.yaml \
${input_folder}/amass_format ${input_folder}/amass_format/data_list.pt --humanoid-type=smpl

# run evaluation
python protomotions/eval_agent.py +robot=smpl +simulator=isaacgym \
+motion_file=${input_folder}/amass_format/data_list.pt \
+checkpoint=data/pretrained_models/motion_tracker/smpl/last.ckpt \
+force_flat_terrain=True +ngpu=1 +eval_overrides.ngpu=1 +eval_compute_metrics=True +headless=True +terrain=flat +save_motion=True \
hydra.run.dir=${input_folder}/deepmimic_output \
+agent.config.eval_metric_keys=["cartesian_err","gt_err","dv_rew","kb_rew","lr_rew","rv_rew","rav_rew","gr_err","gr_err_degrees","power"] \
+eval_overrides.num_envs=256 +env.config.num_envs=256 +num_envs=256 \
+agent.config.motion_yaml=${input_folder}/amass_format/data_list.yaml

rm ${input_folder}/amass_format/data_list.pt