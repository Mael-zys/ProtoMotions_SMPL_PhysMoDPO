export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/
export CUDA_VISIBLE_DEVICES=0,1,2,3


python protomotions/train_agent.py \
+exp=full_body_tracker/transformer_flat_terrain +robot=smpl \
+simulator=isaacgym motion_file=data/Dataset/amass_data/amass_train_smpl.pt \
+experiment_name=full_body_tracker +terrain=flat \
ngpu=4 eval_overrides.ngpu=4 num_envs=2048 agent.config.batch_size=8192 +opt=wandb \
agent.config.eval_metrics_every=1000 training_max_steps=100000000000