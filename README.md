# ProtoMotions: Physics-based Character Animation

# Description
This codebase is a modified version from [ProtoMotions v2](https://github.com/NVlabs/ProtoMotions/tree/9062f0ae139767d9ac05cebf2253583e2652a56a) to perform motion tracking with SMPL robot simulation for [PhysMoDPO](https://github.com/Mael-zys/PhysMoDPO) project.

1. For running PhysMoDPO project, only need to install environment and setup the SMPL models according to [Installation](#installation).
2. In this project, we use the pretrained model from [ProtoMotions v2](https://github.com/NVlabs/ProtoMotions/tree/9062f0ae139767d9ac05cebf2253583e2652a56a). To retrain the corresponding DeepMimic model, please refer to [Training DeepMimic](#train-deepmimic) section to preprocess training data and run the training script. 


# Installation

## Environment

This codebase supports IsaacGym.

First run `git lfs fetch --all` to fetch all files stored in git-lfs.

Run the installation in one script:
```bash
bash scripts/install_protomotion.sh
```

or follow this step-by-step guide below:
<details>
<summary>IsaacGym</summary>

1. Install [IsaacGym](https://developer.nvidia.com/isaac-gym)
```bash
wget https://developer.nvidia.com/isaac-gym-preview-4
tar -xvzf isaac-gym-preview-4
```

Install IsaacGym Python API:

```bash
pip install -e isaacgym/python
```
2. Once IG and PyTorch are installed, from the repository root install the ProtoMotions package and its dependencies with:
```bash
pip install -e .
pip install -r requirements_isaacgym.txt
pip install -e isaac_utils
pip install -e poselib
```
Set the `PYTHON_PATH` env variable (not really needed, but helps the instructions stay consistent between sim and gym).
```bash
alias PYTHON_PATH=python
```

### Potential Issues

If you have python errors:

```bash
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/
```

If you run into memory issues -- try reducing the number of environments by adding to the command line `num_envs=1024`

</details>


## Download SMPL parameters
Download the [SMPL](https://smpl.is.tue.mpg.de/) v1.1.0 parameters and place them in the `data/smpl/` folder. Rename the files:
- basicmodel_neutral_lbs_10_207_0_v1.1.0 -> SMPL_NEUTRAL.pkl
- basicmodel_m_lbs_10_207_0_v1.1.0.pkl -> SMPL_MALE.pkl
- basicmodel_f_lbs_10_207_0_v1.1.0.pkl -> SMPL_FEMALE.pkl


# Train DeepMimic

## Data Preprocessing

Download the [processed data](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/yangsong_zhang_mbzuai_ac_ae/IQB7qLs_a1uRQogJTeA-vTKJASGsI43giNUYGLLE_igwf0g?e=zuUQt4)

Or follow the detail instructions below:

<details>
<summary>Data Preprocessing</summary>

### Data Download

Download the [AMASS](https://amass.is.tue.mpg.de/) dataset and put it in the `data/Dataset/amass_data/` folder.


### Data Conversion

Run the following scripts to convert the AMASS data to the MotionLib format and package them for faster loading.
```bash
bash scripts/convert_amass.sh
```

More details can be found in [ProtoMotions](https://github.com/NVlabs/ProtoMotions/tree/main?tab=readme-ov-file#data).
</details>


Motions can be visualized via kinematic replay by running `PYTHON_PATH protomotions/scripts/play_motion.py <motion file> <simulator isaacgym/isaaclab/genesis> <robot type>`.

## Training


We provide an example training script to train MaskedMimic with our dataset:
```bash
bash scripts/train_deepmimic.sh
```


## Evaluation

We provide an evaluation script to evaluate the performance on AMASS:

```bash
bash scripts/eval_deepmimic.sh data/pretrained_models/motion_tracker/smpl/last.ckpt
```


Here is an example command to visualize the agent's performance for hands control:
```bash
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib/
export CUDA_VISIBLE_DEVICES=0

python protomotions/eval_agent.py +robot=smpl +simulator=isaacgym \
+motion_file=data/Dataset/amass_data/amass_test_smpl.pt \
+checkpoint=./data/pretrained_models/motion_tracker/smpl/last.ckpt \
+force_flat_terrain=True +ngpu=1 +eval_overrides.ngpu=1 
```


We provide a set of pre-defined keyboard controls.

| Key | Description                                                                |
|-----|----------------------------------------------------------------------------|
| `J` | Apply physical force to all robots (tests robustness)                      |
| `R` | Reset the task                                                             |
| `O` | Toggle camera. Will cycle through all entities in the scene.               |
| `L` | Toggle recording video from viewer. Second click will save frames to video |
| `;` | Cancel recording                                                           |
| `U` | Update inference parameters (e.g., in MaskedMimic user control task)       |
| `Q` | Quit       |


# References
This project repository builds upon [ProtoMotions v2](https://github.com/NVlabs/ProtoMotions/tree/9062f0ae139767d9ac05cebf2253583e2652a56a).
