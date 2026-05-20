<h1 align="center">RoboFlow4D</h1>

<p align="center">
  Training, preprocessing, visualization, and policy code for
  <strong>RoboFlow4D: A Lightweight Flow World Model Toward Real-Time Flow-Guided Robotic Manipulation</strong>
</p>

<p align="center">
  <strong>Proceedings of the International Conference on Machine Learning 2026</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Paper-Coming_Soon-1f6feb.svg" alt="Paper coming soon">
  <a href="./pipeline.pdf"><img src="https://img.shields.io/badge/Pipeline-PDF-0ea5e9.svg" alt="Pipeline PDF"></a>
  <a href="#quick-start"><img src="https://img.shields.io/badge/Quick_Start-Pipeline-2ea44f.svg" alt="Quick Start"></a>
  <a href="#visualization"><img src="https://img.shields.io/badge/3D-Visualization-8b5cf6.svg" alt="3D Visualization"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/Citation-BibTeX-b31b1b.svg" alt="Citation"></a>
</p>

<p align="center">
  <a href="./pipeline.pdf">
    <img src="./assets/pipeline.png" alt="RoboFlow4D pipeline" width="100%">
  </a>
</p>

RoboFlow4D is a lightweight flow world model for real-time flow-guided robotic manipulation. This repository contains the public-facing code path for dataset preprocessing, 3D point-flow modeling, visualization, and flow-conditioned action policy training.

<a id="highlights"></a>

## ✨ Highlights

- Metric and SpaTracker/VGGT-style 3D point-flow preprocessing.
- Flow model training and HDF5 prediction export through stable entrypoints.
- Interactive 3D HTML visualization and fixed-view MP4 rendering.
- Flow-conditioned action policy training and evaluation for LIBERO and ManiSkill.
- Clean public layout with local-only backups, checkpoints, caches, and prompt tables ignored by git.

## 🗂️ Table of Contents

- [Highlights](#highlights)
- [Quick Start](#quick-start)
- [Pipeline Overview](#pipeline-overview)
- [Repository Layout](#repository-layout)
- [Environment](#environment)
- [Data Preprocessing](#data-preprocessing)
- [Calibration](#calibration)
- [Train 3D Flow Model](#train-3d-flow-model)
- [Save Predicted Flow To HDF5](#save-predicted-flow-to-hdf5)
- [Visualization](#visualization)
- [Train Action Policy](#train-action-policy)
- [Evaluation](#evaluation)
- [Notes](#notes)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

<a id="quick-start"></a>

## 🚀 Quick Start

```bash
conda env create -f 3DFlowModel/environment.yml
conda activate roboflow

# 1. Preprocess demonstrations.
python process_data/process_libero_hdf5.py --input_dirs /path/to/libero --out_root /path/to/tracks

# 2. Optional: convert track points to metric robot-base coordinates.
python process_data/patch_libero_sim_depth_to_tracks.py --tracks /path/to/tracks --overwrite
python utils/set_point_traj_mode.py --tracks /path/to/tracks --mode metric

# 3. Train the flow model and write predictions back to HDF5.
python 3DFlowModel/train_flow_model.py --flow_root /path/to/tracks --save_dir ckpts/flow/example
python 3DFlowModel/predict_flow_hdf5.py --hdf5_root /path/to/tracks --ckpt /path/to/flow.pt --model_py 3DFlowModel/flow_model.py --overwrite
```

<a id="pipeline-overview"></a>

## 🔁 Pipeline Overview

The main code path is:

1. preprocess demonstrations into track HDF5 files
2. optionally convert 2D tracks into metric 3D robot-base trajectories
3. train the 3D flow model
4. write predicted flow back into HDF5 as `pre_point_traj`
5. visualize predicted or ground-truth 3D flow
6. train and evaluate the flow-conditioned action policy

<a id="repository-layout"></a>

## 🧭 Repository Layout

```text
RoboFlow4D/
  3DFlowModel/          Core flow-model training, inference, and model files.
  Action_policy/        Flow-conditioned action policy training and evaluation.
  process_data/         Dataset preprocessing and metric 3D conversion scripts.
  visualization/        2D/3D flow visualization scripts and HTML/MP4 exporters.
  utils/                Small HDF5, trajectory, export, and mode-switch utilities.
  assets/               README figures and paper assets.

  Grounded-SAM-2/       External dependency checkout or submodule.
  SpaTrackerV2/         External dependency checkout or submodule.
```

<a id="environment"></a>

## 🛠️ Environment

The most complete environment file is:

```bash
conda env create -f 3DFlowModel/environment.yml
conda activate roboflow
```

You also need the external checkpoints for Grounded-SAM2, SpaTrackerV2 / VGGT, SigLIP, and any trained flow or policy checkpoints.

Optional external paths can be provided with environment variables instead of editing source files:

```bash
export LIBERO_ROOT=/path/to/LIBERO
export DOMINO_ROOT=/path/to/DOMINO
export ROBOFLOW4D_LIBERO_FLOW_MODEL_PY=/path/to/custom_flow_model.py  # optional
```

<a id="data-preprocessing"></a>

## 📦 Data Preprocessing

Run commands from the repository root unless noted otherwise.

### LIBERO

```bash
CUDA_VISIBLE_DEVICES=0,1 python process_data/process_libero_hdf5.py \
  --input_dirs /path/to/libero/datasets_no_noops/libero_object_no_noops \
  --out_root /path/to/output/Flow_training_all/libero_object \
  --prompt_csv /path/to/optional_prompts.csv \
  --prompt_csv_key_type auto \
  --add_prefix "white robotic gripper." \
  --save_visuals \
  --vggt_device cuda:1 \
  --vggt_amp fp16 \
  --vggt_chunk 100 \
  --free_sam_after_mask
```

For stride-based processing, use:

```bash
python process_data/process_libero_10_hdf5.py ...
```

### ManiSkill

```bash
CUDA_VISIBLE_DEVICES=0,1 python process_data/process_maniskill_hdf5.py \
  --input_dirs /path/to/maniskill/replay \
  --out_root /path/to/output/ManiSkill/PushCube \
  --prompt_csv /path/to/optional_prompts.csv \
  --prompt_csv_key_type auto \
  --add_prefix "white robotic gripper." \
  --save_visuals \
  --vggt_device cuda:1 \
  --vggt_amp fp16 \
  --vggt_chunk 100 \
  --free_sam_after_mask
```

### Real World (e.g., DROID)

Use the DROID processor when your real-world data is stored in the DROID dataset format:

```bash
CUDA_VISIBLE_DEVICES=0 python process_data/process_droid.py \
  --droid_dir /path/to/droid \
  --out_root /path/to/output/droid_processed \
  --target_mode gripper \
  --mask_prompt_mode auto \
  --save_visuals
```

For raw real-world MP4 recordings, first convert videos into the minimal HDF5 layout, then run the standard preprocessing / flow pipeline on the generated HDF5:

```bash
python process_data/mp4_to_minimal_real_hdf5.py \
  --input_dir /path/to/mp4s \
  --out_hdf5 /path/to/real_input.hdf5
```

<a id="calibration"></a>

## 📐 Calibration

Calibration controls how tracked pixels are lifted into 3D and which coordinate frame is used for flow training. Track HDF5 files can keep two 3D point representations side by side:

- `point_traj_spatracker`: SpaTrackerV2 / VGGT 3D points in the predicted tracker frame.
- `point_traj_base_metric`: metric-scale 3D points in the robot base frame, in meters.
- `point_traj`: the active alias used by older training and visualization code.

To add LIBERO simulator depth, intrinsics, camera poses, and metric 3D trajectories:

```bash
MUJOCO_GL=egl MPLCONFIGDIR=/tmp/matplotlib \
python process_data/patch_libero_sim_depth_to_tracks.py \
  --tracks /path/to/task_tracks.hdf5 \
  --overwrite
```

If a track file already contains metric `depths`, `intrinsics`, and `T_base_cam`, convert directly:

```bash
python process_data/convert_track2d_to_point_traj_base_metric.py \
  --tracks /path/to/task_tracks.hdf5 \
  --overwrite
```

Switch which representation old code reads through `point_traj`:

```bash
# Keep both representations, leave point_traj unchanged.
python utils/set_point_traj_mode.py --tracks /path/to/task_tracks.hdf5 --mode both

# Use SpaTracker / VGGT 3D points.
python utils/set_point_traj_mode.py --tracks /path/to/task_tracks.hdf5 --mode spatracker

# Use metric robot-base 3D points.
python utils/set_point_traj_mode.py --tracks /path/to/task_tracks.hdf5 --mode metric
```

Training compatibility notes:

- The current flow training script reads `point_traj`; it does not automatically read `point_traj_base_metric`.
- To train on metric-scale robot-base trajectories, first run `set_point_traj_mode.py --mode metric`.
- Keep one training run in one coordinate system. Do not mix SpaTracker/VGGT `point_traj` demos with metric robot-base `point_traj` demos.
- Checkpoints trained on SpaTracker/VGGT coordinates should not be treated as metric-scale predictors. Train from scratch or fine-tune on metric targets.
- Downstream `pre_point_traj` and policy training should use the same trajectory coordinate convention as the flow model that produced it.

<a id="train-3d-flow-model"></a>

## 🌊 Train 3D Flow Model

Current default training entry:

```bash
CUDA_VISIBLE_DEVICES=0 python 3DFlowModel/train_flow_model.py \
  --flow_root /path/to/Flow_training_all/libero_object \
  --k_steps 20 \
  --num_points 100 \
  --batch_size 32 \
  --epochs 100 \
  --lr 5e-5 \
  --final_lr 5e-6 \
  --cond_kframes 4 \
  --train_noise_timesteps 100 \
  --align_weight 0.1 \
  --fp16 \
  --save_dir ckpts/flow/libero_object
```

The default model file is:

```text
3DFlowModel/flow_model.py
```

<a id="save-predicted-flow-to-hdf5"></a>

## 💾 Save Predicted Flow To HDF5

This step writes predicted trajectories into each demo as `pre_point_traj`.

```bash
CUDA_VISIBLE_DEVICES=0 python 3DFlowModel/predict_flow_hdf5.py \
  --hdf5_root /path/to/Flow_training_all/libero_object \
  --ckpt /path/to/siglip_flow_futureK_best.pt \
  --model_py 3DFlowModel/flow_model.py \
  --k_steps 20 \
  --ddim_steps 20 \
  --guidance_scale 2.0 \
  --frames_key frames_rgb \
  --query_key query_points \
  --instr_attr prompt \
  --out_key pre_point_traj \
  --window_bs 100 \
  --overwrite
```

2D debug overlays are disabled by default. To save them, add:

```bash
--debug_tracks_dir outputs/debug_tracks
```

<a id="visualization"></a>

## 🎥 Visualization

Interactive 3D HTML viewer:

```bash
python visualization/save_seg_all_starts_to_goal_3d_html.py \
  --flow_root /path/to/Flow_training_all/libero_object \
  --out_dir /path/to/outputs/html \
  --traj_key point_traj_base_metric \
  --scene_stride 2 \
  --max_scene_points 14000
```

Fixed-view 3D MP4 exports:

```bash
python visualization/save_3d_flow_views_mp4.py \
  --flow_root /path/to/Flow_training_all/libero_object \
  --out_dir /path/to/outputs/mp4 \
  --traj_key point_traj_base_metric \
  --views front side top
```

2D overlay helpers are also under `visualization/`.

<a id="train-action-policy"></a>

## 🧠 Train Action Policy

The policy training scripts expect processed HDF5 files with `pre_point_traj`.

```bash
cd Action_policy

CUDA_VISIBLE_DEVICES=0 python train/train_policy_dp_ema.py \
  --flow_root /path/to/Flow_training_all/libero_object \
  --k_steps 20 \
  --num_points 100 \
  --batch_size 64 \
  --epochs 40 \
  --lr 2e-4 \
  --final_lr 5e-5 \
  --fp16 \
  --save_dir ckpts/action/libero_object \
  --anneal_lr
```

<a id="evaluation"></a>

## 📊 Evaluation

LIBERO:

```bash
cd Action_policy

CUDA_VISIBLE_DEVICES=0 python eval/eval_libero_dp.py \
  --action_ckpt /path/to/best_action_policy_ema.pt \
  --policy_cfg model/model_res_dit_flow.yaml \
  --ckpt /path/to/siglip_flow_futureK_best.pt \
  --suit_type libero_goal
```

ManiSkill:

```bash
cd Action_policy

CUDA_VISIBLE_DEVICES=0 python eval/eval_maniskill_dp.py \
  --env_id PushCube-v1 \
  --model_cfg model/model_res_dit_maniskill.yaml \
  --action_ckpt /path/to/best_action_policy_ema.pt \
  --flow_ckpt /path/to/siglip_flow_futureK_best.pt \
  --flow_model_py ../3DFlowModel/flow_model.py
```

<a id="notes"></a>

## 📝 Notes

- `process_data/` and `visualization/` are the main public entry folders.
- Keep local-only experiments and archived files under `backup/`; this directory is intentionally ignored by git.
- Large local outputs, checkpoints, caches, and W&B runs should stay ignored by git.
- Third-party dependencies should be installed as documented checkouts, submodules, or explicit local paths; avoid hardcoding machine-specific paths in source files.

<a id="acknowledgements"></a>

## 🙏 Acknowledgements

RoboFlow4D builds on several excellent open-source projects and benchmark suites. We thank the authors and maintainers of [SpaTrackerV2](https://github.com/henry123-boy/SpaTrackerV2), [VGGT](https://github.com/facebookresearch/vggt), [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2), [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [SAM 2](https://github.com/facebookresearch/sam2), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [ManiSkill](https://github.com/haosulab/ManiSkill), [SigLIP / Big Vision](https://github.com/google-research/big_vision), [PyTorch](https://pytorch.org/), and the broader robotics and embodied AI open-source community.

Third-party code, models, and datasets remain under their original licenses. Please follow the upstream license terms when installing dependencies, downloading checkpoints, or preparing benchmark data.

<a id="citation"></a>

## ✍️ Citation

If you find this repository useful, please consider citing the paper below.

```bibtex
@inproceedings{anonymous2026roboflow4d,
  title={RoboFlow4D: A Lightweight Flow World Model Toward Real-Time Flow-Guided Robotic Manipulation},
  author={Anonymous Authors},
  booktitle={Proceedings of the International Conference on Machine Learning},
  year={2026}
}
```
