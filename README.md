<h2 align="center">
  <b>Beyond the Frame:<br>Generating 360° Panoramic Videos from Perspective Videos</b>
</h2>

<div align="center">
    <a href="https://arxiv.org/abs/2504.07940" target="_blank">
    <img src="https://img.shields.io/badge/Paper-ICCV 2025-red" alt="Paper"/></a>
    <a href="https://red-fairy.github.io/argus/" target="_blank">
    <img src="https://img.shields.io/badge/Demo-Project Page-blue" alt="Project Page"/></a>
</div>

---

This is the official repository for the ICCV 2025 paper:  
**Beyond the Frame: Generating 360° Panoramic Videos from Perspective Videos**

**Authors:** Rundong Luo<sup>1</sup>, Matthew Wallingford<sup>2</sup>, Ali Farhadi<sup>2</sup>, Noah Snavely<sup>1</sup>, Wei-Chiu Ma<sup>1</sup>  
<sup>1</sup>Cornell University, <sup>2</sup>University of Washington

---

## 📝 Abstract

360° videos have emerged as a promising medium to represent our dynamic visual world. Compared to the "tunnel vision" of standard cameras, their borderless field of view offers a more holistic perspective of our surroundings. However, while existing video models excel at producing standard videos, their ability to generate full panoramic videos remains elusive. In this paper, we investigate the task of **video-to-360° generation**: given a perspective video as input, our goal is to generate a full panoramic video that is coherent with the input. Unlike conventional video generation tasks, the output's field of view is significantly larger, and the model is required to have a deep understanding of both the spatial layout of the scene and the dynamics of objects to maintain geometric and dynamic consistency with the input. To address these challenges, we first leverage the abundant 360° videos available online and develop a high-quality data filtering pipeline to curate pairwise training data. We then carefully design a series of geometry- and motion-aware modules to facilitate the learning process and improve the quality of 360° video generation. Experimental results demonstrate that our model can generate realistic and coherent 360° videos from arbitrary, in-the-wild perspective inputs. Additionally, we showcase its potential applications, including video stabilization, camera viewpoint control, and interactive visual question answering.

<!-- ---

## ✅ Checklist

- ✅ Inference code and pretrained models released  
- ✅ Training code  
- ✅ Dataset release -->

---

## 🔧 Installation

The framework is divided into three independent modules:

- **360° Video Generation**  
- **Camera Trajectory Prediction**  
- **Video Enhancement**  

Each module requires a separate Python environment due to potential package conflicts.

### 1. Clone the Repository

```bash
git clone --recurse-submodules https://github.com/Red-Fairy/argus-code
```

If you already cloned the repo without submodules:

```bash
git submodule update --init --recursive
```

### 2. 360° Video Generation (`360VG` environment)

- Create a conda environment named `360VG`.
- Install compatible versions of `torch` and `torchvision`.
- Install dependencies from `requirements.txt`.

**Note:**  
Before installing the other packages, install `numpy<2` to ensure compatibility with `faiss`.

### 3. Camera Trajectory Prediction

We support two methods: **MASt3R** and **MegaSaM**.

#### Option A: MASt3R (Compatible with `360VG`)

```bash
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
git clone https://github.com/jenicek/asmk
cd asmk/cython/
cythonize *.pyx
cd ..
pip install .
cd ..
```

#### Option B: MegaSaM (`mega_sam` environment)

We recommend creating a new conda environment for MegaSaM due to package conflicts.  
Follow the instructions in the [MegaSaM repository](https://github.com/mega-sam/mega-sam).

**Required checkpoints:**

- [DepthAnything](https://huggingface.co/spaces/LiheYoung/Depth-Anything/blob/main/checkpoints/depth_anything_vitl14.pth) → `mega-sam/Depth-Anything/checkpoints/depth_anything_vitl14.pth`
- [RAFT](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT) → `mega-sam/cvd_opt/raft-things.pth`

**To visualize camera trajectories:**

```bash
pip install viser pyliblzfse
```

### 4. Video Enhancement (`venhancer` environment)

We recommend a separate conda environment named `venhancer`. Refer to [VEnhancer](https://github.com/Vchitect/VEnhancer) to set up the enhancement pipeline.

---

## 🚀 Inference

First, download our pretrained model from [Google Drive](https://drive.google.com/drive/folders/1mZpViQY2yvwav-CcxdsQouYu8t52b25G?usp=sharing) and place it in the `checkpoints` folder.

### Option 1: Interactive Demo

Launch the Gradio demo:

```bash
python gradio_demo.py
```

To specify environments for depth and pose estimation:

```bash
python gradio_demo.py \
  --mono_depth_env mega_sam \
  --camera_pose_env mega_sam \
  --venhancer_env venhancer
```

You can configure generation parameters, select calibration methods, and enable enhancement through the UI.

### Option 2: Command Line Interface

Run:

```bash
bash scripts/test/inference.sh \
  [PATH_TO_UNET] \
  [PATH_TO_VIDEO_OR_VIDEO_FOLDER] \
  [PATH_TO_SAVE_FOLDER] \
  [GUIDANCE_SCALE] \
  [NUM_INFERENCE_STEPS]
```

---

## 📦 Data Preparation

### 1. Download the Videos

Download the videos listed in [`process_data/clips_info.jsonl`](process_data/clips_info.jsonl).  
We recommend using the [`yt-dlp`](https://github.com/yt-dlp/yt-dlp) tool to download the videos from YouTube.

### 2. Segment Videos into 10-Second Clips

Split each downloaded video into 10-second clips.  
The start and end frames of each clip are specified in [`process_data/clips_info.jsonl`](process_data/clips_info.jsonl).

As described in the supplementary material, our data processing pipeline consists of the following steps:
1.	Format Filtering. We first sample frames and compute the horizontal image gradient along the 180° boundary to verify whether the frame is 360°. We also compute the vertical gradient at the image center to remove videos that are split into two halves.
2.	Intra-frame Filtering. To detect improperly formatted videos, we compute LPIPS between the left and right halves to filter out 180° videos, and between the top and bottom halves to filter incorrectly formatted 360° videos.
3.	Inter-frame Filtering. To ensure temporal dynamics, we sample frames at random intervals and compute pixel variance. Static videos with minimal inter-frame variation are discarded. Videos with excessive black pixels are also removed.
4.	Clip-level filtering. We divide videos into 10-second clips and further filter them by (i) low optical flow magnitude using RAFT, (ii) shot boundary detection with TransNet-v2, and (iii) text detection with DPText-DETR applied to unwrapped perspective views. All these operations are performed on six 90° FoV projections of each video.

We provide sample scripts in [`process_data/pipeline`](process_data/pipeline) to run the data processing pipeline for the first three steps. **You don't need to run the pipeline if you are using our provided clips.**

### 3. Training Clip Selection

The clips used for training are listed in two files:

- [`process_data/clips_filtered.txt`](process_data/clips_filtered.txt): used for **Stage 1** training.
- [`process_data/clips_filtered_high_quality.txt`](process_data/clips_filtered_high_quality.txt): used for **Stage 2** training with stricter filtering criteria.

Each line in these files is formatted as: `video_category\tvideo_id\tclip_id`.

---

## 📊 Training

### Stage 1: Initial Training at 384×768 Resolution

Begin by training the model using video clips at a resolution of 384×768.  
Before running the script, update the following environment variables in scripts/train/train.sh:

- TRAIN_DATASET_PATH and VAL_DATASET_PATH: directories containing the training and validation video clips.
- TRAIN_CLIP_FILE_PATH and VAL_CLIP_FILE_PATH: text files listing the clip metadata for training and validation, formatted as: `video_category\tvideo_id\tclip_id`.

To launch the training process:

```bash
bash scripts/train/train.sh [EXPERIMENT_NAME] 384 100000
```

### Stage 2: Fine-Tuning at 512×1024 Resolution

Next, fine-tune the model using high-quality clips at a resolution of 512×1024.  
Again, modify the same four paths in scripts/train/train.sh as described above.

Additionally, provide the path to the pretrained UNet model from Stage 1.

To start Stage 2 training:

```bash
bash scripts/train/train.sh [EXPERIMENT_NAME] 512 20000 [PRETRAIN_UNET_PATH_OF_STAGE_1]
```

## 📊 Evaluation

The 101 evaluation clips and the real/simulation camera trajectories can be found here: [Google Drive](https://drive.google.com/drive/folders/1F_RyoDz2r3XSBEzII0Un53ErTk3b8V8S?usp=sharing). For the trajectories, the FOVs are in degrees and the row/pitch/yaw are in radians. In our convention, the positive yaw is rotating rightward, the positive pitch is looking upward, and the positive row is tilting rightward.

## 📖 Citation

If you find our work useful for your research or projects, please cite our paper:

```bibtex
@inproceedings{luo2025beyond,
  title     = {Beyond the Frame: Generating 360° Panoramic Videos from Perspective Videos},
  author    = {Luo, Rundong and Wallingford, Matthew and Farhadi, Ali and Snavely, Noah and Ma, Wei-Chiu},
  booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025},
  url       = {https://arxiv.org/abs/2504.07940}
}
```

## License
This project is released under the MIT License.