<div align=center>

# Frequency Mining Empowered by Text Aggregation: A New Perspective on Document Image Tampering Detection

</div>

<div align="center">
  <a href="http://dlvc-lab.net/lianwen/"> <img alt="SCUT DLVC Lab" src="https://img.shields.io/badge/SCUT-DLVC_Lab-B32952?logo=Academia&logoColor=hsl"></a>
  <a href="https://ojs.aaai.org/index.php/AAAI/article/view/37122"> <img alt="AAAI 2026" src="https://img.shields.io/badge/AAAI-2026-58B822"></a>
  <a href="https://github.com/open-mmlab/mmsegmentation"> <img alt="MMSegmentation" src="https://img.shields.io/badge/Built_on-MMSegmentation_1.x-2376B7"></a>
<p></p>
</div>

This is the official repository of **TAFE-Net** (**T**ext **A**ggregation and multi-**F**requency **E**nhancement **Net**work), presented in the paper *"Frequency Mining Empowered by Text Aggregation: A New Perspective on Document Image Tampering Detection"* (**AAAI 2026 Oral**).

## 🗂️ Repository Structure — two datasets × two backbones

This bundle contains **two self-contained experiment folders, one per dataset**. Each folder is a full copy of the codebase and provides **both backbone variants** (SegFormer and ConvNeXt):

```
TAFENet-All/
├── README.md                    # ► this file (overview of both folders)
├── requirements.txt    # ► shared pip environment (see "Environment")
│
├── TAFENet-Doctamper/           # ► experiments on the DocTamper dataset
│   └── configs/tafenet/
│       ├── tafenet_segformer_doctamper.py   # SegFormer  (TAFE-Net)
│       └── tafenet_convnext_doctamper.py    # ConvNeXt   (TAFE-Net*)
│
└── TAFENet-RTM/                 # ► experiments on the RTM (RealTextManipulation) dataset
    └── configs/tafenet/
        ├── tafenet_segformer_rtm.py   # SegFormer  (TAFE-Net)
        └── tafenet_convnext_rtm.py    # ConvNeXt   (TAFE-Net*)
```

**Pick your experiment (dataset × backbone → folder + config + checkpoint):**

| Dataset | Backbone | Model | Folder | Config (relative to that folder) | Checkpoint |
|---------|----------|-------|--------|----------------------------------|------------|
| **DocTamper** | SegFormer (MiT-B2) | TAFE-Net  | `TAFENet-Doctamper/` | `configs/tafenet/tafenet_segformer_doctamper.py` | `tafenet_segformer_doctamper.pth` |
| **DocTamper** | ConvNeXt-V2        | TAFE-Net\*| `TAFENet-Doctamper/` | `configs/tafenet/tafenet_convnext_doctamper.py`  | `tafenet_convnext_doctamper.pth` |
| **RTM**       | SegFormer (MiT-B2) | TAFE-Net  | `TAFENet-RTM/`       | `configs/tafenet/tafenet_segformer_rtm.py`       | `tafenet_segformer_rtm.pth` |
| **RTM**       | ConvNeXt-V2        | TAFE-Net\*| `TAFENet-RTM/`       | `configs/tafenet/tafenet_convnext_rtm.py`        | `tafenet_convnext_rtm.pth` |

The config filename encodes both choices: `tafenet_<backbone>_<dataset>.py`. All commands below are run **from inside the chosen folder** (`cd TAFENet-Doctamper` or `cd TAFENet-RTM`).

## ⚒️ Environment

All dependencies are pinned in **`requirements.txt`** (bundle root):

```bash
conda create --name tafe python=3.8 -y
conda activate tafe
pip install -r requirements.txt
```

## 📥 Data & Weights

### Datasets

| Dataset | Used by folder | Link |
|---------|----------------|------|
| **DocTamper**            | `TAFENet-Doctamper/` | [GitHub: qcf-568/DocTamper](https://github.com/qcf-568/DocTamper) |
| **RTM / RealTextManip.** | `TAFENet-RTM/`       | [Google Drive](https://drive.google.com/file/d/11AHZ8ih_kDCFilGceevppcGkKR4vDJD2/view?usp=sharing) · [GitHub: DrLuo/RTM](https://github.com/DrLuo/RTM) |

**DocTamper** (`TAFENet-Doctamper/`) — trains/tests on **DocTamperV1 (LMDB format)**. The dataloader reads LMDB directly, so set the `db_path` values inside the config (`train_pipeline` / `test_pipeline` → `LoadImageLabelFromFileLMDB`). Expected layout:

```
DocTamperV1/
├── DocTamperV1-TrainingSet/   # training
├── DocTamperV1-TestingSet/    # test
├── DocTamperV1-FCD/           # cross-domain FCD
└── DocTamperV1-SCD/           # cross-domain SCD
```

The `TAFENet-Doctamper/pks/` folder ships the JPEG re-compression quality tables (`DocTamperV1-{FCD,SCD,TestingSet}_75.pk`, i.e. quality 75) that the test pipeline references via `compress_pk=...`; update that path if you move the folder.

**RTM** (`TAFENet-RTM/`) — trains/tests on **RealTextManipulation**. Set the paths in `configs/_base_/datasets/rtm_crop.py` (`data_root`, and the `train.txt` / `test.txt` `ann_file`s). Images and masks are paired by name via `data_prefix=dict(img_path='JPEGImages', seg_map_path='SegmentationClass')`; the dataset class is `TamperedTextDataset`, labels are read as binary (`LoadAnnotations(binary=True)`). Expected layout:

```
RealTextManipulation/
├── JPEGImages/          # RGB document images
├── SegmentationClass/   # binary tamper masks (0 = authentic, 255 = tampered)
├── train.txt            # training split (one image name per line)
└── test.txt             # test split
```

### Weights (Baidu Netdisk)

All pretrained backbones and trained checkpoints are on Baidu Netdisk under **`TAFE-Net-weight/`**:

> 🔗 **[Baidu Netdisk `TAFE-Net-weight/`](https://pan.baidu.com/s/17bgrqf7erlV2ORvvF0R1bQ?pwd=tafe)** — access code: `tafe`

**1) Pretrained backbones** — copy into the **root of the folder you run** (`TAFENet-Doctamper/` or `TAFENet-RTM/`):

| Item | What it is | Used by | Place at |
|------|-----------|---------|----------|
| `mit_b2_20220624-66e8bf70.pth` | SegFormer **MiT-B2** ImageNet weights | main backbone of the two SegFormer configs **and** the low-frequency Transformer branch (`HubVisionTransformer0521`) of **all four** configs | `<folder>/mit_b2_20220624-66e8bf70.pth` |
| `ss1/` | timm cache of ConvNeXt-V2-**tiny** (`convnextv2_tiny.fcmae_ft_in22k_in1k_384`) | high-frequency **CNN branch** of **all four** configs | `<folder>/ss1/` |
| `ss2/` | timm cache of ConvNeXt-V2-**base** (`convnextv2_base.fcmae_ft_in22k_in1k_384`) | main visual encoder of **TAFE-Net\*** — the two **ConvNeXt** configs only | `<folder>/ss2/` |

`ss1/` and `ss2/` are `timm` download caches: the backbones build with `pretrained=True, cache_dir='./ss1'` (or `'./ss2'`). Dropping the provided folders in place lets `timm` reuse them offline instead of re-downloading.

**2) Trained TAFE-Net checkpoints** — pass as `<checkpoint.pth>` to `dist_test.sh` (see [Inference](#-inference)):

| Checkpoint | Folder | Config |
|-----------|--------|--------|
| `tafenet_segformer_doctamper.pth` | `TAFENet-Doctamper/` | `configs/tafenet/tafenet_segformer_doctamper.py` |
| `tafenet_convnext_doctamper.pth`  | `TAFENet-Doctamper/` | `configs/tafenet/tafenet_convnext_doctamper.py` |
| `tafenet_segformer_rtm.pth`       | `TAFENet-RTM/`       | `configs/tafenet/tafenet_segformer_rtm.py` |
| `tafenet_convnext_rtm.pth`        | `TAFENet-RTM/`       | `configs/tafenet/tafenet_convnext_rtm.py` |

## 🔥 Training

Each folder's `train.sh` is the entry point. Run it **from inside that folder**:

```bash
cd TAFENet-Doctamper       # or: cd TAFENet-RTM
bash train.sh
```

## 🚀 Inference

Each folder's `infer.sh` runs evaluation — run it **from inside that folder**:

```bash
cd TAFENet-Doctamper       # or: cd TAFENet-RTM
bash infer.sh
```

Add `--mask` to the `dist_test.sh` line in `infer.sh` to also export the predicted binary masks.

## 📅 Evaluation

Metrics (IoU / Precision / Recall / F1) are computed automatically during testing (`BinaryIoUMetric`; the DocTamper configs additionally report `DocTamperMetric`). To score exported masks offline, use the standalone tools in each folder's `EvalRTM/`:

```bash
# DocTamper (in TAFENet-Doctamper/) — metrics match the in-training DocTamperMetric
python EvalRTM/eval_doctamper1.py --pred_dir ${PRED_FOLDER} --gt_dir ${GT_FOLDER}

# RTM (in TAFENet-RTM/) — binary-mask metrics with a per-manipulation-type breakdown.
# The method name is taken from the prediction-folder basename (name it {MethodName}_mask).
python EvalRTM/run_eval.py --pred_dir ${PRED_FOLDER} --gt_dir ${RTM_GT_FOLDER} [--save_dir ${OUT_DIR}]
```

## 📫 Contact

If you have any questions, feel free to contact us at eezqyi@mail.scut.edu.cn.

## 💙 Acknowledgement

- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- [SegFormer](https://github.com/NVlabs/SegFormer)
- [ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2)
- [DocTamper / DTD](https://github.com/qcf-568/DocTamper)
- [RTM](https://github.com/DrLuo/RTM)
- [FFDN](https://github.com/Rapisurazurite/FFDN)

## 📜 License

The code should be used and distributed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) for non-commercial research purposes.

## ⛔️ Copyright

- Copyright 2026, [Deep Learning and Vision Computing Lab (DLVC-Lab)](http://www.dlvc-lab.net), South China University of Technology.

## ✒️ Citation

If you find this work helpful, please consider giving this repo a ⭐ and citing:

```latex
@inproceedings{yi2026frequency,
  title={Frequency Mining Empowered by Text Aggregation: A New Perspective on Document Image Tampering Detection},
  author={Yi, Ziqi and Xu, Guitao and Wu, Shihang and Zhang, Peirong and Jin, Lianwen},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={2},
  pages={1471--1479},
  year={2026}
}
```
