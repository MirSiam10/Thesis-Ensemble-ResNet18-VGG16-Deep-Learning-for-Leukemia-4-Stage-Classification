# 🩸 Ensemble Deep Learning for Leukemia Stage Classification

**Final Year B.Sc. Design Project — Department of Computer Science and Engineering, Daffodil International University**

An ensemble deep learning framework (**ResNet18 + VGG16**) that classifies **four stages of Acute Lymphoblastic Leukemia (ALL)** — *Benign, Early Pre-B, Pre-B, and Pro-B* — from microscopic Peripheral Blood Smear (PBS) images, achieving **99.49% test accuracy** with built-in explainability via **Guided Backpropagation**.

> 📄 Full report: [`Defence_Report.pdf`](./Defence_Report_4434_38_docx.pdf)

---

## 🎯 Motivation

Leukemia staging from blood smear images is traditionally done manually by hematopathologists — a process that is slow, expert-dependent, and prone to inter-observer variability, especially between visually similar stages like Pre-B and Pro-B. This project builds an automated, high-accuracy, and **interpretable** deep learning pipeline that could support (not replace) clinical decision-making.

## 🧠 Approach

| Stage | Details |
|---|---|
| **Preprocessing** | Resize (224×224), normalization, augmentation (rotation, flips, zoom, translation) |
| **Class balancing** | Oversampling to equal class counts + `WeightedRandomSampler` |
| **Baseline models** | EfficientNet-B0, MobileNetV3-Large, DenseNet121 (transfer learning, frozen backbones) |
| **Final model** | **Ensemble of ResNet18 + VGG16**, fused via soft-voting (averaged softmax probabilities) |
| **Training** | AdamW, OneCycleLR scheduler, mixed-precision (AMP), label smoothing, early stopping |
| **Explainability** | Guided Backpropagation saliency maps to visualize decision-relevant cellular regions |

**Why ensemble ResNet18 + VGG16?**
ResNet18's residual connections are strong at capturing high-level semantic structure, while VGG16's sequential 3×3 convolutions excel at fine-grained texture and morphology — a natural fit for distinguishing subtle nuclear/cytoplasmic differences across leukemia stages.

## 📊 Results

| Model | Validation Accuracy | Test Accuracy |
|---|---|---|
| MobileNetV3-Large (best baseline) | 94.62% | 93.20% |
| ResNet18 (standalone) | 97.10% | 96.26% |
| VGG16 (standalone) | 97.30% | 96.60% |
| **Ensemble (ResNet18 + VGG16)** | **98.64%** | **99.49%** |

**Per-class performance (Ensemble model):**

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Benign | 0.993 | 0.986 | 0.989 |
| Malignant – Pre-B | 1.000 | 1.000 | 1.000 |
| Malignant – Pro-B | 1.000 | 1.000 | 1.000 |
| Malignant – Early Pre-B | 0.987 | 0.993 | 0.989 |

Guided Backpropagation saliency maps confirmed the model consistently attends to clinically meaningful regions — cell nuclei, chromatin texture, and cytoplasmic patterns.

## 🛠️ Tech Stack

`Python` · `PyTorch` & `torchvision` · `scikit-learn` · `NumPy` / `Pandas` · `OpenCV` / `PIL` · `Matplotlib` / `Seaborn` · Jupyter Notebook · Google Colab (GPU)

## 📁 Repository Structure

```
├── thesis-code.ipynb          # Full pipeline: data loading → preprocessing →
│                               # baseline models → ensemble model → evaluation →
│                               # Guided Backpropagation explainability
├── Defence_Report.pdf         # Full written thesis report
└── README.md
```

## 🚀 Running the Notebook

1. Dataset: [Blood Cell Cancer (ALL) 4-class dataset](https://www.kaggle.com/datasets) on Kaggle
2. Update `DATA_DIR` in the notebook to point to your local/Colab dataset path
3. Run cells sequentially — GPU strongly recommended (mixed-precision training is used)

```bash
pip install torch torchvision scikit-learn numpy pandas opencv-python matplotlib seaborn
```
<img width="311" height="251" alt="sample" src="https://github.com/user-attachments/assets/27ec8803-119c-4f2a-a7b0-72e402865f90" />
<img width="333" height="140" alt="curve" src="https://github.com/user-attachments/assets/2f11d803-1460-4560-91e4-8df67fb89886" />


## 🔮 Future Work

- Validate on multi-institution / real hospital datasets for generalizability
- Multi-modal fusion with clinical/genomic data
- Model compression (pruning/quantization) for edge/mobile deployment
- Additional XAI methods (Grad-CAM, Integrated Gradients, SHAP)

## 👥 Authors

- **Mir Siam** — 213-15-4438
- **Farkulid Araf** — 213-15-4434

Supervised by **Ms. Umme Ayman** (Lecturer, Senior Scale) and co-supervised by **Md. Monarul Islam**, Dept. of CSE, Daffodil International University.

## 📄 License

This project is shared for academic and portfolio purposes. Please cite or credit the authors if you build on this work.
