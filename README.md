# Monash Admission Task – CLIP on CIFAR-10

This project was developed as part of the Monash admission assessment.  
It focuses on exploring **Contrastive Language-Image Pretraining (CLIP)** and its application to **zero-shot image classification** and **linear probe evaluation** on the CIFAR-10 dataset.

---

## 📌 Project Overview

- Implement zero-shot classification on CIFAR-10 using **OpenCLIP (ViT-B-32)**.
- Train a **linear probe classifier** on top of CLIP’s frozen image embeddings.
- Explore **prompt engineering**: analyze how variations in class names influence CLIP’s performance.
- Provide a **well-documented codebase** for reproducibility and further experimentation.

---

## 📂 Repository Structure

```
monash-admission-task/
│
├── src/
│   ├── main.ipynb        # Main notebook containing experiments and results
│   ├── utils.py          # Helper functions (if applicable)
│   └── ...
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/pabdzadeh/monash-admission-task.git
cd monash-admission-task
pip install -r requirements.txt
```

Alternatively, you can run the notebook directly on **[Google Colab](https://colab.research.google.com/github/pabdzadeh/monash-admission-task/blob/main/src/main.ipynb#scrollTo=2fg_4ktaTmj_)**.

---

## 🚀 Usage
### Python Notebook 
1. Open `src/main.ipynb` in Jupyter Notebook or [Google Colab](https://colab.research.google.com/github/pabdzadeh/monash-admission-task/blob/main/src/main.ipynb#scrollTo=2fg_4ktaTmj_).
2. Run the cells step by step:
   - **Load CLIP (ViT-B-32)** from the OpenCLIP repo.
   - **Download CIFAR-10** dataset.
   - Perform **zero-shot classification**.
   - Train and evaluate a **linear probe**.
   - Compare results across different **prompt variations**.

### 🚀 Python Usage

Run `main.py` with different arguments to perform **zero-shot classification** or **linear probe training** using various pretrained CLIP models.  
You can also experiment with different prompt variations, prefixes, and postfixes for class names.

**Examples:**

- **Zero-shot classification with default class names**  
  ```bash
  python src/main.py --zero_shot=True --pretrained_model=ViT-B-32
- **Zero-shot classification with descriptive prompts**  
  ```bash
  python src/main.py --zero_shot True --class_name_type 2
  # class_name_type=2 corresponds to "a photo of an airplane" (and similar for all classes)

- **Linear probe training for 50 epochs**
   ```bash
   python src/main.py --linear_probe True --train_epochs 50 --batch_size 64 --pretrained_model ViT-B-32

- **Resume linear probe training from a checkpoint**
   ```bash
   python src/main.py --linear_probe True --resume_from_checkpoint ./checkpoints/ckpt.pth

- **Add a prefix/postfix to all class names**
   ```bash
   python src/main.py --zero_shot True --class_name_prefix "a photo of " --class_name_postfix " in CIFAR-10"



---

## 📊 Results
### Pretrained on laion2b_s34b_b79k
| Experiment                                                                               | Accuracy (%) |
|------------------------------------------------------------------------------------------|--------------|
| Zero-shot (default prompts)                                                              | 71.7         |
| Zero-shot (max of engineered prompts)                                                    | 91.1         |
| Linear probe  with CLIP's Regularized Logistic Regression and L-BFGS (ViT-B-32 features) | 96.8         |
| Linear probe Simple Logistic Regression 10-epoch (ViT-B-32 features)                     | 96.9         |

### Pretrained on datacomp_xl_s13b_b90k
| Experiment                                                                               | Accuracy (%) |
|------------------------------------------------------------------------------------------|--------------|
| Zero-shot (default prompts)                                                              | 95.2         |
| Zero-shot (max of engineered prompts)                                                    | 92.3         |


---

## 🔍 Key Insights

- Zero-shot CLIP shows strong performance without training, but **prompt wording** significantly affects accuracy.
- Linear probing on CLIP embeddings further improves results compared to zero-shot.
- Synonyms, adjectives, and natural language descriptions for class names can shift performance.

---

## 📌 Future Work

- Extend experiments to **CIFAR-100** or **ImageNet** subsets.
- Explore **prompt ensembling** for improved zero-shot classification.
- Evaluate other CLIP backbones (e.g., ViT-B/16, RN50).

---

## 🤝 Acknowledgements

- [Radford et al., 2021: Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020)  
- [OpenCLIP Repository](https://github.com/mlfoundations/open_clip)
