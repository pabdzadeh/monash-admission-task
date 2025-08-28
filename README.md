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
│   ├── main.py          # Main python file to run
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

## 📝 CIFAR-10 Prompt Variations

This project explores how **different textual descriptions (prompts)** for class names can influence CLIP’s performance in both zero-shot classification and linear probing.  
Each CIFAR-10 class is associated with **six prompt variations**, ranging from short labels to descriptive natural language.  

| Class          | Prompt Variations                                                                                |
|----------------|--------------------------------------------------------------------------------------------------|
| **airplane**   | airplane, Airplane, a photo of an airplane, a flying airplane, a passenger airplane, an aircraft |
| **automobile** | automobile, car, a photo of a car, a small car, a sports car, a vehicle                          |
| **bird**       | bird, a bird, a small bird, a colorful bird, a flying bird, wild bird                            |
| **cat**        | cat, a cat, domestic cat, a small cat, a cute cat, kitten                                        |
| **deer**       | deer, a deer, a wild deer, a forest deer, a brown deer, stag                                     |
| **dog**        | dog, a dog, puppy, a small dog, a cute dog, domestic dog                                         |
| **frog**       | frog, a frog, a green frog, a small frog, amphibian frog, pond frog                              |
| **horse**      | horse, a horse, a brown horse, a white horse, a running horse, stallion                          |
| **ship**       | ship, a ship, a boat, a large ship, a sailing ship, cargo ship                                   |
| **truck**      | truck, a truck, a large truck, a delivery truck, a heavy truck, lorry                            |

---

📌 **Why is this important?**  
- Tests whether **context-rich phrases** (e.g., `"a colorful bird"`) outperform short labels (e.g., `"bird"`).  
- Evaluates CLIP’s **sensitivity to prompt engineering**.  
- Supports systematic experimentation by selecting a **prompt type index** (`--class_name_type`) or adding **prefixes/postfixes**.  

This makes it easy to reproduce results and explore new linguistic variations.

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

## ⚙️ Command-Line Arguments

The training and evaluation pipeline can be customized using the following arguments:

| Argument                        | Type | Default | Description                                                                                                                                    |
|---------------------------------|------|---------|------------------------------------------------------------------------------------------------------------------------------------------------|
| `--batch_size`                  | int  | 16      | Batch size for training and evaluation.                                                                                                        |
| `--linear_probe_type`           | str  | "exact" | Type of linear probe to use: `Simple` or `Exact`.                                                                                              |
| `--output_dir`                  | str  | "."     | Directory to save checkpoints and results.                                                                                                     |
| `--class_name_type`             | int  | 0       | Index of class name variation (e.g., `0 = airplane`, `1 = Airplane`, `2 = a photo of an airplane`, etc.). Applies across all CIFAR-10 classes. |
| `--class_name_prefix`           | str  | ""      | Prefix to be added to all class names. Useful for prompt engineering.                                                                          |
| `--class_name_postfix`          | str  | ""      | Postfix to be added to all class names. Useful for prompt engineering.                                                                         |
| `--resume_from_checkpoint`      | str  | None    | Path to resume training from a saved checkpoint.                                                                                               |
| `--pretrained_model`            | str  | None    | Name of pretrained CLIP variant (e.g., `"ViT-B-32"`).                                                                                          |
| `--linear_probe`                | bool | False   | If `True`, trains a linear probe on CLIP embeddings.                                                                                           |
| `--zero_shot`                   | bool | True    | If `True`, evaluates using zero-shot classification.                                                                                           |
| `--train_epochs`                | int  | 100     | Number of training epochs for the simple linear probe model.                                                                                   |

---

📌 **Usage Example**:
    ```
    python src/main.py --batch_size 32 --linear_probe True --train_epochs 50 --class_name_type 2 --output_dir ./checkpoints
    ```
    
## 📊 Results
### Pretrained on laion2b_s34b_b79k
| Experiment                                                                               | Accuracy (%) |
|------------------------------------------------------------------------------------------|--------------|
| Zero-shot (default prompts)                                                              | 71.7         |
| Zero-shot (max of engineered prompts)                                                    | 91.1         |
| Linear probe  with CLIP's Regularized Logistic Regression and L-BFGS (ViT-B-32 features) | 96.8         |
| Linear probe Simple Logistic Regression 10-epoch (ViT-B-32 features)                     | 96.9         |

### Pretrained on datacomp_xl_s13b_b90k
| Experiment                                                                              | Accuracy (%) |
|-----------------------------------------------------------------------------------------|--------------|
| Zero-shot (default prompts)                                                             | 95.2         |
| Zero-shot (max of engineered prompts)                                                   | 95.3         |
| Linear probe with CLIP's Regularized Logistic Regression and L-BFGS (ViT-B-32 features) | 97.8         |


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
