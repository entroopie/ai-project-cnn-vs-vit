# CNN vs Vision Transformer — Rice Variety Classification

**Course:** AI and Advanced Machine Learning (AAU Mini-Project)

A from-scratch comparison of a **CNN** and a **Vision Transformer (ViT)** on the [Rice Image Dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset). Both models are built in TensorFlow/Keras with no pretraining or transfer learning, isolating architectural design as the sole variable.

---

## Results (run `8f1ed8fb`)

| Model | Params | Test Accuracy | Test Loss | Train Time |
|-------|--------|---------------|-----------|------------|
| CNN   | 11.2 M | 98.71 %       | 0.0427    | 351 s      |
| ViT   | 0.55 M | **99.49 %**   | **0.0177**| 887 s      |

The ViT outperforms the CNN with **20× fewer parameters**, at the cost of ~2.5× longer training.

---

## Project Structure

```
.
├── cnn.py               # CNN model definition & training
├── vit.py               # ViT model definition & training
├── main.py              # Entry point — data loading, training, evaluation
├── results_writer.py    # Saves CSVs, plots, and results index per run
├── which_rice_are_you.py# Fun: classify any image with the rice model
├── Dockerfile           # GPU training container
├── Makefile             # Docker build/run shortcuts
├── requirements.txt     # Pip dependencies
├── models/              # Saved .keras model files (git-ignored)
├── results/             # CSVs, plots, results_index.json
└── report/              # LaTeX report source
```

---

## Quickstart (local, Apple Silicon)

```bash
# 1. Create venv with Python 3.11 (required for tensorflow-macos)
python3.11 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install tensorflow-macos tensorflow-metal pillow pandas scikit-learn matplotlib seaborn

# 3. Download the dataset from Kaggle and extract
unzip rice-image-dataset.zip -d .
# → Rice_Image_Dataset/{Arborio,Basmati,Ipsala,Jasmine,Karacadag}/

# 4. Train both models
python main.py
```

---

## Training on HPC (Docker + GPU)

```bash
# Build image
make build

# Run (mounts dataset, models, results from host)
make run
```

The Dockerfile trains both models sequentially and writes all outputs to `results/`.

---

## Which Rice Are You?

Load any image (yes, including a selfie) and see which rice variety the model thinks you are:

```bash
# Copy model from HPC first
scp user@hpc-host:~/ai-project-cnn-vs-vit/models/rice_vit.keras models/

# Run
python which_rice_are_you.py photo.jpg --model models/rice_vit.keras
```

```
==================================================
  You are... Basmati rice!
  Confidence: 94.3%

  Long, slender, and fragrant — you stand out in any crowd. 🌾
==================================================

Full probability breakdown:
  Basmati       94.3%  ████████████████████████████
  Jasmine        3.1%  █
  Arborio        1.4%
  Ipsala         0.8%
  Karacadag      0.4%
```

---

## Hyperparameters

| Hyperparameter          | Value     |
|-------------------------|-----------|
| Optimiser               | Adam      |
| Learning rate           | 1e-3      |
| Max epochs              | 20        |
| Batch size              | 32        |
| Early stopping patience | 5         |
| LR decay factor         | 0.5       |
| LR decay patience       | 3         |
| Min learning rate       | 1e-6      |

---

## Citation

M. Koklu et al., "Classification of Rice Varieties with Deep Learning Methods," *Computers and Electronics in Agriculture*, 2021.
