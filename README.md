# CNN vs Vision Transformer — Rice Variety Classification

**Course:** AI and Advanced Machine Learning — AAU Mini-Project  
**Goal:** Compare a custom CNN and a custom ViT trained from scratch on the Rice Image Dataset.

---

## Results (run `8f1ed8fb`)

| Model | Params | Test Accuracy | Test Loss | Train Time |
|-------|--------|--------------|-----------|------------|
| CNN   | 11.2M  | 98.71%       | 0.0427    | 351 s      |
| ViT   | 0.55M  | **99.49%**   | **0.0177**| 887 s      |

The ViT outperforms the CNN with ~20× fewer parameters. Full results are in `results/`.

---

## Repo Structure

```
cnn.py               # CNN model definition and training
vit.py               # ViT model definition and training
main.py              # Entry point — runs both models end-to-end
results_writer.py    # Saves metrics, plots, and results index
which_rice_are_you.py # Fun: classify any image with the trained model
requirements.txt     # Python dependencies
Dockerfile           # GPU training container (TF 2.17, CUDA)
Makefile             # Docker build/run shortcuts
results/             # CSV metrics, plots, results_index.json
report/              # LaTeX report source
models/              # Saved .keras models (not tracked in git)
```

---

## Branches

| Branch | Purpose |
|--------|---------|
| `main` | Stable, full training pipeline |
| `which-rice-are-you` | Fun classifier — run any image through the model |

---

## Setup

### Local (Mac / Linux) — venv

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install tensorflow-macos pillow   # macOS
# pip install tensorflow pillow       # Linux
```

### GPU Cluster — Docker

```bash
make build   # build image (tensorflow:2.17.0-gpu)
make run     # train both models (mounts dataset + results)
make stop    # stop container
make clean   # remove image
```

The container expects the dataset at `./Rice_Image_Dataset/`. Download from:  
https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset

```bash
unzip archive.zip
# → Rice_Image_Dataset/{Arborio,Basmati,Ipsala,Jasmine,Karacadag}/
```

---

## Training

```bash
python main.py
```

Trains CNN then ViT with a 70/15/15 stratified split (seed 42). Results are written to `results/` with a unique run ID.

---

## Which Rice Are You?

Copy a trained model from the cluster:

```bash
mkdir -p models
scp user@host:~/ai-project-cnn-vs-vit/models/rice_vit.keras models/
```

Then run:

```bash
python which_rice_are_you.py photo.jpg --model models/rice_vit.keras
```

---

## Source Code

https://github.com/entroopie/ai-project-cnn-vs-vit
