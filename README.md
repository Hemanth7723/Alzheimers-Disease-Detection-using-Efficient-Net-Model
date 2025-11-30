# Alzheimer's Disease Detection using EfficientNet Model 🧠🔬

[![Repository](https://img.shields.io/badge/repo-Alzheimers--Disease--Detection--using--EfficientNet--Model-blue)](https://github.com/Hemanth7723/Alzheimers-Disease-Detection-using-Efficient-Net-Model)
[![Languages](https://img.shields.io/github/languages/top/Hemanth7723/Alzheimers-Disease-Detection-using-Efficient-Net-Model)](https://github.com/Hemanth7723/Alzheimers-Disease-Detection-using-Efficient-Net-Model)
[![Last commit](https://img.shields.io/github/last-commit/Hemanth7723/Alzheimers-Disease-Detection-using-Efficient-Net-Model)](https://github.com/Hemanth7723/Alzheimers-Disease-Detection-using-Efficient-Net-Model/commits/main)

A reproducible, notebook-driven project for classifying Alzheimer's disease severity (or presence) from brain MRI images using transfer learning with the EfficientNet family. The notebooks cover data ingestion, preprocessing, augmentation, EfficientNet-based model building, training, cross-validation, evaluation, explainability (Grad-CAM), and guidance for deployment.

---

## **Table of contents**
- [**Project overview**](#project-overview)
- [**Dataset & sources**](#dataset--sources)
- [**Repository structure**](#repository-structure)
- [**Primary goals & experiments**](#primary-goals--experiments)
- [**Quick start**](#quick-start)
  - [**Run in Google Colab**](#run-in-google-colab)
  - [**Run locally**](#run-locally)
- [**Environment & dependencies**](#environment--dependencies)
- [**Notebook workflow (what each notebook does)**](#notebook-workflow-what-each-notebook-does)
- [**Data preprocessing & augmentation details**](#data-preprocessing--augmentation-details)
- [**Modeling details: EfficientNet & transfer learning**](#modeling-details-efficientnet--transfer-learning)
- [**Training, evaluation & metrics**](#training-evaluation--metrics)
- [**Explainability / Interpretability**](#explainability--interpretability)
- [**Model export & Streamlit demo**](#model-export--streamlit-demo)
- [**Reproducibility & checkpoints**](#reproducibility--checkpoints)
- [**Hardware recommendations**](#hardware-recommendations)
- [**Troubleshooting & tips**](#troubleshooting--tips)
- [**Extending & customizing experiments**](#extending--customizing-experiments)
- [**Ethical considerations & clinical caution**](#ethical-considerations--clinical-caution)
- [**Licenses & acknowledgements**](#licenses--acknowledgements)
- [**Contributing**](#contributing)

---

## **Project overview**
This project explores using EfficientNet (B0–B7 variants) as a backbone for detecting Alzheimer's disease from MRI scans. The work focuses on:
- Building robust preprocessing pipelines for MRI image data
- Applying transfer learning from EfficientNet pretrained weights
- Training and evaluating classification models with cross-validation
- Interpreting model predictions with Grad-CAM/visual explanations
- Documenting experiments in self-contained Jupyter notebooks for reproducibility

This repository is notebook-centric to make it easy for researchers and learners to follow each step interactively.

---

## **Dataset & sources**
Primary dataset used for training and experiments in this project:
- OASIS MRI dataset on Kaggle: https://www.kaggle.com/datasets/pulavendranselvaraj/oasis-dataset

Notes about the data source:
- The OASIS dataset (as provided on Kaggle) contains labeled MRI images suitable for Alzheimer’s research and classification experiments. Please review the dataset page for details, licensing, and attribution requirements before reuse.
- If you use ADNI or other datasets instead, please follow their access and usage rules and update notebook metadata accordingly.

Dataset characteristics to expect:
- MRI slices or preprocessed 2D images (sometimes 3D volumes)
- Labels like: `Normal`, `very mild`, `Mild Impairment`, `Moderate Alzheimer's Disease (AD)` — or binary labels (AD vs. Non-AD)
- Possible class imbalance (common), missing values, and varying image shapes/resolutions

Important: If using ADNI or other restricted datasets, ensure you follow the dataset license/usage policies.

---

## **Repository structure**
A recommended/typical structure used by the notebooks in this repo (actual filenames may vary; open the repo to confirm exact names):

- data/
  - raw/ (original images, if included)
  - processed/ (resized, split datasets)
- models/
  - checkpoints/ (saved model weights .h5 / .pt)
  - final/ (best saved models)
- app/
  - streamlit_app.py (or app.py) — Streamlit frontend for custom 
- utils/
  - data_utils.py (dataset loaders, transforms)
  - viz_utils.py (plot & Grad-CAM helpers)
- requirements.txt or environment.yml
- README.md

---

## **Primary goals & experiments**
- Baseline: Train a simple CNN / logistic baseline for sanity check
- Transfer learning: Fine-tune EfficientNetB0/B3/B4 (experiment with deeper variants if resources allow)
- Data augmentation: Evaluate augmentation schemes (rotation, flip, intensity, elastic transforms)
- Cross-validation: K-fold strategy to estimate generalization
- Interpretability: Produce Grad-CAM visualizations for model predictions

Document each experiment (hyperparameters, date, metrics) in the notebooks for traceability.

---

## **Quick start**

### **Run in Google Colab (recommended for quick GPU access)**
1. Open a notebook from the repository in Colab:
   - Visit a notebook file on GitHub and click "Open in Colab" or use `https://colab.research.google.com/github/<USER>/<REPO>/blob/main/notebooks/<NOTEBOOK>.ipynb`.
2. Mount Google Drive if you want to read/write data or save models persistently:
```python
from google.colab import drive
drive.mount('/content/drive')
```
3. Install dependencies (if `requirements.txt` exists):
```bash
!pip install -r requirements.txt
```
4. Run the cells in order. When training, choose "Runtime > Change runtime type" and select GPU (preferably Tesla T4, P100, or better).

### **Run locally**
1. Clone the repository:
```bash
git clone https://github.com/Hemanth7723/Alzheimers-Disease-Detection-using-Efficient-Net-Model.git
cd Alzheimers-Disease-Detection-using-Efficient-Net-Model
```
2. Create & activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate    # macOS / Linux
venv\Scripts\activate       # Windows
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Start Jupyter Lab or Notebook:
```bash
jupyter lab
# or
jupyter notebook
```
5. Open the notebooks and run them sequentially (start with data exploration).

---

## **Environment & dependencies**
Typical packages used in this repo (exact versions in requirements.txt if present):
- Python 3.8+
- numpy, pandas
- scikit-learn
- matplotlib, seaborn, plotly (optional)
- Pillow, imageio
- albumentations (recommended) or torchvision transforms
- TensorFlow (tensorflow or tensorflow-gpu) >=2.x or PyTorch >=1.8 (notebooks will indicate which framework is used)
- efficientnet (tf.keras.applications includes EfficientNet in TF >= 2.3) or timm (for PyTorch)
- opencv-python
- sklearn, imbalanced-learn (for resampling approaches)
- shap, captum, or tf-keras-vis (for interpretability)
- streamlit (for demo app) — see `requirements-streamlit.txt` for minimal pinning

If both TF and PyTorch code appear in the repo, pick the notebook that matches your preferred framework.

---

## **Notebook workflow (what each notebook does)**
- 00-data-exploration.ipynb
  - Load sample images, visualize class distribution, check for corrupted files, basic EDA
- 01-preprocessing-and-augmentation.ipynb
  - Standardization, resizing to EfficientNet input size (e.g., 224x224, 240x240), augmentation recipes
  - Train/validation/test split with stratification
- 02-training-efficientnet.ipynb
  - Build model: EfficientNet base + custom classifier head
  - Freeze/unfreeze strategy, learning rate schedules, optimizer choices (Adam/AdamW/SGD)
  - Metrics logging (accuracy, precision, recall, F1, ROC AUC)
- 03-evaluation-and-interpretability.ipynb
  - Confusion matrices, ROC curves, per-class metrics
  - Grad-CAM / saliency maps to verify model focuses on plausible brain regions
- 04-export-and-inference.ipynb
  - Save best checkpoint, show sample inferences, how to load & run inference on new images

---

## **Data preprocessing & augmentation details**
Key steps for MRI image preprocessing:
- Resize images to the chosen EfficientNet input (e.g., 224x224).
- Convert grayscale MRI slices to 3-channel images (repeat channels) or use a single-channel input pipeline if supported.
- Intensity normalization (either zero-mean unit-variance per-image or per-dataset min-max scaling).
- Optionally perform histogram equalization or CLAHE for contrast enhancement.

Recommended augmentations (use albumentations for flexible pipelines):
- Random rotation (±10–20°)
- Horizontal/vertical flips (if anatomically acceptable)
- Random brightness & contrast
- Elastic transforms (careful; medical images must preserve anatomy)
- Gaussian noise or small blurring (to improve robustness)

Be cautious with aggressive spatial transforms that could produce anatomically impossible samples.

---

## **Modeling details: EfficientNet & transfer learning**
- Use EfficientNet pretrained on ImageNet as the feature extractor.
- Typical pipeline:
  1. Load EfficientNetB0/B3/B4 (select by resource availability).
  2. Remove top classification head and add:
     - GlobalAveragePooling
     - Dropout (e.g., 0.3–0.5)
     - Dense + activation (softmax for multiclass, sigmoid for binary)
  3. Freeze base layers initially; train classifier head for a few epochs.
  4. Unfreeze some top blocks and fine-tune with a lower learning rate (e.g., 1e-5 – 1e-4).
- Loss functions:
  - Binary crossentropy for binary classification
  - Categorical crossentropy for multiclass
  - Consider focal loss if class imbalance is severe
- Optimizers & LR scheduling:
  - Adam / AdamW for fast convergence
  - Cosine annealing or ReduceLROnPlateau for LR scheduling

Variant tips:
- EfficientNetB0 is lightweight and efficient for prototyping.
- EfficientNetB4+ may yield better accuracy but requires more GPU memory.
- If using PyTorch, timm library provides many EfficientNet variants.

---

## **Training, evaluation & metrics**
- Use stratified K-fold cross-validation (k=5) to obtain robust estimates.
- Metrics to track:
  - Accuracy, Precision, Recall, F1-score (per-class and macro/micro)
  - ROC AUC (binary or one-vs-rest for multiclass)
  - Confusion matrix for qualitative error analysis
- Save checkpoints on best validation metric (e.g., highest F1 or AUC).
- Log training with TensorBoard or Weights & Biases (wandb) for experiment tracking.

Example pseudocode for training (TensorFlow / Keras style):
```python
# build model
base = EfficientNetB3(include_top=False, input_shape=(224,224,3), weights='imagenet')
x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.4)(x)
out = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base.input, outputs=out)

# compile
model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
```

---

## **Explainability / Interpretability**
- Implement Grad-CAM or Grad-CAM++ to visualize regions driving model predictions.
- Produce side-by-side comparisons: original MRI, Grad-CAM overlay, predicted label & confidence.
- Use libraries such as tf-keras-vis (TensorFlow) or Captum (PyTorch) or create custom implementations.

Interpretability checklist:
- Verify that model activations correspond to plausible brain regions and not to image artifacts
- Use multiple examples per class (true positive, false positive, false negative) to analyze failure modes

---

## **Model export & Streamlit demo** 🚀🖼️

**Overview — what this section covers**  
This project exports the trained EfficientNet-based model into a portable format and provides a Streamlit web application that allows users to load custom MRI images, apply image-enhancement operations, and run model inference interactively. The Streamlit app is designed for research / demo purposes (not for clinical use).

**Saved model formats & typical locations**
- TensorFlow / Keras:
  - HDF5: `models/final/model_best.h5`
  - SavedModel dir: `models/final/saved_model/`
- PyTorch:
  - Torch checkpoint: `models/final/model_best.pt`
- ONNX export (optional, for cross-framework deployment): `models/final/model_best.onnx`
- The notebooks include example code to save/load these formats in `04-export-and-inference.ipynb`.

**How the Streamlit app works (high level)**
1. User uploads one or more images (PNG/JPG/NIfTI export as 2D slices).
2. App applies the same preprocessing pipeline used during training:
   - Resize to model input (e.g., 224x224)
   - Channel handling (grayscale->3-channel repeat if necessary)
   - Intensity normalization (match training normalization)
3. Optional image-enhancement step (see list below). Enhancements are applied client-side before inference so you can visually compare effects.
4. The app runs the model inference and displays:
   - Predicted label(s) and confidence score(s)
   - Grad-CAM overlay (if supported) to visualize attention regions
   - Option to download prediction results or save logs

**Image enhancement features included in the app**
- Resize & resample (keep aspect ratio or force target size)
- Histogram equalization / CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Contrast/Brightness adjustment slider
- Denoising (median blur or non-local means options via OpenCV)
- Sharpening filter (unsharp mask)
- Gamma correction
- Optional CLAHE + normalization combo recommended for MRI contrast tuning

These enhancements are exposed via UI sliders and toggle buttons so the user can experiment and see how preprocessing affects model confidence and explanation maps.

**Streamlit app UI flow (typical)**
- Sidebar:
  - Model select (e.g., EfficientNetB0/B3 and weights)
  - Enhancement toggles + sliders
  - Thresholds (confidence threshold for binary labeling)
  - Option to enable Grad-CAM overlay
- Main area:
  - Image upload / drag-and-drop
  - Original image preview
  - Enhanced image preview
  - Prediction card showing class probabilities + top prediction
  - Grad-CAM visualization panel
  - Download button (save enhanced image + prediction)

**Run the Streamlit app locally**
1. Install the Streamlit-specific requirements (if provided):
```bash
pip install -r requirements-streamlit.txt
# or
pip install streamlit opencv-python pillow numpy matplotlib tensorflow  # or torch + related deps
```
2. Run the app (example):
```bash
cd app
streamlit run streamlit_app.py
```
3. Open the URL printed in the console (typically http://localhost:8501).

**Docker example**
Create a small Dockerfile (example):
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements-streamlit.txt
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.enableCORS=false"]
```
Build & run:
```bash
docker build -t ad-detect-streamlit .
docker run -p 8501:8501 --gpus all? -v /path/to/models:/app/models ad-detect-streamlit
```
(Adjust GPU flags per your environment; `--gpus` requires Docker with GPU support.)

**Model loading & inference (implementation notes)**
- Ensure the app uses the *same* preprocessing (resize, normalization) as training. Discrepancies produce unreliable outputs.
- For TensorFlow:
```python
from tensorflow.keras.models import load_model
model = load_model('models/final/model_best.h5', compile=False)
preds = model.predict(preprocessed_image[np.newaxis, ...])
```
- For PyTorch:
```python
import torch
model = torch.load('models/final/model_best.pt', map_location='cpu')
model.eval()
with torch.no_grad():
    preds = model(preprocessed_tensor.unsqueeze(0))
```
- For ONNX: use onnxruntime InferenceSession to run inference.

**Grad-CAM integration**
- Compute Grad-CAM heatmap for the final convolutional layer and overlay onto the enhanced image.
- Optionally allow different colormaps and opacity sliders.

**Limitations & privacy**
- Uploaded images are processed locally in the Streamlit session (unless explicitly configured to upload to a server). If you deploy publicly, be explicit about whether images are stored or transmitted.
- The demo is for research/educational purposes; do not use predictions for clinical decisions.

**Performance considerations**
- Model inference on CPU is slower; use GPU-backed environment for faster feedback during large batches.
- For responsive UX, consider running Grad-CAM asynchronously or behind an "Explain" button.

**Logging & reproducibility**
- The app can optionally save a small JSON log entry per inference:
  - timestamp, model_version, enhancement_settings, prediction, confidence
- Store logs locally under `app/logs/` or allow export via UI.

---

## **Reproducibility & checkpoints**
- Set random seeds for numpy, tensorflow/torch, and Python's random module.
- Use deterministic data loaders where possible during evaluation.
- Save model weights and a JSON/yaml file with the experiment config (architecture, hyperparameters, data split seeds).
- If running long experiments, use checkpoints and store:
  - model_epoch_{n}.h5 / .pt
  - training_log.csv (metrics per epoch)
  - notes.md with observations

---

## **Hardware recommendations**
- Small/fast experiments: CPU or modest GPU (e.g., T4)
- Fine-tuning larger EfficientNet variants: at least 8–12 GB GPU RAM (e.g., Tesla T4, P100). For B5–B7 you may need 16+ GB.
- Use mixed precision training (FP16) if supported to speed up training and reduce memory usage.

---

## **Troubleshooting & tips**
- Out of memory (OOM): reduce batch size, use smaller EfficientNet (B0/B1), enable mixed precision.
- Slow training: check data pipeline (use tf.data or PyTorch DataLoader with prefetch), store images as TFRecords or bundled arrays for faster I/O.
- Poor generalization: try stronger regularization (dropout), more augmentation, or class rebalancing (weighted loss or resampling).
- Overfitting: monitor train vs validation curves; use early stopping on validation metric.

---

## **Extending & customizing experiments**
Ideas to extend:
- 3D approaches: train models on 3D MRI volumes (3D CNNs or 3D EfficientNet adaptations)
- Multi-modal inputs: combine imaging with clinical features (age, gender, cognitive scores)
- Ensemble multiple EfficientNet variants
- Hyperparameter search with Optuna or scikit-optimize
- Convert trained model to ONNX or TensorFlow SavedModel for deployment

---

## **Ethical considerations & clinical caution**
- This project is intended for research / educational use only.
- Models trained on publicly-available datasets are NOT clinical tools. They require rigorous validation, external testing, regulatory review, and domain expert oversight before any clinical use.
- Be mindful of dataset biases and ensure fair evaluation across demographic groups.

---

## **Licenses & acknowledgements**
- Data source used in experiments: OASIS dataset on Kaggle — https://www.kaggle.com/datasets/pulavendranselvaraj/oasis-dataset (please review dataset page for license and attribution).
- Acknowledge the original EfficientNet paper and libraries used (TensorFlow / PyTorch / timm / albumentations).
- If you plan to release models or derivative datasets, include an appropriate LICENSE file in the repo.

---

## **Contributing**
Contributions are welcome. Suggested workflow:
1. Fork the repo
2. Create a feature branch: `git checkout -b feature/your-change`
3. Add code / notebooks and ensure they run end-to-end
4. Commit, push and open a clear Pull Request describing your changes

Guidelines:
- Keep notebooks runnable and consider clearing large outputs before committing
- Add or update requirements.txt when new packages are used
- Prefer modular utility scripts in `utils/` for repeated code

---
