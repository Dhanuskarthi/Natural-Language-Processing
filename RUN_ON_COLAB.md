# How to Run the Full Training Pipeline on Google Colab

This guide walks you through executing the full training, distillation, ONNX export, and INT8 quantization run on a free Google Colab GPU.

---

## ⏱️ Overview
- **Hardware**: Free Google Colab **T4 GPU**
- **Estimated Runtime**: ~8 to 12 minutes total
- **Expected Results**:
  - BERT-base Teacher Accuracy: **~92–93%** on the full 3,080 Banking77 test set
  - DistilBERT Baseline Accuracy: **~90–91%**
  - 4-layer Distilled Student Accuracy: **~88–90%** (outperforming non-distilled student)
  - Full ONNX FP32 & INT8 dynamic quantized models

---

## 📋 Step-by-Step Instructions

### 1. Open the Notebook in Google Colab
1. Navigate to [Google Colab](https://colab.research.google.com/).
2. Select the **GitHub** tab.
3. Paste your repository URL: `https://github.com/Dhanuskarthi/Natural-Language-Processing`
4. Select `notebooks/colab_training.ipynb`.
*(Alternatively: click "Upload", select the `notebooks/colab_training.ipynb` file from your computer, and open it.)*

### 2. Enable Free GPU
1. In the top navigation menu, click **Runtime** > **Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU**.
3. Click **Save**.

### 3. Run All Cells
1. Click **Runtime** > **Run all** (or press `Ctrl + F9`).
2. The notebook will automatically:
   - Check GPU status (`nvidia-smi`).
   - Clone the repository and install `requirements-train.txt`.
   - Train the Teacher (`bert-base-uncased`) and Baseline (`distilbert-base-uncased`) for 6 epochs on the full Banking77 dataset (lr=3.5e-5).
   - Run the Knowledge Distillation ablation study and save the top 4-layer student model.
   - Export all models to ONNX FP32 and quantize to INT8.
   - Evaluate all 9 variants on the full 3,080-sample test set.
   - Print the final accuracy & compression matrix table.
   - Bundle all trained models, ONNX files, and metric files into `compressroute_full_run.zip`.

### 4. Download and Extract Files Locally
1. The notebook triggers an automatic browser download of `compressroute_full_run.zip`.
2. Move `compressroute_full_run.zip` into your local project root:
   ```text
   c:\Users\dhanu\Downloads\NLP\Project NLP\
   ```
3. Extract `compressroute_full_run.zip` directly into the project root folder.
   - This populates:
     - `models/teacher_bert/`
     - `models/baseline_distilbert/`
     - `models/student_distilled/`
     - `models/onnx/` (with all 6 `.onnx` files)
     - `benchmark/results/` (with real accuracy metrics and ablation JSON/CSVs)

### 5. Next Step
Once you have extracted the files into your local project root, return here and say **"done"**. We will then proceed to **Step 5: Re-benchmark with real models on your CPU**.
