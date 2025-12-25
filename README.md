# Age Estimation with Deep EXpectation (DEX)

This project implements a deep learning–based **face age estimation** system using the **DEX (Deep EXpectation)** approach.  
Instead of treating age estimation as a direct regression problem, age is modeled as a **classification task** followed by an **expected value computation** over softmax probabilities.

The model is built on a **pretrained ResNet50 backbone** and fine-tuned for age prediction.

---

## Method Overview

Age estimation is formulated as a **classification problem** with N discrete age classes (e.g. 1–80).

Instead of predicting age directly via regression, the model outputs a **probability distribution** over age classes using a softmax layer.  
The final predicted age is computed as the **expected value** of this distribution.

Predicted Age = Σ (probability_i × age_i), for i = 1 to N

Where:
- probability_i is the softmax probability of age class i
- age_i is the corresponding age label

This approach is inspired by the **DEX (Deep EXpectation)** method proposed by Rothe et al.

---

## Model Architecture

- Backbone: **ResNet50** (ImageNet pretrained)
- Input size: 224 × 224 RGB
- Output: Softmax over age classes
- Loss function: Categorical Cross-Entropy
- Metric: Mean Absolute Error (MAE)
- Fine-tuning strategy:
  - Stage 1: Frozen backbone
  - Stage 2: Micro fine-tuning with low learning rate

---

## Dataset

- **UTKFace Dataset**
- Face images labeled with age information
- Images are cleaned and resized before training
- Age labels are converted into discrete class indices

> ⚠️ The dataset is **not included** in this repository due to size limitations.

---

## Training Strategy

1. Classification-based age prediction
2. Expected value computation from softmax output
3. Standard fine-tuning
4. Micro fine-tuning with a very low learning rate

This training strategy improves stability and reduces validation MAE.

---

## Results

| Model Stage | Validation MAE |
|------------|----------------|
| Baseline | ~9.9 |
| Fine-tuned | ~7.4 |
| Micro Fine-tuned | **~6.8** |

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- OpenCV

---

## References

- R. Rothe, R. Timofte, L. Van Gool  
  **DEX: Deep EXpectation of Apparent Age from a Single Image**  
  ICCV, 2015.

---

## Authors

- **Elif Yetim**
- **Abdullah Zengin**

Software Engineering Students
