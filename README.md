# AI-Based-Pneumonia-Detection
## 📌 Project Overview

Pneumonia is a serious lung infection that can be life-threatening if not detected early. This project presents a **deep learning–based automated system** to detect pneumonia from **chest X-ray images** using a **Convolutional Neural Network (CNN)**.

The system leverages **transfer learning with MobileNetV2**, performs preprocessing and fine-tuning, and provides predictions through an **interactive user interface built with Gradio**.

---

## 🎯 Objectives

* Automatically classify chest X-ray images as **NORMAL** or **PNEUMONIA**
* Reduce dependency on manual diagnosis
* Improve early detection using AI
* Build an easy-to-use prediction interface

---

## 🧠 Technologies Used

* **Python**
* **TensorFlow / Keras**
* **MobileNetV2 (Transfer Learning)**
* **NumPy, Matplotlib, Seaborn**
* **Scikit-learn**
* **Gradio (UI)**
* **Google Colab**

---

## 📂 Dataset

* **Source:** Kaggle – Chest X-ray Pneumonia Dataset
* **Classes:**

  * NORMAL
  * PNEUMONIA
* **Total Images:** ~5,800+
* **Data Split:**

  * Training
  * Validation
  * Testing

The dataset is **imbalanced**, with more pneumonia images than normal images.

---

## 🔧 Data Preprocessing

* Image resizing to **224 × 224**
* Feature scaling using **rescale = 1/255**
* Validation split (10%)
* **Data augmentation**:

  * Rotation
  * Zoom
  * Horizontal flip
  * Width & height shift
* **Class weights** applied to handle data imbalance

---

## 🧩 Model Architecture

* **Base Model:** MobileNetV2 (pre-trained on ImageNet)
* **Why MobileNetV2?**

  * Lightweight
  * Faster training
  * High accuracy with fewer parameters
  * Suitable for medical image classification

### Custom Layers Added:

* Global Average Pooling
* Dropout (to prevent overfitting)
* Dense layer with **Sigmoid activation**

---

## 🔁 Training Strategy

1. **Transfer Learning**

   * Base MobileNetV2 layers frozen
   * Train only the custom classification layers

2. **Fine-Tuning**

   * Unfreeze last few layers of MobileNetV2
   * Train with a very low learning rate
   * Helps the model learn pneumonia-specific patterns

3. **Loss Function**

   * Binary Cross-Entropy

4. **Optimizer**

   * Adam Optimizer

---

## 📊 Model Evaluation

* Training Accuracy: ~95%
* Validation Accuracy: ~92%
* Test Accuracy: ~85–90%
* Evaluation Metrics:

  * Accuracy
  * Confusion Matrix
  * Precision
  * Recall
  * F1-score

---

## 🖥️ User Interface (UI)

* Built using **Gradio**
* Allows users to:

  * Upload chest X-ray images
  * Get real-time predictions
  * View confidence scores
* Color-coded output:

  * 🟢 Green → NORMAL
  * 🔴 Red → PNEUMONIA

---

## ▶️ How to Run the Project

1. Open the notebook in **Google Colab**
2. Mount Google Drive
3. Load the saved `.keras` model
4. Run the Gradio UI cell
5. Upload a chest X-ray image to get prediction

---

## 📌 Conclusion

This project demonstrates how deep learning and transfer learning can be effectively used for medical image classification. The system provides a fast, reliable, and user-friendly way to assist in pneumonia detection, which can support healthcare professionals in diagnosis.

---


## 📜 License

This project is for **educational purposes only** and not intended for clinical use.

