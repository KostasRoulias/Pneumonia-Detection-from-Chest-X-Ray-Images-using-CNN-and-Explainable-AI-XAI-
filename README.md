# 🩻 Pneumonia Detection from Chest X-Ray Images using CNN and Explainable AI (XAI)

A deep learning project for **automatic pneumonia detection** from chest X-ray images, combining **Convolutional Neural Networks (CNNs)** with **Explainable AI (XAI)** methods to enhance transparency and interpretability in medical image diagnosis.

---

## 🌐 Overview

The goal of this project is to develop a **computer vision model** that can accurately classify chest X-rays as **Normal** or **Pneumonia**.  
In addition to achieving high accuracy, the project integrates **Explainable AI (XAI)** techniques to visualize and understand *how* the model makes its predictions — a crucial aspect for **medical AI trustworthiness**.

This project was developed as part of a university assignment on **Deep Learning and Explainable AI**.

---

## 📊 Dataset

**Source:** [Kaggle – Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

- Total images: ~5,863  
- Categories: **Normal** and **Pneumonia**  

---

## 🧠 Model Architecture

A **Convolutional Neural Network (CNN)** was implemented using **TensorFlow / Keras**.  
The architecture includes:

- Convolutional layers with ReLU activation  
- MaxPooling for spatial dimension reduction  
- Dropout layers to prevent overfitting  
- Dense layers for classification  
- Sigmoid output layer (binary classification)

Training strategy:
- Loss: Binary Cross-Entropy  
- Optimizer: Adam  
- Epochs: 20–30  
- Batch size: 32  
- Validation split: 20%

---

## ⚙️ Methodology

1. **Data Preprocessing**
 - Image resizing and normalization (scaling pixel values 0–1)
 - Data augmentation (rotation, zoom, horizontal flip)
 - Train/validation/test split

2. **Model Training**
 - CNN built and trained using Keras Sequential API  
 - Early stopping to prevent overfitting  
 - Accuracy and loss curves tracked per epoch  

3. **Evaluation**
 - Metrics: Accuracy, Precision, Recall, F1-score  
 - Confusion matrix visualization  
 - Comparison between training and validation performance

---

## 🧩 Explainable AI (XAI) — Model Interpretability

To improve model transparency, **Grad-CAM (Gradient-weighted Class Activation Mapping)** was applied to visualize the regions in each chest X-ray that influenced the CNN’s predictions.

**Key benefits:**
- Highlights lung regions contributing most to the classification  
- Verifies that the model focuses on medically relevant areas  
- Helps detect potential dataset bias or overfitting

## 📈 Results

- **Best model accuracy:** ~93% (on validation data)  
- **Loss:** steadily decreased with regularization  
- **Grad-CAM visualizations:** correctly highlighted pneumonia-affected lung regions  

**Confusion Matrix:**
         Predicted
         Normal   Pneumonia
Actual Normal 94% 6%
Actual Pneumonia 8% 92%


---

## 🧰 Technologies Used

| Category | Tools / Libraries |
|-----------|------------------|
| **Language** | Python |
| **Deep Learning** | TensorFlow, Keras |
| **Data Processing** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Explainability** | Grad-CAM (Keras + OpenCV) |
| **Environment** | Jupyter Notebook |

---

## 📂 Files

| File | Description |
|------|--------------|
| `100675765_Assignment2.ipynb` | Jupyter Notebook with full pipeline (preprocessing, CNN training, Grad-CAM visualization) |
| `README.md` | Project documentation |

---

## 👨‍💻 Author

**Konstantinos Roulias**  
📧 [LinkedIn](https://www.linkedin.com/in/konstantinosroulias/)  
💻 [GitHub](https://github.com/KostasRoulias)

---

## 📜 License

This project is licensed under the **MIT License**.  
You are free to use and modify it with proper credit.

---

⭐ **If you found this project useful, give it a star on GitHub!**
