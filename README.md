# 🧠 Medical Image Diagnosis System using AI

An AI-powered medical image diagnosis system that leverages deep learning models to analyze medical images and assist in disease detection. The system supports multiple medical domains such as retina scans, chest X-rays, and dermatological images.

---

## 📌 Overview

Medical imaging plays a critical role in modern healthcare, enabling early detection and diagnosis of diseases. This project implements an end-to-end deep learning pipeline that:

- Accepts medical images as input
- Processes them using trained models
- Outputs predicted disease classes

---

## 🚀 Features

- Multi-disease prediction system:
  - Retina (Diabetic Retinopathy)
  - Chest X-ray (Multi-label classification)
  - Skin disease detection
- Deep learning-based inference
- Modular and scalable architecture
- Predefined label mappings
- Easy integration with web frameworks

---

## 🏗️ Project Structure

```
Medical_Image_Diagnosis_System_using_AI/
│
├── src/
│   ├── inference/
│   │   ├── predict.py
│   │
│   ├── models/
│
├── data/
│
├── requirements.txt
├── app.py
└── README.md
```

---

## 🧠 Model Details

| Module | Task | Classes |
|--------|------|--------|
| Retina | Diabetic Retinopathy | 5 |
| Chest  | X-ray Classification | 14 |
| Derma  | Skin Lesions         | 7 |

---

## 🏷️ Label Mapping Example

```python
RETINA_LABELS = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/VardhanBoi/Medical_Image_Diagnosis_System_using_AI.git
cd Medical_Image_Diagnosis_System_using_AI
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the application:

```bash
python app.py
```

Or (if using Streamlit):

```bash
streamlit run app.py
```

---

## 🔍 Workflow

1. Upload a medical image
2. Select the modality (Retina / Chest / Derma)
3. Model processes the image
4. Output: predicted disease class

---

## 🛠️ Tech Stack

- Python
- PyTorch / TensorFlow
- OpenCV
- NumPy / Pandas
- Flask / Streamlit

---

## ⚠️ Disclaimer

This project is for educational purposes only and should not be used for real medical diagnosis.

---

## 🤝 Contributing

Contributions are welcome. You can improve models, add datasets, or enhance UI.

---

## 👤 Author

**Vardhan Boi**  
GitHub: https://github.com/VardhanBoi

