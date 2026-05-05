# 🌿 Plant Disease Detection Using Deep Learning

## 📌 Overview

Plant diseases cause significant crop losses worldwide. This project uses **Deep Learning (CNN)** to automatically detect plant diseases from leaf images.

The system allows users to upload an image and get **instant predictions** of possible diseases with confidence scores.

---

## 🚀 Features

* 🌱 Detects **39 plant disease classes**
* 📊 Achieves **96.4% accuracy**
* ⚡ Fast prediction using trained CNN model
* 🌐 Flask API for real-time inference
* 📱 Easy-to-use interface

---

## 🧠 Technologies Used

* Python
* TensorFlow / Keras
* Flask
* HTML, CSS
* NumPy, OpenCV

---

## 📂 Project Structure

```
Plant-Disease-Detection/
│── app.py                  # Flask backend
│── index.html              # Frontend UI
│── plant_disease_model.keras  # Trained model
│── class_names.json        # Class labels
│── requirements.txt       # Dependencies
│── runtime.txt
│── start.bat
```

---

## ⚙️ How It Works

1. User uploads a plant leaf image
2. Image is preprocessed (resize, normalize)
3. CNN model analyzes the image
4. System returns **Top-3 disease predictions**

---

## 📊 Dataset

* **PlantVillage Dataset**
* 54,000+ images
* 39 classes
* 14 crop species

---

## 🧪 Model Details

* Custom CNN Architecture (~3.8M parameters)
* Optimizer: Adam
* Loss: Categorical Crossentropy
* Accuracy: **96.4%**

---

## ▶️ How to Run the Project

### 1. Clone Repository

```
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Run Flask App

```
python app.py
```

### 4. Open Browser

```
http://127.0.0.1:5000/
```

---

## 📸 Output

* Displays predicted disease
* Shows confidence score
* Returns top 3 predictions

---

## 🔮 Future Scope

* Real-world image dataset integration
* Mobile app deployment
* GPS-based disease tracking
* More crop disease classes

---

## 👨‍💻 Authors

* Pushkar Singh
* Team Members

---

## 📄 License

This project is for educational purposes.

---

## 🙏 Acknowledgement

* PlantVillage Dataset
* IILM University
* Guide: Dr. Sahil Kansal

---
