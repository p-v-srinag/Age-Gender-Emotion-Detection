# Deep-Learning Based Age, Gender & Emotion Detection for Retail Customer Profiling

This repository contains a complete deep learning-based facial analysis pipeline capable of predicting an individual's age group, gender, and emotional state from static facial images. The system utilizes specialized Convolutional Neural Network (CNN) architectures and is designed as a modular multi-model pipeline to provide real-time, multi-attribute predictions.

## 🚀 Key Features
* **Age Classification:** Predicts age across seven distinct demographic ranges: `1-2`, `3-9`, `10-20`, `21-27`, `28-45`, `46-65`, and `66-116` years.
* **Gender Identification:** Performs binary classification to identify individuals as male or female.
* **Emotion Recognition:** Maps facial expressions into three core categories: positive, negative, and neutral.
* **Face Detection:** Utilizes OpenCV's Haar Cascade classifier to automatically isolate and crop facial regions from raw input images.
* **Visual Output:** Integrates predictions by overlaying bounding boxes and prediction labels directly onto the detected faces.

## 🛠️ Technology Stack
* **Programming Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras
* **Computer Vision Library:** OpenCV (utilized for face detection and image preprocessing)
* **Model Storage:** H5 file format for pre-trained CNN models

## 📊 Datasets Used
The independent models were trained on specific datasets optimized for their respective tasks:
* **Gender Prediction:** UTKFace dataset
* **Age Prediction:** Custom augmented facial dataset
* **Emotion Classification:** CK+48 dataset

## 📂 Project Structure
The repository is organized into specific Jupyter Notebooks for training and inference:
* `2.1_train_age_model.ipynb`: Data preprocessing, model definition, and training pipeline for the Age classification CNN.
* `2.2_train_gender_model.ipynb`: Data preprocessing, model definition, and training pipeline for the Gender identification CNN.
* `2.3_train_emotion_model.ipynb`: Data preprocessing, model definition, and training pipeline for the Emotion recognition CNN.
* `3.1_Pred_Final.ipynb`: The integrated inference script. It loads the saved `.h5` models, utilizes Haar Cascades for face detection, and outputs the final annotated images with multi-attribute predictions.

## ⚙️ System Workflow
1. **Input Image Module:** The system accepts raw facial images of varying resolutions, lighting, and postures.
2. **Face Detection:** Haar Cascade classifiers scan the image to locate and extract facial regions.
3. **Image Preprocessing:** Extracted faces undergo grayscale conversion, normalization, and resizing to specific dimensions (e.g., 200x200, 100x100, 48x48) required by each model.
4. **CNN Inference:** Preprocessed facial tensors are fed into the three distinct CNN prediction modules simultaneously.
5. **Output Integration:** Predictions from all three networks are synthesized and visually displayed as bounding boxes and labels on the original image.

## 🔮 Future Scope
* **Multi-Task Architecture:** Developing a combined multi-output CNN to capture age, gender, and emotion simultaneously to reduce computational costs.
* **Video-Based Analysis:** Incorporating temporal models (LSTMs or 3D-CNNs) to evaluate dynamic emotional states in live video feeds.
* **Application Deployment:** Transitioning the pipeline into a real-time web or mobile application utilizing frameworks like Flask, FastAPI, or TensorFlow Lite.

## 👥 Team
Developed by undergraduate students at the Department of Computer Science & Engineering, SRM University-AP:
* **B Praneeth** * **P V Sri Nag** * **V Mohan Balu** * **M Sri Hari** **Project Guide:** Dr. Ch. Mallikarjuna, Asst. Professor, CSE Department.
