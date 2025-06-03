# Carpo: Automatic Bone Age Assessment System

This project delivers a full pipeline for bone age estimation from hand X-rays. It combines computer vision, deep learning, and medical software tools to create a deployable, user-friendly solution.

## 🔍 Main Features

- **Automatic ROI Detection**: Using a YOLO-based model to detect and extract key regions of interest.
- **Bone Age Prediction**: Deep learning regression model implemented in TensorFlow/Keras.
- **Graphical Interface**: CustomTkinter GUI for image loading, prediction, and DICOM export.
- **DICOM Export**: Results are saved with proper metadata in DICOM format, enabling medical integration.

## 📁 Project Structure

- `frontend.py`: Graphical interface and central control logic.
- `AI_classes/ROI_detection.py`: Region detection and preprocessing via YOLO.
- `AI_classes/BAA_prediction.py`: Bone age prediction and image utilities.
- `Model_weights/`: Contains YOLO and Keras trained models.

## 🧠 Skills Demonstrated

- Digital processing of medical images (alignment, cropping, normalization).
- Neural networks applied to biomedical data.
- Integration of TensorFlow and YOLO into a GUI-based pipeline.
- Export and manipulation of DICOM files with Python.
- Good practices in modular software development.
