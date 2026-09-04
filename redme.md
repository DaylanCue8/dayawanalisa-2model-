Dayaw: Baybayin Recognition & Translation System
Dayaw is an intelligent mobile application designed to bridge the gap between modern Tagalog and the ancient script of Baybayin. Using a Support Vector Machine (SVM) with a Radial Basis Function (RBF) kernel, the system recognizes handwritten Baybayin characters and provides real-time linguistic translation.

🌟 Key Features
Mode 1: Baybayin to Tagalog (Image Recognition)

Uses HOG (Histogram of Oriented Gradients) for feature extraction.

Classifies characters using an RBF SVM Model.

Provides an Evaluation Modal showing confidence scores and stroke feedback.

Mode 2: Tagalog to Baybayin (Linguistic Engine)

Regex-based tokenization of Tagalog syllables.

Supports modern Baybayin rules (Kudlit and Virama/Pamudpod).

Data Logging

MySQL integration to track processing sessions and character accuracy for research analysis.

🛠️ Technical Stack
Frontend: Flutter (Dart)

Backend: Flask (Python)

Machine Learning: Scikit-learn, OpenCV, Skimage

Database: MySQL

Algorithm: RBF Support Vector Machine

🚀 Installation & Setup
1. Prerequisites
Flutter SDK

Python 3.10+

XAMPP / MySQL Server

2. Backend Setup (Flask)
Navigate to the backend folder.

Install dependencies:

Bash
pip install flask flask-cors opencv-python numpy joblib scikit-learn scikit-image mysql-connector-python
Ensure your model files (baybayin_svm_model.sav and class_names.sav) are in the root of the backend folder.

Run the server:

Bash
python app.py
3. Database Setup
Open phpMyAdmin.

Create a database named dayaw.

Import the following schema:

SQL
CREATE TABLE processing_sessions (
    session_id INT AUTO_INCREMENT PRIMARY KEY,
    status VARCHAR(50),
    ip_address VARCHAR(50),
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
4. Frontend Setup (Flutter)
Navigate to the dayaw_app folder.

Update the baseUrl in lib/services/api_service.dart to your machine's IPv4 address.

Install packages:

Bash
flutter pub get
Run the app:

Bash
flutter run