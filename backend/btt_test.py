import os
import cv2
import joblib
import numpy as np
from skimage.feature import hog
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Configuration - CHANGE THIS TO YOUR ACTUAL TEST FOLDER PATH
TEST_DATA_PATH = "path/to/your/test_images_folder" 
MODEL_PATH = "baybayin_svm_model.sav"

def extract_hog_features(image_path):
    # Matches the preprocessing logic from your White Box testing description
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64)) # Ensure this matches your training size
    features = hog(img, orientations=9, pixels_per_cell=(8, 8), 
                   cells_per_block=(2, 2), visualize=False)
    return features

print("Script started...")

# 2. Load Model
model = joblib.load(MODEL_PATH)

X_test = []
y_test = []

# 3. Process Images from Folders (Assuming subfolders are named by character: 'ba', 'ka', etc.)
print("Extracting HOG features from test images...")
for label in os.listdir(TEST_DATA_PATH):
    label_path = os.path.join(TEST_DATA_PATH, label)
    if os.path.isdir(label_path):
        for img_file in os.listdir(label_path):
            if img_file.endswith(('.png', '.jpg', '.jpeg')):
                features = extract_hog_features(os.path.join(label_path, img_file))
                X_test.append(features)
                y_test.append(label)

if not X_test:
    print("Error: No images found in the test path!")
else:
    # 4. Predict
    y_pred = model.predict(X_test)

    # 5. Effectiveness Results
    print(f"\nOverall BTT Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    # 6. Confusion Matrix (The visual proof for Objective 2)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=sorted(set(y_test)), yticklabels=sorted(set(y_test)))
    plt.title('BTT SVM Confusion Matrix (Effectiveness Evaluation)')
    plt.ylabel('Actual Character')
    plt.xlabel('Predicted Character')
    plt.show()

print("Script finished!")