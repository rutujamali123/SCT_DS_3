import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Path to Kaggle dataset
# Inside "train" folder -> filenames like "cat.0.jpg", "dog.1.jpg"
DATASET_PATH = "datasets/cats_and_dogs/train"  

# Image parameters
IMG_SIZE = 64  # Resize all images to 64x64

def load_data(path, img_size=IMG_SIZE, limit=2000):
    X = []
    y = []
    count = 0
    for file in os.listdir(path):
        label = 0 if "cat" in file else 1  # 0=cat, 1=dog
        img_path = os.path.join(path, file)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (img_size, img_size))
            X.append(img.flatten())   # Flatten 3D -> 1D vector
            y.append(label)
            count += 1
            if count >= limit:  # Limit number for speed
                break
        except:
            continue
    return np.array(X), np.array(y)

print("Loading data...")
X, y = load_data(DATASET_PATH, limit=4000)  # Load 4000 images
print("Dataset shape:", X.shape, y.shape)

# Normalize pixel values (0-255 -> 0-1)
X = X / 255.0  

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train SVM
print("Training SVM...")
svm = SVC(kernel='linear', random_state=42)  # linear kernel
svm.fit(X_train, y_train)

# Predict
y_pred = svm.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
