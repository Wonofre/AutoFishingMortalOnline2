import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
import os
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler  # New import for feature scaling

# Specify the path to the reference audio file
reference_audio_file = "C:/Users/Bostil/Desktop/Python/Midias/So-o-Splash.mp3"

# Load the reference audio file
reference_audio, sample_rate = librosa.load(reference_audio_file, sr=None, mono=True)

SAMPLE_RATE = 44100  # Sample rate in Hz
DURATION = 1.3  # Duration of each audio frame in seconds
N_MFCC = 35  # Number of MFCC coefficients

# Specify the directory and file name for the training data
training_data_dir = "C:/Users/Bostil/Desktop/Python/training_dataMortal"
training_data_file = "training_dataMortal.npz"
training_data_path = os.path.join(training_data_dir, training_data_file)

# Check if the training data file exists
if not os.path.exists(training_data_path):
    raise FileNotFoundError(f"The training data file '{training_data_file}' does not exist.")

# Load the training data
training_data = np.load(training_data_path)
X = training_data["X"]
y = training_data["y"]

# Convert y to integer data type
y = y.astype(int)

# Calculate the class distribution
class_distribution = np.bincount(y) / len(y)

# Print the class distribution
print("Class distribution:", class_distribution)

# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Flatten the MFCC features for each segment
X_train_flattened = X_train.reshape(X_train.shape[0], -1)
X_val_flattened = X_val.reshape(X_val.shape[0], -1)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_flattened)
X_val_scaled = scaler.transform(X_val_flattened)

# Define the parameter grid for grid search
param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": [0.001, 0.01, 0.1, 1]
}

# Initialize the SVM model
svm_model = SVC(kernel='rbf')

# Specify a fraction for the subset (e.g., 10%)
subset_fraction = 0.1

# Calculate the number of samples in the subset
subset_size = int(subset_fraction * len(X_train_scaled))

# Randomly sample a subset of the training data
indices = np.random.choice(len(X_train_scaled), size=subset_size, replace=False)
X_train_subset = X_train_scaled[indices]
y_train_subset = y_train[indices]

# Perform grid search using the subset of the training data
grid_search = GridSearchCV(svm_model, param_grid, cv=5)
grid_search.fit(X_train_subset, y_train_subset)

# Get the best hyperparameters and the corresponding model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluate the best model on the validation set
accuracy = best_model.score(X_val_scaled, y_val)  # Use the scaled data
print("Validation accuracy:", accuracy)
print("Best hyperparameters:", best_params)

# Get predictions on validation set
y_pred = best_model.predict(X_val_scaled)  # Use the scaled data

# Calculate precision, recall and F1 score
precision, recall, fscore, _ = precision_recall_fscore_support(y_val, y_pred, average=None)

print("Precision:", precision) 
print("Recall:", recall)
print("F1 Score:", fscore)
