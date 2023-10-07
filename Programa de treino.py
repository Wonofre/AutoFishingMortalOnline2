import soundcard as sc
import sounddevice as sd
import numpy as np
import librosa
from sklearn.utils import shuffle
import os
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Specify the path to the reference audio file
reference_audio_file = "C:/Users/Bostil/Desktop/Python/Midias/So-o-Splash.mp3"
SAMPLE_RATE = 44100  # Sample rate in Hz
DURATION = 1.2  # Duration of each audio frame in seconds
N_MFCC = 13  # Number of MFCC coefficients

# Load the reference audio file
reference_audio, sample_rate = librosa.load(reference_audio_file, sr=SAMPLE_RATE, mono=True)

svm_model = SVC(kernel='rbf', C=1, gamma="scale")
                
def process_audio(buffer, sample_rate):
    window_size = int(DURATION * sample_rate)  # Size of the sliding window
    overlap = 0.1  # Overlap in seconds
    step_size = int((DURATION - overlap) * sample_rate)  # Step size for sliding the window

    X = []
    y = []

    for i in range(0, len(audio) - window_size + 1, step_size):
        # Extract a segment of the captured audio
        segment = audio[i:i + window_size]

        # Compute audio features (MFCC) for the captured audio segment
        segment_features = extract_audio_features(segment, sample_rate).flatten()

        # Get user input for labeling the segment as similar or not similar
        label = get_user_label(segment)

        # Append the segment features and label to the training dataset
        X.append(segment_features)
        y.append(label)

    return np.array(X), np.array(y)

def extract_audio_features(audio, sample_rate):
    # Compute MFCC features using librosa
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
    return mfcc

def get_user_label(segment):
    # Play the segment for the user
    sd.play(segment, samplerate=SAMPLE_RATE)
    sd.wait()

    # Get user input for labeling (1 for similar, 0 for not similar)
    label = int(input("Is the captured audio segment similar? (1 for similar, 0 for not similar): "))
    return label

def save_training_data(X, y):
    if not os.path.exists("training_dataMortal"):
        os.makedirs("training_dataMortal")

    training_data_file = "training_dataMortal/training_dataMortal.npz"
    if os.path.exists(training_data_file):
        existing_data = np.load(training_data_file)
        existing_X = existing_data["X"]
        existing_y = existing_data["y"]
        X = np.concatenate([existing_X, X], axis=0)
        y = np.concatenate([existing_y, y], axis=0)

    np.savez(training_data_file, X=X, y=y)
    print("Training data saved.")

# Start capturing audio with loopback from default speaker
with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True).recorder(samplerate=SAMPLE_RATE) as mic:
    print("Recording audio...")

    X = []
    y = []

    for _ in range(40):  # Repeat 4 times
        # Record audio for the specified duration
        data = mic.record(numframes=int(DURATION * SAMPLE_RATE))
        audio = data[:, 0]

        # Process the captured audio segment and get the training data
        segment_X, segment_y = process_audio(audio, SAMPLE_RATE)

        # Append the segment data to the training dataset
        X.append(segment_X)
        y.append(segment_y)

        # Save the training data after each segment
        save_training_data(segment_X, segment_y)

# Concatenate the existing and new training data
X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0)

# Save the updated training data
save_training_data(X, y)

# Flatten the MFCC features for each segment
X = X.reshape(X.shape[0], -1)

# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42)


# Fit the SVM model to the training dataset
svm_model.fit(X_train, y_train)

# Evaluate the model on the validation set
accuracy = svm_model.score(X_val, y_val)

print("Validation accuracy:", accuracy)