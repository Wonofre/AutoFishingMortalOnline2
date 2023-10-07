import librosa
import numpy as np
import soundcard as sc
import sounddevice as sd
import time
from sklearn import svm
from collections import deque
import pyautogui

# Specify the path to the reference audio file
reference_audio_file = "C:/Users/Bostil/Desktop/Python/Midias/So-o-Splash.mp3"

SAMPLE_RATE = 44100  # Sample rate in Hz
DURATION = 1.2  # Duration of each audio frame in seconds
N_MFCC = 13  # Number of MFCC coefficients

# Load the reference audio file
reference_audio, sample_rate = librosa.load(reference_audio_file, sr=SAMPLE_RATE, mono=True)

# Load the trained SVM model
training_data = np.load("training_dataMortal/training_dataMortal.npz")
X_train = training_data["X"]
y_train = training_data["y"]

# Initialize the SVM model
svm_model = svm.SVC(kernel='rbf', C=1, gamma="scale")

# Fit the SVM model to the training dataset
svm_model.fit(X_train, y_train)

def process_audio(buffer, sample_rate):
    window_size = int(DURATION * sample_rate)  # Size of the sliding window
    overlap = 0.15  # Overlap in seconds
    step_size = int((DURATION - overlap) * sample_rate)  # Step size for sliding the window
    
    # Initialize the circular buffer
    circular_buffer = np.zeros((window_size,))
    circular_buffer_idx = 0
    
    # Append the new audio frames to the circular buffer
    for i in range(len(buffer)):
        circular_buffer[circular_buffer_idx] = buffer[i]
        circular_buffer_idx = (circular_buffer_idx + 1) % window_size
        
        # Check if the circular buffer is filled
        if circular_buffer_idx == 0:
            # Compute audio features (MFCC) for the captured audio segment
            segment_features = extract_audio_features(circular_buffer, sample_rate)

            # Classify the captured audio segment using the SVM model
            label = svm_model.predict(segment_features.flatten().reshape(1, -1))

            # Perform an action based on the classification result
            if label == 1:
                print("Captured audio segment matches the reference audio.")
                # Perform your action here
                # Hold the left mouse button for 50 seconds
                pyautogui.mouseDown()
                time.sleep(12)
                pyautogui.mouseUp()
                # Click the left mouse button for 1.2 seconds
                pyautogui.mouseDown()
                time.sleep(0.2)
                pyautogui.mouseUp()
            else:
                print("Captured audio segment does not match the reference audio.")
                # Perform a different action or take no action

            # Shift the circular buffer by the step size
            circular_buffer[:step_size] = circular_buffer[window_size-step_size:]
            circular_buffer[step_size:] = 0

    return circular_buffer, label

def extract_audio_features(audio, sample_rate):
    # Compute MFCC features using librosa
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)

    return mfcc

print("Listening for audio...")

duration_samples = int(DURATION * SAMPLE_RATE)

# Initialize a deque for storing captured audio data
captured_audio = deque(maxlen=int(DURATION * sample_rate))

# Start capturing audio with loopback from default speaker
with sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True).recorder(samplerate=SAMPLE_RATE) as mic:
    while True:
        # Record audio
        #start_time = time.time()
        data = mic.record(numframes=duration_samples)
        #recording_time = time.time() - start_time

        # Append captured audio frames to the deque
        captured_audio.extend(data[:, 0])

        # Check if captured audio is long enough to process
        if len(captured_audio) >= int(DURATION * sample_rate):
            # Process the captured audio
            #start_time = time.time()
            process_audio(np.array(captured_audio), sample_rate)  # Convert the deque to a numpy array for processing
            #processing_time = time.time() - start_time

            # Print the timestamp and processing time for the audio frame
            #print("Timestamp:", time.time())
            #print("Recording time:", recording_time)
            #print("Processing time:", processing_time)
