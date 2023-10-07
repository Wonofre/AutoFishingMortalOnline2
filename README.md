# AutoFishingMortalOnline2
Fishing automation in Mortal Online 2. Capturing, training and processing real time audio with Ml.

Overview
Welcome to one of my first projects! Initially, I created this tool for my personal use in Mortal Online 2, but feel free to try it out for yourself. This project aims to assist with fishing activities in the game. Utilizing machine learning techniques and real-time audio processing, the program listens for a specific sound event that occurs when a fish "bites the bait." When this event is detected, the script triggers a keypress to "reel in" the fish. Please note that this project does not fully automate the fishing process; rather, it serves as an aid for more efficient fishing.

Note: For optimal performance, the game's sound must be set to a high volume.

Installation Steps
Install Required Python Packages
Be sure of the paths.

Game Settings:
Make sure the game's sound is set to a high volume for optimal performance.

Components:
"Programa de treino.py"
This script captures sound samples from your computer and plays them back to you. You classify each sound as a "Yes" or "No" to indicate whether the specific "fish bite" sound was present or not. If the provided training data is not satisfactory, you can use this script to create your own training model.

"ProgramaRealTimeMacro.py"
This script runs the real-time fishing macro. It listens for the specific "fish bite" sound and triggers a keypress to reel in the fish.

"Teste de qualidade dos dados de treino.py"
This script evaluates the quality of the training data and the machine learning model. Use this script to get an idea of how to fine-tune the model parameters.

Contributing
If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

