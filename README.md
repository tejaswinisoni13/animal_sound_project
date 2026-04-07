🔊 Animal Sound Classification using Machine Learning
📌 Project Overview

This project focuses on audio classification using Machine Learning techniques.
The system takes an audio input file and predicts the animal sound.

🚀 Features
Classifies animal sounds:
Cow
Dog
Frog
Pig
Rooster
Uses MFCC (Mel Frequency Cepstral Coefficients)
Displays prediction with confidence
Generates sound frequency visualization
🛠️ Technologies Used
Python
Librosa
TensorFlow / Keras
NumPy, Pandas
Flask (for UI)
📂 Project Structure
audio_project/
│
├── model/
│   └── audio_model.keras
│
├── static/
│   ├── css/
│   ├── images/
│
├── templates/
│   └── index.html
│
├── app.py
├── requirements.txt
└── README.md
⚙️ How It Works
Upload audio file (.wav)
Extract MFCC features using Librosa
Convert into feature vector
Pass into trained model
Predict animal class
▶️ How to Run
git clone <repo-link>
cd audio_project
pip install -r requirements.txt
python app.py

Open: http://127.0.0.1:5000/

📊 Dataset
ESC-10 / Animal Sound Dataset
📈 Results
Accuracy: ~85–90%
📌 Future Improvements
Real-time sound detection
Improve accuracy
Add more animal classes
Deploy online
👩‍💻 Team Members
Tejaswini Soni
Suhani Khare
Shrishty Alanse
📢 Acknowledgement

We thank our mentors and SGSITS, Indore for guidance and support.
