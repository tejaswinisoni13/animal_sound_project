import os
import json
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# =========================
# CONFIG
# =========================
UPLOAD_FOLDER = "uploads"
MODEL_PATH = "final_model.keras"
CLASS_NAMES_PATH = "class_names.json"

GRAPH_FOLDER = os.path.join("static", "generated")
WAVEFORM_FOLDER = os.path.join(GRAPH_FOLDER, "waveforms")
SPECTROGRAM_FOLDER = os.path.join(GRAPH_FOLDER, "spectrograms")

SR = 22050
MAX_AUDIO_LENGTH = 5
IMG_SIZE = (128, 128)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(WAVEFORM_FOLDER, exist_ok=True)
os.makedirs(SPECTROGRAM_FOLDER, exist_ok=True)

# =========================
# LOAD MODEL + CLASS NAMES
# =========================
model = load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

# =========================
# AUDIO PREPROCESSING
# =========================
def load_fixed_audio(file_path, sr=SR, max_length=MAX_AUDIO_LENGTH):
    audio, sample_rate = librosa.load(file_path, sr=sr)

    target_length = sr * max_length
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]

    return audio, sample_rate


def extract_mel_spectrogram(file_path, sr=SR, max_length=MAX_AUDIO_LENGTH, img_size=IMG_SIZE):
    try:
        audio, sample_rate = load_fixed_audio(file_path, sr=sr, max_length=max_length)

        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_mels=128
        )

        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        mel_spec_db = librosa.util.fix_length(
            mel_spec_db,
            size=img_size[1],
            axis=1
        )

        mel_spec_db = mel_spec_db[:img_size[0], :]

        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (
            mel_spec_db.max() - mel_spec_db.min() + 1e-6
        )

        return mel_spec_db

    except Exception as e:
        print(f"Error in extract_mel_spectrogram: {e}")
        return None


def prepare_input(file_path):
    feature = extract_mel_spectrogram(file_path)

    if feature is None:
        return None

    feature = np.expand_dims(feature, axis=-1)      # (128,128,1)
    feature = np.repeat(feature, 3, axis=-1)        # (128,128,3)
    feature = tf.image.resize(feature, (224, 224)).numpy()
    feature = preprocess_input(feature * 255.0)
    feature = np.expand_dims(feature, axis=0)       # (1,224,224,3)

    return feature

# =========================
# DOMINANT FREQUENCY
# =========================
def get_dominant_frequency(file_path, sr=SR, max_length=MAX_AUDIO_LENGTH):
    try:
        audio, sample_rate = load_fixed_audio(file_path, sr=sr, max_length=max_length)

        fft = np.fft.fft(audio)
        freqs = np.fft.fftfreq(len(fft), d=1 / sample_rate)
        magnitude = np.abs(fft)

        positive_freqs = freqs[:len(freqs) // 2]
        positive_magnitude = magnitude[:len(magnitude) // 2]

        dominant_freq = positive_freqs[np.argmax(positive_magnitude)]
        return round(float(dominant_freq), 2)

    except Exception as e:
        print(f"Frequency extraction error: {e}")
        return None

# =========================
# GRAPH GENERATION
# =========================
def sanitize_name(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def generate_waveform_plot(file_path, filename_base, sr=SR, max_length=MAX_AUDIO_LENGTH):
    try:
        audio, sample_rate = load_fixed_audio(file_path, sr=sr, max_length=max_length)

        output_filename = f"{filename_base}_waveform.png"
        output_path = os.path.join(WAVEFORM_FOLDER, output_filename)

        plt.figure(figsize=(10, 3))
        librosa.display.waveshow(audio, sr=sample_rate)
        plt.title("Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return f"generated/waveforms/{output_filename}"

    except Exception as e:
        print(f"Waveform generation error: {e}")
        return None


def generate_spectrogram_plot(file_path, filename_base, sr=SR, max_length=MAX_AUDIO_LENGTH):
    try:
        audio, sample_rate = load_fixed_audio(file_path, sr=sr, max_length=max_length)

        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        output_filename = f"{filename_base}_spectrogram.png"
        output_path = os.path.join(SPECTROGRAM_FOLDER, output_filename)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_spec_db, sr=sample_rate, x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Mel Spectrogram")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        return f"generated/spectrograms/{output_filename}"

    except Exception as e:
        print(f"Spectrogram generation error: {e}")
        return None

# =========================
# ROUTE TO PLAY UPLOADED AUDIO
# =========================
@app.route("/uploads/<filename>")
def uploaded_audio(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# =========================
# MAIN ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    error = None
    filename = None
    image_file = None
    dominant_frequency = None
    waveform_file = None
    spectrogram_file = None

    if request.method == "POST":
        if "audio_file" not in request.files:
            error = "No file uploaded."
            return render_template(
                "index.html",
                prediction=prediction,
                confidence=confidence,
                error=error,
                filename=filename,
                image_file=image_file,
                dominant_frequency=dominant_frequency,
                waveform_file=waveform_file,
                spectrogram_file=spectrogram_file
            )

        file = request.files["audio_file"]

        if file.filename == "":
            error = "No file selected."
            return render_template(
                "index.html",
                prediction=prediction,
                confidence=confidence,
                error=error,
                filename=filename,
                image_file=image_file,
                dominant_frequency=dominant_frequency,
                waveform_file=waveform_file,
                spectrogram_file=spectrogram_file
            )

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        input_data = prepare_input(file_path)

        if input_data is None:
            error = "Feature extraction failed."
            return render_template(
                "index.html",
                prediction=prediction,
                confidence=confidence,
                error=error,
                filename=filename,
                image_file=image_file,
                dominant_frequency=dominant_frequency,
                waveform_file=waveform_file,
                spectrogram_file=spectrogram_file
            )

        pred = model.predict(input_data, verbose=0)[0]
        pred_index = int(np.argmax(pred))

        prediction = class_names[pred_index]
        confidence = round(float(np.max(pred)) * 100, 2)
        filename = file.filename
        dominant_frequency = get_dominant_frequency(file_path)

        filename_base = sanitize_name(file.filename)
        waveform_file = generate_waveform_plot(file_path, filename_base)
        spectrogram_file = generate_spectrogram_plot(file_path, filename_base)

        image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
        for ext in image_extensions:
            test_path = os.path.join("static", "images", prediction + ext)
            if os.path.exists(test_path):
                image_file = prediction + ext
                break

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        error=error,
        filename=filename,
        image_file=image_file,
        dominant_frequency=dominant_frequency,
        waveform_file=waveform_file,
        spectrogram_file=spectrogram_file
    )

# =========================
# RUN APP
# =========================
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
