# simple_voice_auth_interactive.py
"""
Simple Interactive Voice Recognition
------------------------------------
This version will ask for:
1. Whether you want to enroll or authenticate
2. Your username
Then it records, extracts features, and compares voices.
"""

import os
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
from sklearn.metrics.pairwise import cosine_similarity

SR = 16000           # Sampling rate
DURATION = 3.0       # Recording time (in seconds)
TEMPLATE_DIR = "templates"
os.makedirs(TEMPLATE_DIR, exist_ok=True)

def record(filename, duration=DURATION, sr=SR):
    print(f"\n🎙️ Recording for {duration} seconds... Speak now!")
    rec = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    rec = rec.flatten()
    sf.write(filename, rec, sr)
    print("✅ Recording saved as:", filename)

def extract_features(wav_path, sr=SR, n_mfcc=13):
    y, _ = librosa.load(wav_path, sr=sr)
    # Extract MFCC mean values
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Estimate pitch (fundamental frequency)
    pitches, mags = librosa.piptrack(y=y, sr=sr)
    pitch_vals = [pitches[mags[:, i].argmax(), i] for i in range(pitches.shape[1]) if pitches[mags[:, i].argmax(), i] > 0]
    pitch_mean = float(np.mean(pitch_vals)) if len(pitch_vals) > 0 else 0.0

    return np.concatenate([mfcc_mean, [pitch_mean]])

def enroll_user(user_id):
    timestamp = int(time.time())
    wav = f"{TEMPLATE_DIR}/{user_id}_enroll_{timestamp}.wav"
    record(wav)
    features = extract_features(wav)
    np.save(f"{TEMPLATE_DIR}/{user_id}_template.npy", features)
    print(f"✅ User '{user_id}' enrolled successfully!")

def authenticate_user(user_id, threshold=0.85):
    wav = f"{TEMPLATE_DIR}/probe_{int(time.time())}.wav"
    record(wav)
    probe_features = extract_features(wav)

    template_path = f"{TEMPLATE_DIR}/{user_id}_template.npy"
    if not os.path.exists(template_path):
        print(f"⚠️ No stored voice for '{user_id}'. Please enroll first.")
        return

    template_features = np.load(template_path)
    similarity = cosine_similarity(template_features.reshape(1, -1), probe_features.reshape(1, -1))[0, 0]
    print(f"\n🧠 Similarity Score: {similarity:.3f}")

    if similarity >= threshold:
        print("✅ Authentication Result: ACCEPTED")
    else:
        print("❌ Authentication Result: REJECTED")

def main():
    print("🎧 SIMPLE VOICE RECOGNITION SYSTEM 🎧")
    print("1. Enroll new user")
    print("2. Authenticate existing user")

    choice = input("\nEnter your choice (1 or 2): ").strip()
    user_id = input("Enter your username: ").strip()

    if choice == '1':
        enroll_user(user_id)
    elif choice == '2':
        authenticate_user(user_id)
    else:
        print("⚠️ Invalid choice! Please restart and enter 1 or 2.")

if __name__ == "__main__":
    main()
