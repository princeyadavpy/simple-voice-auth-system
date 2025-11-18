# Simple Voice Authentication System

Python-based interactive voice recognition system using MFCC + pitch + cosine similarity.

## Features
- Enroll user with voice sample
- Authenticate user by comparing voice
- Stores templates in `templates/` as `.npy` files

## Requirements
pip install numpy sounddevice soundfile librosa scikit-learn

python simple_voice_auth_interactive.py
Then:

Press 1 → Enroll new user

Press 2 → Authenticate existing user

Notes
Audio recorded at 16 kHz for 3 seconds.

Templates stored as <username>_template.npy.

Author
Prince Yadav
