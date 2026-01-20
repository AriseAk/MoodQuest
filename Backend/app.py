import os
import cv2
import torch
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from torchvision import transforms
from dotenv import load_dotenv
from timm import create_model
from groq import Groq
from collections import deque
import time

# Helper imports (Ensure these files are in your Hugging Face Space too!)
from helpers import get_random_joke, get_one_fact
from trivia import fetch_questionnaire, score_assessment, interpret_score

# Load environment variables
load_dotenv()

app = Flask(__name__)

# --- CORS CONFIGURATION ---
# IMPORTANT: We allow "*" (all origins) initially to prevent Vercel connection errors.
# For higher security later, you can change "*" to ["https://your-frontend.vercel.app"]
CORS(app, resources={r"/api/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"]}})

# --- MODEL CONFIGURATION ---
MODEL_PATH = 'best_fer2013_model_70.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 7
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

print(f"⚙️  Initializing on device: {DEVICE}")

# --- GLOBAL STATE ---
# NOTE: In a professional production app with multiple users, you would use a database
# or session IDs instead of global variables. For a portfolio demo, this is acceptable.
emotion_history = deque(maxlen=30)
current_emotion = "Neutral"
emotion_confidence = 0.0
stress_level = "Low"
MODEL_LOADED = False

# --- LOAD MODEL ---
try:
    print("Loading fine-tuned model...")
    model = create_model('efficientnet_b0', pretrained=False, num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    MODEL_LOADED = True
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    MODEL_LOADED = False

# --- LOAD GROQ ---
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    print("⚠️ Warning: GROQ_API_KEY not found in environment")
    groq_client = None
else:
    print("✨ Connecting to Groq...")
    groq_client = Groq(api_key=GROQ_API_KEY)
    print("✅ Groq Ready!")

# --- PREPROCESSING ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load Face Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- HELPER FUNCTIONS ---

def calculate_stress_level(history):
    """Calculates stress based on the history of detected emotions."""
    if len(history) < 5:
        return "Low"
    
    # Count negative emotions
    negative_emotions = sum(1 for e in history if e in ['Angry', 'Fear', 'Sad'])
    negative_ratio = negative_emotions / len(history)
    
    if negative_ratio > 0.6:
        return "High"
    elif negative_ratio > 0.3:
        return "Medium"
    else:
        return "Low"

def detect_emotion_from_frame(face_img):
    """Runs inference on a single cropped face."""
    if not MODEL_LOADED:
        return "Neutral", 0.0
    try:
        face_tensor = transform(face_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = model(face_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
        
        emotion = EMOTION_LABELS[predicted_idx.item()]
        return emotion, confidence.item()
    except Exception as e:
        print(f"Inference Error: {e}")
        return "Neutral", 0.0

# --- API ENDPOINTS ---

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'online', 
        'model_loaded': MODEL_LOADED, 
        'device': DEVICE,
        'hosting': 'Hugging Face Spaces (Docker)'
    })

@app.route('/api/analyze-frame', methods=['POST'])
def analyze_frame():
    """
    Receives an image file from the frontend, detects faces,
    predicts emotion, and updates global stress state.
    """
    global current_emotion, emotion_confidence, stress_level

    if not MODEL_LOADED:
        return jsonify({'error': 'Model not loaded'}), 503

    # Check if image part exists
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    
    try:
        # Read image file into numpy array (OpenCV format)
        file_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        # Convert to Gray for Face Detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

        detected_something = False

        if len(faces) > 0:
            detected_something = True
            # Find the largest face (closest to camera)
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            face_roi = frame[y:y+h, x:x+w]

            # Predict
            emotion, confidence = detect_emotion_from_frame(face_roi)
            
            # Update State
            current_emotion = emotion
            emotion_confidence = confidence
            emotion_history.append(emotion)
            stress_level = calculate_stress_level(emotion_history)
        else:
            # If no face detected, we keep the previous state or just return 'No Face'
            # But we do NOT update history with 'Neutral' to avoid skewing stress calculation
            pass

        return jsonify({
            'emotion': current_emotion,
            'confidence': round(emotion_confidence * 100, 1),
            'stress_level': stress_level,
            'face_detected': detected_something
        })

    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analytics', methods=['GET'])
def analytics():
    """Returns the current session statistics."""
    emotion_counts = {}
    for emotion in emotion_history:
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    return jsonify({
        'current_emotion': current_emotion,
        'confidence': round(emotion_confidence * 100, 2),
        'stress_level': stress_level,
        'emotion_distribution': emotion_counts,
        'total_frames_analyzed': len(emotion_history),
        'history_length': len(emotion_history)
    })

@app.route('/api/emotion/reset', methods=['POST'])
def reset_emotion_history():
    global emotion_history, current_emotion, emotion_confidence, stress_level
    emotion_history.clear()
    current_emotion = "Neutral"
    emotion_confidence = 0.0
    stress_level = "Low"
    return jsonify({'message': 'Emotion history reset successfully'})

# --- CHAT / JOKE / FACT ENDPOINTS ---

@app.route('/api/joke', methods=['GET'])
def get_joke():
    joke = get_random_joke()
    return jsonify({'joke': joke})

@app.route('/api/fact', methods=['GET'])
def get_fact():
    # Note: Ensure API_NINJAS_KEY is set in HF Spaces Secrets
    fact = get_one_fact(os.getenv('API_NINJAS_KEY'))
    return jsonify({'fact': fact})

@app.route('/api/questionnaire/<name>', methods=['GET'])
def get_questionnaire(name):
    questions = fetch_questionnaire(name)
    if not questions:
        return jsonify({'error': 'Questionnaire not found'}), 404
    return jsonify({'name': name, 'questions': questions})

@app.route('/api/questionnaire/submit', methods=['POST'])
def submit_questionnaire():
    data = request.json
    name = data.get('name')
    answers = data.get('answers')
    if not name or not answers:
        return jsonify({'error': 'Missing name or answers'}), 400
    score = score_assessment(name, answers)
    interpretation = interpret_score(name, score)
    return jsonify({'name': name, 'score': score, 'interpretation': interpretation})

@app.route('/api/grok', methods=['POST'])
def grok_reply():
    data = request.json
    question = data.get('question')
    
    if not question:
        return jsonify({'error': 'Question is required'}), 400

    if not groq_client:
        return jsonify({'error': 'Groq API not configured.'}), 503

    # Stress Intervention Logic
    if stress_level in ["High", "Medium"]:
        try:
            # We occasionally inject a joke if stressed
            # (Simple random check could be added here to not do it every time)
            pass 
        except Exception as e:
            print(f"Error: {e}")

    system_instruction = f"""
    You are a helpful, witty, and empathetic emotional support assistant.
    Current User Status:
    - Emotion: {current_emotion}
    - Stress Level: {stress_level}
    
    FORMATTING RULES:
    1. Keep responses SHORT (max 2-3 sentences).
    2. Use bullet points for steps.
    3. Speak naturally, like a friend.
    4. AVOID overly formal language.
    """

    try:
        completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": question}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
        )
        
        response_text = completion.choices[0].message.content
        cleaned_reply = response_text.replace("*", "").replace("#", "").replace("Groq", "I")
        
        return jsonify({"reply": cleaned_reply})

    except Exception as e:
        print(f"Groq API Error: {e}")
        return jsonify({"reply": "I'm having a bit of trouble connecting right now."})

if __name__ == '__main__':
    # Hugging Face Spaces Docker usually exposes port 7860
    # The CMD in Dockerfile should run this file
    port = int(os.environ.get("PORT", 7860))
    print(f"🚀 Starting Server on port {port}...")
    app.run(host='0.0.0.0', port=port)