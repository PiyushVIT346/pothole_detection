from flask import Flask, render_template, Response, request, redirect, url_for, session, flash, jsonify
import sqlite3
import cv2
from ultralytics import YOLO
from supervision import BoxAnnotator, LabelAnnotator, Detections
from supervision.draw.color import Color
from datetime import datetime
import json

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set your own secret key

# Initialize database
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

PR_MODEL_PATH = "best.pt"
VIDEO_PATH = "demo.mp4"
detection_count = 0


class PyResearchVisualizer:
    def __init__(self):
        self.model = YOLO(PR_MODEL_PATH)
        self.box_annotator = BoxAnnotator(thickness=2, color=Color.from_hex("#0055FF"))
        self.label_annotator = LabelAnnotator(
            text_scale=0.5,
            text_thickness=1,
            text_color=Color.WHITE,
            text_padding=4
        )

    def process_frame(self, frame):
        global detection_count

        results = self.model(frame, verbose=False)[0]
        detections = Detections.from_ultralytics(results)
        detection_count = len(detections)
        print("Pothole Cout :",detection_count)

        labels = [
            f"{results.names[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        annotated_frame = self.box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        return annotated_frame


def generate_frames():
    """Generate video frames with pothole detection"""
    visualizer = PyResearchVisualizer()
    
    # Try to open video file first, then fallback to webcam
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Warning: Could not open {VIDEO_PATH}, trying webcam...")
        cap = cv2.VideoCapture(0)  # Try webcam
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else float('inf')
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            # Loop video if it's a file
            if total_frames != float('inf'):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break

        annotated_frame = visualizer.process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', annotated_frame)

        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )

    cap.release()


@app.route('/')
def index():
    """Main dashboard route - redirect to login if not authenticated"""
    if 'username' not in session:
        return redirect(url_for('welcome'))
    return render_template('index.html')


@app.route('/welcome')
def welcome():
    """Welcome page with login/register options"""
    return render_template('welcome.html')


@app.route('/first')
def first_page():
    """Legacy route redirect"""
    return redirect(url_for('welcome'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
            conn.commit()
            flash('Registration successful. Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists.', 'error')
        finally:
            conn.close()

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username=? AND password=?', (username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            session['username'] = username
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials.', 'error')

    return render_template('login.html')


@app.route('/index')
def dashboard():
    """Legacy route redirect"""
    return redirect(url_for('index'))


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Upload video for analysis"""
    if 'username' not in session:
        return redirect(url_for('first_page'))
    
    if request.method == 'POST':
        # Handle file upload - placeholder implementation
        flash('Upload feature coming soon! File upload functionality will be implemented.', 'info')
        return redirect(url_for('upload'))
    
    return render_template('upload.html')


@app.route('/map')
def pothole_map():
    """Full-screen pothole map view"""
    if 'username' not in session:
        return redirect(url_for('first_page'))
    return render_template('map.html')


@app.route('/map_data')
def map_data():
    """API endpoint for map marker data"""
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    # Return sample data - in production this would come from database
    sample_data = [
        {
            'id': '1',
            'lat': 40.7589,
            'lon': -73.9851,
            'label': 'Pothole detected on 5th Ave',
            'severity': 'high',
            'ts': '2024-01-15T10:30:00'
        },
        {
            'id': '2', 
            'lat': 40.7505,
            'lon': -73.9934,
            'label': 'Surface damage on Broadway',
            'severity': 'medium',
            'ts': '2024-01-15T11:15:00'
        },
        {
            'id': '3',
            'lat': 40.7614,
            'lon': -73.9776,
            'label': 'Minor crack detected',
            'severity': 'low',
            'ts': '2024-01-15T09:45:00'
        },
        {
            'id': '4',
            'lat': 40.7282,
            'lon': -74.0776,
            'label': 'Large pothole - immediate attention needed',
            'severity': 'high',
            'ts': '2024-01-15T12:00:00'
        },
        {
            'id': '5',
            'lat': 40.7412,
            'lon': -74.0055,
            'label': 'Road surface deterioration',
            'severity': 'medium',
            'ts': '2024-01-15T08:30:00'
        }
    ]
    
    return jsonify(sample_data)


@app.route('/video_feed')
def video_feed():
    if 'username' not in session:
        return redirect(url_for('login'))
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detection_count')
def get_detection_count():
    """API endpoint for current detection count"""
    if 'username' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    return jsonify({'detections': detection_count})


@app.route('/logout')
def logout():
    """Logout and redirect to welcome page"""
    session.pop('username', None)
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('welcome'))


if __name__ == "__main__":
    app.run(debug=True,use_reloader=False ,host='0.0.0.0')
