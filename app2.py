from flask import Flask, render_template, Response, jsonify
import cv2
import supervision as sv
from ultralytics import YOLO
from supervision.draw.color import Color
from supervision import BoxAnnotator, LabelAnnotator
from supervision.detection.core import Detections

app = Flask(__name__)

PR_MODEL_PATH = "best.pt"   
VIDEO_PATH = "demo.mp4"     


detection_count = 0
detection_details = []


class PyResearchVisualizer:
    def __init__(self):
        self.model = YOLO(PR_MODEL_PATH)

        self.box_annotator = BoxAnnotator(
            thickness=2,
            color=Color.from_hex("#0055FF")
        )

        self.label_annotator = LabelAnnotator(
            text_scale=0.5,
            text_thickness=1,
            text_color=Color.WHITE,
            text_padding=4
        )

    def process_frame(self, frame):
        global detection_count, detection_details

        
        results = self.model(frame, verbose=False)[0]
        detections = Detections.from_ultralytics(results)

        detection_count = len(detections)
        detection_details = [] 


        bboxes = detections.xyxy  
        labels = []
        for (x_min, y_min, x_max, y_max), class_id, confidence in zip(
            bboxes, detections.class_id, detections.confidence
        ):
            width = int(x_max - x_min)
            height = int(y_max - y_min)

            
            detection_details.append({
                "class": results.names[class_id],
                "confidence": round(float(confidence), 2),
                "width": width,
                "height": height
            })

            
            label = f"{results.names[class_id]} {confidence:.2f} W:{width}px H:{height}px"
            labels.append(label)

    
        annotated_frame = self.box_annotator.annotate(
            scene=frame.copy(),
            detections=detections
        )

        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )

        return annotated_frame



def generate_frames():
    visualizer = PyResearchVisualizer()
    cap = cv2.VideoCapture(VIDEO_PATH)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
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
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detection_count')
def get_detection_count():
    return jsonify({
        'detections': detection_count,
        'boxes': detection_details
    })



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
