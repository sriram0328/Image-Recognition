from flask import Flask, render_template, request, redirect, url_for, jsonify
import cv2
import torch
import numpy as np
import os
from ultralytics import YOLO
import io
from PIL import Image
import base64

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = YOLO("yolov8l.pt").to(device) #use this for better accuracy and faster results
# model = YOLO("yolov8n.pt").to(device)  # Smaller model (works on Render Free Plan)
model = YOLO("ultralytics/yolov8l.pt")  # Automatically downloads model from Ultralytics



def object_detection(image, conf_threshold=0.3):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    max_size = 640
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    results = model(image)
    class_names = model.names

    for box, class_id, confidence in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
        if confidence < conf_threshold:
            continue

        x1, y1, x2, y2 = map(int, box.cpu().numpy())
        label = f"{class_names[int(class_id)]}: {confidence:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def encode_image(image):
    ret, frame = cv2.imencode('.jpg', image)
    return base64.b64encode(frame).decode('utf-8')

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                try:
                    img = Image.open(io.BytesIO(file.read()))
                    img = np.array(img)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    processed_img = object_detection(img)

                    return jsonify({
                        "uploaded_image": encode_image(img),
                        "processed_image": encode_image(processed_img)
                    })
                except Exception as e:
                    return jsonify({"error": str(e)})

        elif 'webcam_data' in request.form:
            try:
                data = request.form['webcam_data']
                img_data = base64.b64decode(data.split(',')[1])
                img = Image.open(io.BytesIO(img_data))
                img = np.array(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                processed_img = object_detection(img)

                return jsonify({
                    "uploaded_image": encode_image(img),
                    "processed_image": encode_image(processed_img)
                })
            except Exception as e:
                return jsonify({"error": str(e)})

        elif 'clear' in request.form:
            return jsonify({"uploaded_image": None, "processed_image": None})  #  Reset everything in frontend

    return render_template("index.html")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Render assigns a port dynamically
    app.run(host="0.0.0.0", port=port, debug=False)
