from ultralytics import YOLO
from flask import request, Response, Flask
from waitress import serve
from PIL import Image
import json
from collections import Counter
import numpy as np
from dominant_color import detect_objects_on_image

app = Flask(__name__)

model = YOLO("best-cloth.pt")

@app.route("/")
def root():
    """
    Site main page handler function.
    :return: Content of index.html file
    """
    with open("index.html") as file:
        return file.read()


@app.route("/detect", methods=["POST"])
def detect():
    """
        Handler of /detect POST endpoint
        Receives uploaded file with a name "image_file", 
        passes it through YOLOv8 object detection 
        network and returns an array of bounding boxes.
        :return: a JSON array of objects bounding 
        boxes in format 
        [[x1,y1,x2,y2,object_type,probability],..]
    """
    buf = request.files["image_file"]
    boxes = detect_objects_on_image(model, Image.open(buf.stream))
    return Response(
      json.dumps(boxes),  
      mimetype='application/json'
    )

serve(app, host='0.0.0.0', port=8080)

