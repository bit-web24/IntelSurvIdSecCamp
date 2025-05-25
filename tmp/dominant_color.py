from ultralytics import YOLO
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

# Class groups
SHIRT_CLASSES = {
    "blazer", "cardigan", "coat", "hoodies", "jacket", 
    "longsleeve", "mtm", "padding", "shirt", "shortsleeve", 
    "sweater", "zipup"
}
PANT_CLASSES = {
    "cottonpants", "denimpants", "shortpants", "skirt", "slacks", "trainingpants"
}

def get_dominant_color(image):
    data = np.array(image)
    data = data.reshape((-1, 3))  # Flatten pixels to rows
    kmeans = KMeans(n_clusters=3, random_state=0).fit(data)
    counts = np.bincount(kmeans.labels_)
    dominant = kmeans.cluster_centers_[np.argmax(counts)]
    dominant = tuple(map(int, dominant))
    return '#{:02x}{:02x}{:02x}'.format(*dominant)


def detect_objects_on_image(model, image):
    results = model.predict(image)
    result = results[0]
    output = []

    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = int(box.cls[0].item())
        prob = round(box.conf[0].item(), 2)
        label = result.names[class_id]

        if label in SHIRT_CLASSES:
            category = "shirt"
        elif label in PANT_CLASSES:
            category = "pant"
        else:
            category = label

        department = None
        color = None

        if category == "shirt":
            cropped = image.crop((x1, y1, x2, y2))
            color = get_dominant_color(cropped)
            department = get_department_from_color(color)

        output.append([x1, y1, x2, y2, category, prob, color, department])

    return output



from PIL import ImageColor

def hex_to_rgb(hex_color):
    return ImageColor.getrgb(hex_color)

def color_distance(c1, c2):
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5

def get_department_from_color(color_hex, threshold=100):
    if not color_hex:
        return None

    color_rgb = hex_to_rgb(color_hex)

    color_map = {
        "BCA": hex_to_rgb("#ffadbb"),      # Light pink
        "B.Tech": hex_to_rgb("#d7c18b"),   # Wheat (your image)
        "B.Pharma": hex_to_rgb("#62b2d1"), # Sky blue
        "BMLT": hex_to_rgb("#c0588c"),     # Dark pink
        "MBA": hex_to_rgb("#ffffff"),      # Yellow
    }

    closest_dept = "Unknown"
    min_distance = float("inf")

    for dept, ref_rgb in color_map.items():
        dist = color_distance(color_rgb, ref_rgb)
        if dist < min_distance:
            min_distance = dist
            closest_dept = dept

    return closest_dept if min_distance < threshold else "Unknown"
