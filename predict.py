import io

import requests
import supervision as sv
from PIL import Image

from rfdetr import (
    RFDETRBase,
    RFDETRLarge,
    RFDETRMedium,
    RFDETRNano,
    RFDETRSegPreview,
    RFDETRSmall,
)
from rfdetr.util.coco_classes import COCO_CLASSES

# This also downloads model, if it's not present in working directory of script.
model = RFDETRMedium()

model.optimize_for_inference()

# url = "https://media.roboflow.com/notebooks/examples/dog-2.jpeg"
# image = Image.open(io.BytesIO(requests.get(url).content))
image = Image.open("sample.jpg")
detections = model.predict(image, threshold=0.5)

labels = [
    f"{COCO_CLASSES[class_id]} {confidence:.2f}"
    for class_id, confidence in zip(detections.class_id, detections.confidence)
]

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

# save image to output.jpg
annotated_image.save("output.jpg")

# sv.plot_image(annotated_image)
