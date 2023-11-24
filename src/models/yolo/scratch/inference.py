# NOTE: This file is not used in the final pipeline. It is used for testing purposes only. (And is not intended for use here, just for reference.)

from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import (
    Annotator,
)  # ultralytics.yolo.utils.plotting is deprecated


def main():
    # Load a pretrained YOLOv8n model
    model = YOLO("models/best.pt")

    # Define source as YouTube video URL
    source = "spot_youtube/00/frame1990.png"
    img = cv2.imread(source, 0)

    # Run inference on the source
    results = model.predict(source, save=True, stream=True, conf=0.5)

    for r in results:
        annotator = Annotator(img)
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            top, left, bottom, right = b
            c = box.cls
            annotator.box_label(b, model.names[int(c)])
    img = annotator.result()

    x_min, y_min = left, top
    width = right - left
    height = bottom - top
    cropped_image = img[y_min : y_min + height, x_min : x_min + width]

    # Save the detected and cropped images
    cv2.imwrite("cropped_image.png", cropped_image)
    cv2.imwrite("detected_image.png", img)

    return


if __name__ == "__main__":
    main()
