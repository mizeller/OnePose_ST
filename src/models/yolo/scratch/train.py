# NOTE: This file is not used in the final pipeline. It is used for testing purposes only. (And is not intended for use here, just for reference.)

from ultralytics import YOLO


def main():
    # Load a model
    model = YOLO(
        "models/yolov8n.pt"
    )  # load a pretrained model (recommended for training)

    results = model.train(
        data="datasets/spot.v2i.yolov8/data.yaml", epochs=25, imgsz=640, device="mps"
    )

    # NOTE - this actually worked...but the training on COLAB was faster... so I'm simply going to use that one.


if __name__ == "__main__":
    main()
