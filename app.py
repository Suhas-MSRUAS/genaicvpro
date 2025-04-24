from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11m.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="D:/genaicvpro/dataset/data.yaml", epochs=100, imgsz=640)