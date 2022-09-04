# ONNX YOLOv7 Object Detection
- This repo has been forked from https://github.com/ibaiGorordo/ONNX-YOLOv7-Object-Detection

# Installation
```
git clone https://github.com/ucekmez/yolov7-onnxruntime.git
cd yolov7-onnxruntime
pip install -r requirements.txt
```

ps: replace `onnxruntime` to `onnxruntime-gpu` in requirements.txt, if you want to use GPU instead of CPU

# Download ONNX models
```
bash download_models.sh
```
- The License of the models is GPL-3.0 license: [License](https://github.com/WongKinYiu/yolov7/blob/main/LICENSE.md)

# Sample Usage
```
from yolov7 import yolov7
yolov7_detector = yolov7("models/yolov7-tiny_640x640.onnx", conf_thres=0.5, iou_thres=0.5)
boxes, scores, class_ids = yolov7_detector("sample.jpg")
drawings = yolov7_detector.draw_detections(yolov7_detector.current_image)
```

# Additional
yolov7_detector accepts its input with the following ways:
>     local image path            : yolov7_detector("/path/to/image.jpg")
>     base64 encoded image string : yolov7_detector("/9j/4AAQSkZJRgABAQAAAQABAAD/4gIcSUNDX1BST....9h+sPs+0H1h+vs+/sf6e0TH9DH/AKj7f//Z")
>     image url                   : yolov7_detector("http://address/image.jpg")

# To see drawings
```
from PIL import Image
Image.fromarray(drawings)
```

# Original YOLOv7 model
The original YOLOv7 model can be found in this repository: [YOLOv7 Repository](https://github.com/WongKinYiu/yolov7)