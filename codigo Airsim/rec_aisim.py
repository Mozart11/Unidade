import airsim #pip install airsim
import cv2
import numpy as np
import time
# for car use CarClient()
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

def transform_input(responses):
    img1d = np.frombuffer(responses.image_data_uint8, dtype=np.uint8)

    #img1d = 255 / np.maximum(np.ones(img1d.size), img1d)

    img_rgba = img1d.reshape(responses.height, responses.width, 3)

    from PIL import Image

    image = Image.fromarray(img_rgba)

    im_final = np.array(image.convert('RGB'))

    return im_final

def index_of_biggest_object(object_list):
    WIdthPlusHeight = []
    for i in range(len(object_list)):
        x, y, w, h = object_list[i]
        WIdthPlusHeight.append(w+h)
    max_value = max(WIdthPlusHeight)
    return  WIdthPlusHeight.index(max_value)

def position_of_object(biggest_object):
    x, y, w, h = biggest_object
    x_box = (x+w)/2
    if x_box < 200: # 받아오는 이미지의 크기를 몰라서 임의로 설정한 값
        return "left"
    else:
        return "right"

# Load Yolo
net = cv2.dnn.readNetFromDarknet("yolov4-custom.cfg", "yolov4.weights")
classes = []
with open("obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

while(True):
    # Loading image
    img = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
    img = transform_input(img)
    img = cv2.resize(img, None, fx=3, fy=3)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN

    ObjectWidthPlusHeight = []
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 1, color, 3)

    #png_image = cv2.resize(png_image, dsize=None, fx=3, fy=3)
    cv2.imshow('img',img)
    key = cv2.waitKey(1)

    