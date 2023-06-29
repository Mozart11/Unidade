import airsim 
import cv2
import numpy as np

i = 0
ims = 0
start = 0
velocidade = 2
posição_max_direita = 33
posição_max_esquerda = 3
posição_baixo_1 = -9
posição_baixo_2 = -18


client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)             #Inicia as APIs do Airsim e liga o drone
client.takeoffAsync().join()
client.moveToZAsync(-8,3).join()

def movimento_Drone():   #Movimento predefinido do drone
    global start
    if ims == 1 and start == 0 :
        client.moveToPositionAsync(posicao_drone_x, posição_max_direita, posicao_drone_z, velocidade)
        start = 1
    if posicao_drone_y >= 33 and start == 1 : 
        client.moveToPositionAsync(posição_baixo_1, posicao_drone_y, posicao_drone_z, velocidade)
        start = 2

    if posicao_drone_x <= -9 and start == 2:
        client.moveToPositionAsync(posicao_drone_x, 3 , posicao_drone_z, velocidade)
        start = 3

    if posicao_drone_y <= 3 and start == 3 :
        client.moveToPositionAsync(posição_baixo_2, posicao_drone_y, posicao_drone_z, velocidade)
        start = 4

    if posicao_drone_x <= -18 and start == 4 :
        client.moveToPositionAsync(posicao_drone_x, posição_max_direita, posicao_drone_z, velocidade)
        start = 5
        
    if posicao_drone_y >= 33 and start == 5 :
        client.moveToPositionAsync(0, 0, posicao_drone_z, velocidade).join()
        client.goHomeAsync().join()
        client.landAsync().join()
        client.armDisarm(False)    

def transform_input(responses):  # Converte as imagens para RGB e transforam em um array
    img1d = np.frombuffer(responses.image_data_uint8, dtype=np.uint8)
    img_rgba = img1d.reshape(responses.height, responses.width, 3)

    from PIL import Image

    image = Image.fromarray(img_rgba)

    im_final = np.array(image.convert('RGB'))

    return im_final

# Carrega o YoLo e o nosso modelo
net = cv2.dnn.readNetFromDarknet("yolov4-custom.cfg", "yolov.weights")
classes = []
with open("obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

while(True):
    # Carrega a imagem na API do Airsim
    img = client.simGetImages([airsim.ImageRequest("3", airsim.ImageType.Scene, False, False)])[0]
    img = transform_input(img)
    img = cv2.resize(img, None, fx=3, fy=3)
    height, width, channels = img.shape

    # Detecta os objetos
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Mostra as informações na tela
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Detecção do objeto
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Coordenadas do retângulo
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Desenha o retângulo e coloca a legenda
            cv2.putText(img, label, (x, y + 30), font, 1, (0, 255, 0), 3)           
            ims = 1

    posicao_drone_y = client.getMultirotorState().kinematics_estimated.position.y_val
    posicao_drone_x = client.getMultirotorState().kinematics_estimated.position.x_val # Define as posições atuais do drone
    posicao_drone_z = client.getMultirotorState().kinematics_estimated.position.z_val # no sistema de coordenadas do Airsim
    movimento_Drone()

    cv2.imshow('img',img)  # Carrega visão do modelo para reconhecimento da plantação
    key = cv2.waitKey(1)

    
                   
         
      
    


    