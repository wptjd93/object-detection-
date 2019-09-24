from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import cv2
import os
from imagetext import frame_textsave


def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


classes = None
with open("name.txt", 'r') as f:
    classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNet('my12000.weights', 'my.cfg')

video = input("비디오명 : ")

vs = VideoStream(video).start()

fps = FPS().start()
n = 0
i = 0
videodir = video.replace(".mp4", "")

if not (os.path.isdir(videodir)):
    os.mkdir(videodir)

while True:
    image = vs.read()
    copyimage = image
    image = imutils.resize(image, width=500)
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

    cv2.imshow("object detection", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if 0 in class_ids:
        objtime = int(vs.get() / 1000)
        mmin = str(int(objtime / 60))
        mmin = mmin.zfill(2)
        ssec = str(int(objtime % 60))
        ssec = ssec.zfill(2)
        cv2.imwrite(videodir + "\\%s_%s.jpg" % (mmin, ssec), copyimage)

    fps.update()

fps.stop()
cv2.destroyAllWindows()
vs.stop()
frame_textsave(videodir)


