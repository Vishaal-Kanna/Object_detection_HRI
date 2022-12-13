# read input image
import cv2
import numpy as np

def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cap = cv2.VideoCapture('/home/vishaal/Vishaal/UMD_Sem_3/ENPM645/Final_Project/test.mp4')
outw = cv2.VideoWriter('object_detection.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1920,1080))

# read class names from text file
classes = None
net = cv2.dnn.readNet('/home/vishaal/Vishaal/UMD_Sem_3/ENPM808X/yolov3.weights',
                      '/home/vishaal/Vishaal/UMD_Sem_3/ENPM808X/ENPM808X_Midterm_Project/yolov3.cfg')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
with open('/home/vishaal/Vishaal/Projects/Event_vision/yolov3/coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]
while(cap.isOpened()):
  # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:

        image = frame
        # image = cv2.flip(image, 0)

        Width = image.shape[1]
        Height = image.shape[0]
        # print(Width, Height)
        scale = 0.00392


        # generate different colors for different classes
        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

        # read pre-trained model and config file

        # create input blob
        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

        # set input blob for the network
        net.setInput(blob)

        # run inference through the network
        # and gather predictions from output layers
        outs = net.forward(get_output_layers(net))

        # initialization
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        # for each detetion from each output layer
        # get the confidence, class id, bounding box params
        # and ignore weak detections (confidence < 0.5)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
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

        # go through the detections remaining
        # after nms and draw bounding box
        for i in indices:
            i = i
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]

            draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

        # display output image
        cv2.imshow("object detection", image)
        outw.write(image)

        # wait until any key is pressed
        cv2.waitKey(1)
    else:
        break

#
# # save output image to disk
# cv2.imwrite("object-detection.jpg", image)

# release resources
cv2.destroyAllWindows()