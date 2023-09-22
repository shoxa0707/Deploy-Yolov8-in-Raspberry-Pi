import cv2
import numpy as np
import tensorflow as tf
import time

# Load the labels
with open('coco.names') as f:
    labels = f.read().split("\n")

def letterbox(im, new_shape = (640, 640), color = (0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[
        1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im,
                            top,
                            bottom,
                            left,
                            right,
                            cv2.BORDER_CONSTANT,
                            value=color)  # add border
    return im, r, (dw, dh)

def blob(im, return_seg = False):
    if return_seg:
        seg = im.astype(np.float32) / 255
    im = im.transpose([2, 0, 1])
    im = im[np.newaxis, ...]
    im = np.ascontiguousarray(im).astype(np.float32) / 255
    if return_seg:
        return im, seg
    else:
        return im


def nms(boxes, scores, iou_threshold):
    # Convert to xyxy
    boxes = xywh2xyxy(boxes)
    
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest 
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over threshold
        keep_indices = np.where(ious < iou_threshold)[0] + 1

        sorted_indices = sorted_indices[keep_indices]

    return keep_boxes

def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou

def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path='yolov8n.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# connecting setup of ip camera
username = 'admin'
password = 'Hikvision07!'
ip = '192.169.0.100'
rtsp_port = '554'

# connect to ip camera
cap = cv2.VideoCapture(f"rtsp://{username}:{password}@{ip}:{rtsp_port}/h264/ch1/main/av_stream")

# cv2.namedWindow("video", cv2.WINDOW_NORMAL)
# cv2.resizeWindow('video', 640, 640)

if cap.isOpened() == False:
    print("Error opening video stream or file")

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        start = time.time()
        input_shape = (640, 640)  # Replace with the expected input shape of your YOLO model
        bgr, ratio, dwdh = letterbox(frame, input_shape)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb)
        tensor = np.ascontiguousarray(tensor)
        # Run the model
        interpreter.set_tensor(input_details[0]['index'], tensor)
        interpreter.invoke()

        predictions = interpreter.get_tensor(output_details[0]['index'])
        predictions = np.array(predictions).reshape((84, 8400))

        predictions = predictions.T


        ########################
        conf_thresold = 0.7
        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > conf_thresold, :]
        scores = scores[scores > conf_thresold]
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = predictions[:, :4]
        iou_thres = 0.3
        indices = nms(boxes, scores, iou_thres)
        for (bbox, score, label) in zip(xywh2xyxy(boxes[indices]), scores[indices], class_ids[indices]):
            bbox[0] -= dwdh[0]
            bbox[1] -= dwdh[1]
            bbox[2] -= dwdh[0]
            bbox[3] -= dwdh[1]
            bbox /= ratio
            bbox = bbox.round().astype(np.int32).tolist()
            
            cls_id = int(label)
            cls = labels[cls_id]
            cv2.rectangle(frame, tuple(bbox[:2]), tuple(bbox[2:]), color=(0, 255, 0), thickness=2)
            cv2.putText(frame,
                        f'{cls}:{int(score*100)}', (bbox[0], bbox[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.60, [225, 0, 0],
                        thickness=2)
        cv2.putText(frame,
                        f'FPS: {round(1/(time.time() - start), 2)}', (200, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, [0, 0, 255],
                        thickness=3)
        cv2.imshow("video", frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Destroys all the windows created
cv2.destroyAllWindows()
