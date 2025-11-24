import sys
import cv2
import numpy as np
import onnxruntime as ort

def preprocess_image(image, input_size):
    original_size = image.shape[:2]  # (height, width)
    input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(input_img, (input_size, input_size))
    input_img = input_img / 255.0
    input_img = np.transpose(input_img, (2, 0, 1))  # HWC -> CHW
    input_img = np.expand_dims(input_img, axis=0)  # Add batch dimension
    input_img = input_img.astype(np.float32)
    #input_img = input_img[np.newaxis, :, :, :].astype(np.float32)
    return input_img, original_size

def rescale_boxes(boxes, input_size, original_height, original_width):
    input_shape = np.array([input_size, input_size, input_size, input_size])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([original_width, original_height, original_width, original_height])
    return boxes

def postprocess_yolov10_output(output, conf_threshold, original_size):
    output = output.squeeze()
    boxes = output[:, :4]
    confidences = output[:, 4]
    class_ids = output[:, 5].astype(int)

    # filter out low conf
    mask = confidences > conf_threshold
    boxes = boxes[mask, :]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    # Rescale boxes to original image dimensions
    boxes = rescale_boxes(boxes, 640, original_size[0], original_size[1])
    return class_ids, boxes, confidences

def postprocess_yolov8_output(output, conf_threshold, original_size):
    output = output.squeeze()
    boxes = output[:4, :]
    class_probs = output[4:, :]
    class_ids = np.argmax(class_probs, axis=0)
    class_scores = class_probs[class_ids, range(class_probs.shape[1])]

    # filter out low class score
    scores_mask = class_scores > conf_threshold
    boxes = boxes[:, scores_mask]
    class_scores = class_scores[scores_mask]
    class_ids = class_ids[scores_mask]

    nms_mask = cv2.dnn.NMSBoxes(boxes.transpose(), class_scores, 0.2, 0.5)

    boxes = boxes[:, nms_mask]
    class_scores = class_scores[nms_mask]
    class_ids = class_ids[nms_mask]

    # Rescale boxes to original image dimensions
    boxes = rescale_boxes(boxes.transpose(), 640, original_size[0], original_size[1])
    return class_ids, boxes, class_scores

# Usage: python3 test_onnx_inference.py <model> <filename>
if __name__ == '__main__':

    # Preprocessing
    image = cv2.imread(sys.argv[2])
    input_tensor, original_size = preprocess_image(image, input_size=640)

    if sys.argv[1] == 'yolov10n':
        onnx_model_path = '../models/yolov10n.onnx'
    elif sys.argv[1] == 'yolov8n':
        onnx_model_path = '../models/yolov8n.onnx'
    else:
        print('Unknown model')

    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})
    
    # Postprocessing
    if sys.argv[1] == 'yolov10n':
        class_ids, boxes, confidences = postprocess_yolov10_output(outputs[0], 0.2, original_size)
    elif sys.argv[1] == 'yolov8n':
        class_ids, boxes, confidences = postprocess_yolov8_output(outputs[0], 0.2, original_size)

    # Draw the boxes on the image
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(int)
        label = f"Class {class_ids[i]}: {confidences[i]:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    # Save the image with bounding boxes
    output_image_path = 'output.jpg'
    cv2.imwrite(output_image_path, image)
    print(f"Image saved as {output_image_path}")