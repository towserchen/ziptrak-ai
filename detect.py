import os.path as osp
import cv2
import numpy as np
import supervision as sv
import onnxruntime as ort
from utils.preview import fast_process

try:
    import torch
    from torchvision.ops import nms
except Exception as e:
    print(e)

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
MASK_ANNOTATOR = sv.MaskAnnotator()


class LabelAnnotator(sv.LabelAnnotator):
    @staticmethod
    def resolve_text_background_xyxy(
        center_coordinates,
        text_wh,
        position,
    ):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h


LABEL_ANNOTATOR = LabelAnnotator(text_padding=4,
                                 text_scale=0.5,
                                 text_thickness=1)

window_model = ort.InferenceSession('weights/window.onnx', providers=['CPUExecutionProvider'])
pillar_model = ort.InferenceSession('weights/pillar.onnx', providers=['CPUExecutionProvider'])

def save_result(file_name, annotations, original_image, device):
    """
    Save the result to disk

    Args:
        file_name: The image file name
        annotations: Bondingboxes that are detected
        original_image: Original image object, Image.open(url)
        device: Which devices to run the inference, cpu or gpu

    Returns:
        None
    """
    fig = fast_process(
        annotations=annotations,
        image=original_image,
        device=device,
        scale=(1024 // 1024),
        better_quality=False,
        mask_random_color=True,
        use_retina=False,
        bbox = None,
        withContours=True,
    )

    file_name = file_name.rsplit('/', 1)[1]
    file_name = file_name.rsplit('.', 1)[0] + '.png'

    fig.save('results/' + file_name)


def preprocess(image, size=(640, 640)):
    """
    Preprocess the image
    Adjust to target size and get the scale factor

    Args:
        image: The image file name
        size: The target size

    Returns:
        image, scale_facotr, padding height and padding width
    """
    h, w = image.shape[:2]
    max_size = max(h, w)
    scale_factor = size[0] / max_size
    pad_h = (max_size - h) // 2
    pad_w = (max_size - w) // 2
    pad_image = np.zeros((max_size, max_size, 3), dtype=image.dtype)
    pad_image[pad_h:h + pad_h, pad_w:w + pad_w] = image
    image = cv2.resize(pad_image, size,
                       interpolation=cv2.INTER_LINEAR).astype('float32')
    image /= 255.0
    image = image[None]
    return image, scale_factor, (pad_h, pad_w)


def visualize(image, bboxes, labels, scores, texts):
    """
    Preprocess the image
    Adjust to target size and get the scale factor

    Args:
        image: The image file name
        bboxes: Boundboxes that are detected
        labels: Detection lables
        scores: Confidence score
        texts: Classes of detection objects

    Returns:
        The image width annotated rectangles and lables, scores
    """
    detections = sv.Detections(xyxy=bboxes, class_id=labels, confidence=scores)
    labels = [
        f"{texts[class_id][0]} {confidence:0.2f}" for class_id, confidence in
        zip(detections.class_id, detections.confidence)
    ]

    image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
    image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels)
    return image


def inference(ort_session,
              image_path,
              texts,
              output_dir,
              score_thrushold=0.15,
              size=(640, 640),
              **kwargs):
    """
    Run the inference

    Args:
        ort_session: Onnx model session
        image_path: The path of image wants to be detected
        texts: Classes
        output_dir: Output path
        score_thrushold: Confidence score thrushold
        size: The image size that model can detect

    Returns:
        Boundboxes and scores
    """
    ori_image = cv2.imread(image_path)
    h, w = ori_image.shape[:2]
    image, scale_factor, pad_param = preprocess(ori_image[:, :, [2, 1, 0]],
                                                size)
    input_ort = ort.OrtValue.ortvalue_from_numpy(image.transpose((0, 3, 1, 2)))
    results = ort_session.run(["num_dets", "labels", "scores", "boxes"],
                              {"images": input_ort})
    num_dets, labels, scores, bboxes = results
    
    num_dets = num_dets[0][0]
    labels = labels[0, :num_dets]
    scores = scores[0, :num_dets]
    bboxes = bboxes[0, :num_dets]

    print(scores)

    filtered_indices = scores >= score_thrushold
    filtered_scores = scores[filtered_indices]
    filtered_labels = labels[filtered_indices]
    filtered_bboxes = bboxes[filtered_indices]

    filtered_bboxes /= scale_factor
    filtered_bboxes -= np.array(
        [pad_param[1], pad_param[0], pad_param[1], pad_param[0]])
    filtered_bboxes[:, 0::2] = np.clip(filtered_bboxes[:, 0::2], 0, w)
    filtered_bboxes[:, 1::2] = np.clip(filtered_bboxes[:, 1::2], 0, h)
    filtered_bboxes = filtered_bboxes.round().astype('int')

    print(filtered_bboxes)

    image_out = visualize(ori_image, filtered_bboxes, filtered_labels, filtered_scores, texts)
    cv2.imwrite(osp.join(output_dir, osp.basename(image_path)), image_out)

    return [{'bbox': bbox, 'score': score} for bbox, score in zip(filtered_bboxes, filtered_scores)]


def inference_model(is_window_detected, image_path, texts, score_thrushold=0.15, output_dir='./result'):
    #model = ort.InferenceSession(onnx_file, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    
    if is_window_detected:
        return inference(window_model,
              image_path,
              texts,
              output_dir,
              score_thrushold)
    else:
        return inference(pillar_model,
              image_path,
              texts,
              output_dir,
              score_thrushold)