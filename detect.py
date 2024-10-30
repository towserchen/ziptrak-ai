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

def save_result(file_name, annotations, original_image, device, input_size=1024, better_quality=False, mask_random_color=True, use_retina=False, withContours=True):
    fig = fast_process(
        annotations=annotations,
        image=original_image,
        device=device,
        scale=(1024 // input_size),
        better_quality=better_quality,
        mask_random_color=mask_random_color,
        use_retina=use_retina,
        bbox = None,
        withContours=withContours,
    )

    file_name = file_name.rsplit('/', 1)[1]
    file_name = file_name.rsplit('.', 1)[0] + '.png'

    fig.save('results/' + file_name)


def preprocess(image, size=(640, 640)):
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
    # normal export
    # with NMS and postprocessing
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

    # 过滤 scores 和 bboxes
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

    print('no-nms')
    print(filtered_bboxes)

    image_out = visualize(ori_image, filtered_bboxes, filtered_labels, filtered_scores, texts)
    cv2.imwrite(osp.join(output_dir, osp.basename(image_path)), image_out)

    return [{'bbox': bbox, 'score': score} for bbox, score in zip(filtered_bboxes, filtered_scores)]


def inference_with_postprocessing(ort_session,
                                  image_path,
                                  texts,
                                  output_dir,
                                  size=(640, 640),
                                  nms_thr=0.7,
                                  score_thr=0.3,
                                  max_dets=300):
    # export with `--without-nms`
    ori_image = cv2.imread(image_path)
    h, w = ori_image.shape[:2]
    image, scale_factor, pad_param = preprocess(ori_image[:, :, [2, 1, 0]],
                                                size)
    input_ort = ort.OrtValue.ortvalue_from_numpy(image.transpose((0, 3, 1, 2)))
    results = ort_session.run(["scores", "boxes"], {"images": input_ort})
    scores, bboxes = results
    # move numpy array to torch
    ori_scores = torch.from_numpy(scores[0]).to('cuda:0')
    ori_bboxes = torch.from_numpy(bboxes[0]).to('cuda:0')

    scores_list = []
    labels_list = []
    bboxes_list = []
    # class-specific NMS
    for cls_id in range(len(texts)):
        cls_scores = ori_scores[:, cls_id]
        labels = torch.ones(cls_scores.shape[0], dtype=torch.long) * cls_id
        keep_idxs = nms(ori_bboxes, cls_scores, iou_threshold=nms_thr)
        cur_bboxes = ori_bboxes[keep_idxs]
        cls_scores = cls_scores[keep_idxs]
        labels = labels[keep_idxs]
        scores_list.append(cls_scores)
        labels_list.append(labels)
        bboxes_list.append(cur_bboxes)

    scores = torch.cat(scores_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    bboxes = torch.cat(bboxes_list, dim=0)

    keep_idxs = scores > score_thr
    scores = scores[keep_idxs]
    labels = labels[keep_idxs]
    bboxes = bboxes[keep_idxs]
    if len(keep_idxs) > max_dets:
        _, sorted_idx = torch.sort(scores, descending=True)
        keep_idxs = sorted_idx[:max_dets]
        bboxes = bboxes[keep_idxs]
        scores = scores[keep_idxs]
        labels = labels[keep_idxs]

    # Get candidate predict info by num_dets
    scores = scores.cpu().numpy()
    bboxes = bboxes.cpu().numpy()
    labels = labels.cpu().numpy()

    bboxes -= np.array(
        [pad_param[1], pad_param[0], pad_param[1], pad_param[0]])
    bboxes /= scale_factor
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, w)
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, h)
    bboxes = bboxes.round().astype('int')

    print('nms')
    print(bboxes)

    image_out = visualize(ori_image, bboxes, labels, scores, texts)
    cv2.imwrite(osp.join(output_dir, osp.basename(image_path)), image_out)

    #return image_out
    return bboxes


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