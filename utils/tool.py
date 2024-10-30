import numpy as np
import cv2


def is_completely_inside(boxA, boxB):
    """
    Is boxB is inside boxA

    Args:
        np(4,) boxA: first box
        np(4,) boxB: second box

    Returns:
        Bool
    """
    return (
        boxA[0] >= boxB[0] and boxA[1] >= boxB[1] and
        boxA[2] <= boxB[2] and boxA[3] <= boxB[3]
    )



def filter_overlapped_windows(detect_result):
    """
    Filter out low score windows when multiple windows overlap

    Args:
        detect_result: [{'bbox': xx, 'score': 1}]

    Returns:
        filtered detect result
    """
    eliminated_result = []
    filtered_result = []
    large_rectangles = []

    # First step, find out all rects that contains at least one small rect
    for i, big_rect in enumerate(detect_result):
        big_bbox = big_rect["bbox"]
        internal_small_rectangles = []

        for j, small_rect in enumerate(detect_result):
            if i != j and is_completely_inside(small_rect["bbox"], big_bbox):
                internal_small_rectangles.append(small_rect)

        if internal_small_rectangles:
            large_rectangles.append((big_rect, internal_small_rectangles))

    

    # Second step, determine if keeps the big rect or small rects
    for big_rect, small_rects in large_rectangles:
        big_score = big_rect["score"]
        has_higher_score = any(small_rect["score"] > big_score for small_rect in small_rects)

        if has_higher_score:
            # Keep small rects
            eliminated_result.append(big_rect)
            continue
        else:
            # Keep big rect
            eliminated_result.extend(small_rects)

    # Stores all rects that not be eliminated
    for rect in detect_result:
        isEliminated = False

        for eliminated_item in eliminated_result:
            if np.array_equal(eliminated_item['bbox'], rect['bbox']):
                isEliminated = True
                break
        
        if not isEliminated:
            filtered_result.append(rect)

    return filtered_result


# 大矩形内部包含小矩形，则过滤掉大矩形
def filter_larger_rectangle(detect_result):
    """
    If a big rect contains small rects, filter out the big rect

    Args:
        detect_result: [{'bbox': xx, 'score': 1}]

    Returns:
        filtered detect result
    """
    to_keep = set(range(len(detect_result)))

    for i, boxA in enumerate(detect_result):
        for j, boxB in enumerate(detect_result):
            if i != j and is_completely_inside(boxA['bbox'], boxB['bbox']):
                if j in to_keep:
                    to_keep.remove(j)

    return [detect_result[i] for i in to_keep]


def get_corners(mask):
    """
    Get four cornors' coordinates of a mask

    Args:
        mask: the mask detected from SAM AI

    Returns:
        Dict coordinates
    """
    if len(mask.shape) == 3 and mask.shape[0] == 1:
        mask = np.squeeze(mask, axis=0)

    points = np.argwhere(mask)

    if points.size == 0:
        return None

    top_left = points[np.argmin(points[:, 0] + points[:, 1])]
    top_right = points[np.argmin(points[:, 0] - points[:, 1])]
    bottom_right = points[np.argmax(points[:, 0] + points[:, 1])]
    bottom_left = points[np.argmax(points[:, 0] - points[:, 1])]

    return {
        'top_left': (top_left[1], top_left[0]), # (x, y)
        'top_right': (top_right[1], top_right[0]),
        'bottom_left': (bottom_left[1], bottom_left[0]),
        'bottom_right': (bottom_right[1], bottom_right[0])
    }


def drawPolyLine(file_path, cornersList):
    """
    Draw a poly line, will open a window, not suitable for Non-GUI environment

    Args:
        String file_path: the file path of the image
        List cornersList: corners' coordinates

    Returns:
        None
    """
    image = cv2.imread(file_path)

    for corners in cornersList:
        points = np.array(corners, np.int32)

        color = (0, 255, 0)
        thickness = 2
        cv2.polylines(image, [points], isClosed=True, color=color, thickness=thickness)            


    cv2.namedWindow('Image with Quadrilateral', cv2.WINDOW_NORMAL)
    cv2.imshow('Image with Quadrilateral', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Get the height of a mask
def calculate_mask_height(mask):
    """
    Get the height of a mask

    Args:
        mask: mask

    Returns:
        Double height
    """
    if len(mask.shape) == 3 and mask.shape[0] == 1:
        mask = np.squeeze(mask, axis=0)

    points = np.argwhere(mask)

    if points.size == 0:
        return 0

    min_y = np.min(points[:, 0])
    max_y = np.max(points[:, 0])

    height = max_y - min_y + 1

    return height


def filter_small_masks(mask, threshold = 0.1):
    """
    Filter out the rect that area smaller than a threshold of the max area

    Args:
        mask: mask
        Double: threshold, the max area will multiply by this value

    Returns:
        Filtered mask
    """
    original_shape = mask.shape

    if mask.dtype != bool:
        raise ValueError("mask dtype must be bool")
    
    if len(mask.shape) == 3 and mask.shape[0] == 1:
        mask = np.squeeze(mask, axis=0)

    mask_uint8 = mask.astype(np.uint8)

    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]

    max_area = max(areas) if areas else 0
    threshold_area = max_area * threshold

    filtered_mask = np.zeros_like(mask_uint8)


    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= threshold_area:
            cv2.drawContours(filtered_mask, [contour], -1, (1), thickness=cv2.FILLED)

    final_mask = filtered_mask.astype(bool)

    if original_shape[0] == 1:
        final_mask = np.expand_dims(final_mask, axis=0)

    return final_mask


def filter_short_masks(masks, threshold = 0.45):
    """
    Filter out the rect that the height smaller than a threshold of the max height

    Args:
        mask: mask
        Double: threshold, the max height will multiply by this value

    Returns:
        Filtered mask
    """
    
    heights = [calculate_mask_height(mask) for mask in masks]
    
    if not heights:
        return []

    max_height = max(heights)
    threshold_height = max_height * threshold

    filtered_masks = [mask for mask, height in zip(masks, heights) if height >= threshold_height]

    return filtered_masks


def calculate_mask_width(mask):
    """
    Get the width of the mask

    Args:
        mask: mask

    Returns:
        Double height
    """
    
    if len(mask.shape) == 3 and mask.shape[0] == 1:
        mask = np.squeeze(mask, axis=0)

    points = np.argwhere(mask)

    if points.size == 0:
        return 0

    rows = np.unique(points[:, 0])
    max_width = 0

    for row in rows:
        cols = points[points[:, 0] == row][:, 1]
        if len(cols) > 0:
            width = cols.max() - cols.min() + 1
            max_width = max(max_width, width)

    return max_width


def filter_masks_by_aspect_ratio_avg(masks, threshold = 0.6):
    """
    Filter out the rects according to the threshold of avg aspect ratio

    Args:
        mask: mask
        Double: threshold, the avg aspect ratio will multiply by this value

    Returns:
        Filtered masks
    """
    masks_formated = []
    filtered_masks = []
    aspect_ratio_total = 0
    aspect_ratio_threshold = 0

    for mask in masks:
        width = calculate_mask_width(mask)
        height = calculate_mask_height(mask)

        if height > 0:
            aspect_ratio = width / height
            aspect_ratio_total += aspect_ratio

            masks_formated.append({
                'mask': mask,
                'aspect_ratio': aspect_ratio
            })
    
    del masks
    
    aspect_ratio_threshold = aspect_ratio_total / len(masks_formated) * threshold

    print(f"avg: {aspect_ratio_total / len(masks_formated)}, thre: {aspect_ratio_threshold}")

    for mask in masks_formated:
        if mask['aspect_ratio'] <= aspect_ratio_threshold:
            filtered_masks.append(mask['mask'])                

    return filtered_masks



def filter_masks_by_aspect_ratio(masks, threshold = 0.5):
    """
    Filter out the rects according to the threshold 

    Args:
        mask: mask
        Double: threshold, the avg aspect ratio will multiply by this value

    Returns:
        Filtered mask
    """

    filtered_masks = []

    for mask in masks:
        width = calculate_mask_width(mask)
        height = calculate_mask_height(mask)

        if height > 0:
            aspect_ratio = width / height
            
            if aspect_ratio < threshold:
                filtered_masks.append(mask)           

    return filtered_masks


def get_mask_bounds(mask):
    """
    Get bounds from a mask

    Args:
        mask: mask

    Returns:
        Tuple bounds
    """

    points = np.argwhere(mask)
    if points.size == 0:
        return (None, None)

    left_bound = points[:, 1].min()
    right_bound = points[:, 1].max()

    return (left_bound, right_bound)



def sort_masks_left_to_right(masks):
    """
    Sort masks from left to right

    Args:
        masks: masks

    Returns:
        sorted masks
    """
    bounds = [get_mask_bounds(mask) for mask in masks]
    sorted_masks = [mask for _, mask in sorted(zip(bounds, masks), key=lambda x: x[0][0] if x[0][0] is not None else float('inf'))]

    return sorted_masks



def filter_low_score_pillar_bboxes(detect_result, thrushhold = 0.38):
    """
    Filter out low score bboxs

    Args:
        detect_result: the result from detection AI
        Double thrushold: thrushhold

    Returns:
        filtered detect_result
    """
    filtered_result = []

    for item in detect_result:
        if item['score'] > thrushhold:
            filtered_result.append(item)

    return filtered_result


def format_results(masks, scores, logits, filter=0):
    annotations = []
    n = len(scores)
    for i in range(n):
        annotation = {}

        mask = masks[i]
        tmp = np.where(mask != 0)
        if np.sum(mask) < filter:
            continue
        annotation["id"] = i
        annotation["segmentation"] = mask
        annotation["bbox"] = [
            np.min(tmp[0]),
            np.min(tmp[1]),
            np.max(tmp[1]),
            np.max(tmp[0]),
        ]
        annotation["score"] = scores[i]
        annotation["area"] = annotation["segmentation"].sum()
        annotations.append(annotation)
    return annotations