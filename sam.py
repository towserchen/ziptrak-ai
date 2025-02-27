import numpy as np
import torch
from torchvision.transforms import ToTensor
from utils import tool
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gpu_checkpoint_path = "weights/efficientsam_s_gpu.jit"
cpu_checkpoint_path = "weights/efficientsam_s_cpu.jit"

start_time = time.time()

if torch.cuda.is_available():
    model = torch.jit.load(gpu_checkpoint_path)
else:
    model = torch.jit.load(cpu_checkpoint_path)

model.eval()

end_time = time.time()

print(f"SegModel loading time: {end_time - start_time:.4f} seconds")


def segment_with_boxs(
    image,
    global_points,
    global_point_label,
    input_size=1024
):
    if len(global_points) < 2:
        return global_points, global_point_label
    
    print("Original Image : ", image.size)

    input_size = int(input_size)
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))

    print("Scaled Image : ", image.size)

    scaled_points = np.array(
        [[int(x * scale) for x in point] for point in global_points]
    )
    scaled_points = scaled_points[:2]
    scaled_point_label = np.array(global_point_label)[:2]

    if scaled_points.size == 0 and scaled_point_label.size == 0:
        print("No points selected")
        return image, global_points, global_point_label

    nd_image = np.array(image)
    img_tensor = ToTensor()(nd_image)

    pts_sampled = torch.reshape(torch.tensor(scaled_points), [1, 1, -1, 2])
    pts_sampled = pts_sampled[:, :, :2, :]
    pts_labels = torch.reshape(torch.tensor([2, 3]), [1, 1, 2])

    predicted_logits, predicted_iou = model(
        img_tensor[None, ...].to(device),
        pts_sampled.to(device),
        pts_labels.to(device),
    )
    predicted_logits = predicted_logits.cpu()
    all_masks = torch.ge(torch.sigmoid(predicted_logits[0, 0, :, :, :]), 0.5).numpy()
    predicted_iou = predicted_iou[0, 0, ...].cpu().detach().numpy()


    max_predicted_iou = -1
    selected_mask_using_predicted_iou = None
    selected_predicted_iou = None

    for m in range(all_masks.shape[0]):
        curr_predicted_iou = predicted_iou[m]
        if (
            curr_predicted_iou > max_predicted_iou
            or selected_mask_using_predicted_iou is None
        ):
            max_predicted_iou = curr_predicted_iou
            selected_mask_using_predicted_iou = all_masks[m:m+1]
            selected_predicted_iou = predicted_iou[m:m+1]

    results = tool.format_results(selected_mask_using_predicted_iou, selected_predicted_iou, predicted_logits, 0)

    annotations = results[0]["segmentation"]
    annotations = np.array([annotations])

    return annotations