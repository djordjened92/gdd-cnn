import argparse
import logging
from utils.utils import parse_command
from matplotlib import pyplot as plt

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import numpy as np

from utils.distributed import get_rank
from utils.net_utils import extract_net_params, extract_optim_params
from fvcore.nn import FlopCountAnalysis

from utils.net_utils import select_arch
from datasets.dataloader import get_dataset

import torchvision

def image_preprocess(image, target_size=224):
    resize = torchvision.transforms.Resize(target_size)
    image = resize(image)
    image = image.to(torch.float32)
    image = image / 255.
    return image

def inference(args: argparse.Namespace) -> None:
    target_size = (args.img_height, args.img_width)

    device = torch.device(f"cuda:{str(get_rank())}" if torch.cuda.is_available() else "cpu")
    args.accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    testset = get_dataset(dataset=args.dataset,
                          data_path=args.data_path,
                          target_size=target_size,
                          num_classes=args.num_classes,
                          subset='test',
                          seed=args.seed,
                          split=args.split,
                          k_folds=args.k_folds,
                          no_validation=args.no_validation)

    if args.num_classes != testset.num_classes:
        raise ValueError(f"Number of classes in config file ({args.num_classes}) does not match the number of classes in the dataset ({testset.num_classes})")

    net_kwargs = extract_net_params(args)
    optim_kwargs = extract_optim_params(args)

    net_kwargs['classes'] = testset.classes
    optim_kwargs['dataset_length'] = 1

    loss = nn.CrossEntropyLoss(reduction="mean", 
                               label_smoothing=args.label_smoothing, 
                               weight=torch.tensor(args.class_weights).to(device) if args.class_weights is not None else None
                               )
    model = select_arch(net_kwargs, loss, optim_kwargs, ckpt_path=args.ckpts_path)

    model.eval()
    summary(model, input_size=(args.batch_size, args.input_channels, *target_size), mode="eval")
    logging.info(f"is cuda available? {torch.cuda.is_available()}")
    logging.info(f"Using device {device}")
    logging.info("\n")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name().lower()
        high_precision_gpus = ["a100", "a30", "a40", "a5000", "a6000", "3090", "6000", "5000", "4070", "orin"]
        if any([gpu.lower() in gpu_name for gpu in high_precision_gpus]):
            logging.info("Setting float32_matmul_precision to highest")
            torch.set_float32_matmul_precision("high")

    # Add hooks
    # Choose the target layer (e.g., layer4 for deeper or layer2 for middle)
    target_layer = model.backbone[4].aggregate_downsample.activation

    # Register hooks to capture gradients and activations
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)

    # Load image
    ind = 254
    i = -1
    ds_iter = iter(testset)
    with torch.no_grad():
        while True:
            i += 1
            image, label = next(ds_iter)
            if i == ind:
                break
    
    # Forward pass
    input_tensor = image_preprocess(image).to('cuda')
    output = model(input_tensor.unsqueeze(0))
    class_idx = output.argmax().item()

    # Backward pass for target class
    model.zero_grad()
    output[0, class_idx].backward()

    # Compute Grad-CAM
    grad = gradients[0][0]           # [C, H, W]
    act = activations[0][0]         # [C, H, W]
    weights = grad.mean(dim=(1, 2)) # Global average pooling
    cam = torch.zeros(act.shape[1:], dtype=torch.float32, device=input_tensor.device)

    for i, w in enumerate(weights):
        cam += w * act[i]

    # cam = F.relu(cam)
    cam -= cam.min()
    cam /= cam.max()
    cam_np = cam.detach().cpu().numpy()

    # Resize CAM to original image size
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy()
    cam_resized = cv2.resize(cam_np, (image.shape[1], image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = image * 0.5 + heatmap * 0.5
    overlay = np.uint8(overlay)

    plt.figure(figsize=(4, 8))
    plt.subplot(2, 1, 1)
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(2, 1, 2)
    plt.imshow(overlay)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('grad_cam_dir/overlay_1233.png')

if __name__=='__main__':
    args = parse_command()
    inference(args)