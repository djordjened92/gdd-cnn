import argparse
import logging
from collections import defaultdict
from utils.utils import parse_command
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torchinfo import summary

from utils.distributed import get_rank, is_main_process
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
    inspection = summary(model, input_size=(args.batch_size, args.input_channels, *target_size), mode="eval")
    logging.info(f"is cuda available? {torch.cuda.is_available()}")
    logging.info(f"Using device {device}")
    logging.info("\n")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name().lower()
        high_precision_gpus = ["a100", "a30", "a40", "a5000", "a6000", "3090", "6000", "5000", "4070", "orin"]
        if any([gpu.lower() in gpu_name for gpu in high_precision_gpus]):
            logging.info("Setting float32_matmul_precision to highest")
            torch.set_float32_matmul_precision("high")

    logging.info(inspection)
    logging.info("\n")

    inputs = torch.rand((1, 3, args.img_height, args.img_width)).to(device)
    flops = FlopCountAnalysis(model, inputs)
    print("FLOPs:", flops.total())
    logging.info(f"FLOPs: {flops.total()}")

    # Add forward hooks
    activation_outs = defaultdict(list)

    def get_activation_hook(name):
        def hook(module, input, output):
            activation_outs[name].append(output.detach())
        return hook
    
    print(model.backbone[1].stage[3].activation)
    model.backbone[1].stage[3].activation.register_forward_hook(get_activation_hook('stage1_GDB3'))

    # Load image
    with torch.no_grad():
        ds_iter = iter(testset)
        image, label = next(ds_iter)
        image = image_preprocess(image).to('cuda')
        pred = torch.argmax(model(image.unsqueeze(0))).item()

    print(activation_outs['stage1_GDB3'][0].shape)

    # Plot feature maps
    fmaps = activation_outs['stage1_GDB3'][0][0].cpu()

    x_min = fmaps.amin(dim=(1, 2), keepdim=True)  # shape: (N, 1, 1)
    x_max = fmaps.amax(dim=(1, 2), keepdim=True)  # shape: (N, 1, 1)

    # Normalize to [0, 1] per slice
    fmaps_norm = (fmaps - x_min) / (x_max - x_min + 1e-8)

    fmaps_split = torch.split(fmaps_norm, 10, 0)
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))  # Width x Height in inches

    for ax, tensor in zip(axes, fmaps_split):
        ax.imshow(tensor.mean(dim=0), cmap='viridis')
        ax.axis('off')  # Hide axes

    plt.tight_layout()
    plt.savefig('stage1_GDB3.png', dpi=300, bbox_inches='tight', pad_inches=0)

if __name__=='__main__':
    args = parse_command()
    inference(args)