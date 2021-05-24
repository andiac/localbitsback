import json
import math
import time

import numpy as np
import torch
import torch.utils.data
import torchvision
from tqdm import tqdm, trange

import compression.logistic
import compression.models
import compression.mydatasets
from compression.utils import setup, load_imagenet_data, Imgnet32ValWithoutLabels, CIFAR10WithoutLabels, SVHNWithoutLabels, make_testing_dataloader


def main_val(dataloader, model, device):
    """
    Negative log likelihood evaluation only; no compression
    """

    start_time = time.time()
    all_bpds = []
    for batch, in tqdm(dataloader):
        batch_result = model.forward(batch.to(dtype=torch.float64, device=device))
        all_bpds.extend((batch_result['total_logd'] / (-math.log(2.) * int(np.prod(batch.shape[1:])))).tolist())
    print('overall bpd: {} +/- {}, total time: {}, num datapoints: {}'.format(
        np.mean(all_bpds), np.std(all_bpds), time.time() - start_time, len(all_bpds)
    ))
    return len(all_bpds)


def main():
    import argparse, os
    parser = argparse.ArgumentParser()
    # Common arguments
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--encode_bs', type=int, default=100)
    parser.add_argument('--black_box_jacobian_bs', type=int, default=None)
    parser.add_argument('--cpu', action='store_true')
    # Dataset arguments
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--limit_dataset_size', type=int, default=None)
    parser.add_argument('--cifar10_data_path', type=str, default='data/cifar10')
    parser.add_argument('--imagenet32_data_path', type=str, default='data/imagenet32val')
    parser.add_argument('--imagenet64_data_path', type=str, default='data/imagenet64val')
    parser.add_argument('--svhn_data_path', type=str, default='data/svhn')
    parser.add_argument('--celeba_data_path', type=str, default='data/celeba')
    # Model arguments
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--cifar10_model', type=str, default='~/data/flowpp_cifar_model.npz')
    parser.add_argument('--imagenet32_model', type=str, default='~/data/flowpp_imagenet32_model.npz')
    parser.add_argument('--imagenet64_model', type=str, default='~/data/flowpp_imagenet64_model.npz')
    # Script mode
    parser.add_argument('--test_output_filename', type=str, default=None)
    parser.add_argument('--timing_test_count', type=int, default=6)
    # Default compression options
    parser.add_argument('--neg_log_noise_scale', type=int, default=14)
    parser.add_argument('--disc_bits', type=int, default=32)
    parser.add_argument('--disc_range', type=int, default=256)
    parser.add_argument('--ans_init_bits', type=int, default=10000000)
    parser.add_argument('--ans_num_streams', type=int, default=16)
    parser.add_argument('--ans_mass_bits', type=int, default=60)  # probably never need to change this
    args = parser.parse_args()

    setup(seed=args.seed)

    # Load model
    if args.model == 'cifar10':
        model_ctor = compression.models.load_cifar_model
        model_filename = os.path.expanduser(args.cifar10_model)

    elif args.model == 'imagenet32':
        model_ctor = compression.models.load_imagenet32_model
        model_filename = os.path.expanduser(args.imagenet32_model)

    else:
        raise NotImplementedError(args.model)

    # Load data
    if args.dataset == 'imagenet32':
        dataset = Imgnet32ValWithoutLabels(
            data_dir=os.path.expanduser(args.imagenet32_data_path),
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(lambda x_: (x_ * 255).to(dtype=torch.int64)),
            ])
        )
    elif args.dataset == 'imagenet64':
        dataset = load_imagenet_data(os.path.expanduser(args.imagenet64_data_path))
    elif args.dataset == 'cifar10':
        dataset = CIFAR10WithoutLabels(
            root=os.path.expanduser(args.cifar10_data_path), train=False, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(lambda x_: (x_ * 255).to(dtype=torch.int64)),
            ])
        )
    elif args.dataset == 'svhn':
        dataset = SVHNWithoutLabels(
            root=os.path.expanduser(args.svhn_data_path), split="test", download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(lambda x_: (x_ * 255).to(dtype=torch.int64)),
            ])
        )
    else:
        raise NotImplementedError(args.dataset)

    dataloader, dataset = make_testing_dataloader(
        dataset, seed=args.seed, limit_dataset_size=args.limit_dataset_size, bs=args.encode_bs
    )

    device = torch.device('cpu' if args.cpu else 'cuda')
    model = model_ctor(model_filename, force_float32_cond=True).to(device=device)

    num_params = 0
    for param in model.parameters():
        nn = 1
        for s in list(param.size()):
            nn *= s
        num_params += nn

    print("number of params: " + str(num_params))

    num_datapoints_processed = main_val(dataloader, model, device)
    assert num_datapoints_processed == len(dataset)



if __name__ == '__main__':
    main()
