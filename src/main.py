import argparse
import sys
import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import Subset

import engine
import open_clip
import torch.nn as nn
import torch.optim as optim
import numpy as np

import torch
import torch.nn as nn

import torch
import torch.nn as nn


class CLIPWithLinearProbe(nn.Module):
    def __init__(self, clip_model, num_classes, dropout=0.5):
        super().__init__()
        self.clip = clip_model

        # Freeze backbone
        for param in self.clip.parameters():
            param.requires_grad = False

        # Get feature dimension
        dummy = torch.randn(1, 3, 224, 224)  # adjust input size if needed
        feature_dim = self.clip.visual(dummy).shape[1]
        print(f"Visual feature dimension: {feature_dim}")

        # Linear head (trainable)
        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(feature_dim, num_classes))
        self.linear_head = nn.Sequential(*layers)  # keep it separate

    def forward(self, image=None, text=None):
        # Get frozen features from visual backbone
        features = self.clip.visual(image)  # [B, feature_dim]
        # Pass through trainable linear head
        logits = self.linear_head(features)
        return logits

def parse_args():
    """
    Parse the following arguments for a default parser.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
    """
    parser = argparse.ArgumentParser(
        description="Provide training and testing pipeline."
    )

    parser.add_argument(
        "--batch_size",
        help="batch size for training and evaluation",
        default=16,
        type=int,
    )

    parser.add_argument(
        "--output_dir",
        help="checkpoints root dir",
        default=".",
        type=str,
    )

    parser.add_argument(
        "--pretrained_model",
        help="name of  pretrained variants of CLIP on ViT-b-32",
        default="openai",
        type=str,
    )

    parser.add_argument(
        "--linear_probe",
        help="train linear probe on ViT-b-32",
        default=True,
        type=bool,
    )

    parser.add_argument(
        "--zero_shot",
        help="evaluation mode",
        default=False,
        type=bool,
    )

    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def apply_preprocess(x, preprocess):
    return preprocess(x)


def prepare_dataset(args, preprocess):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.ToPILImage(), transforms.Lambda(preprocess)],
        )

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    dataset_size = len(train_set)
    indices = list(range(dataset_size))
    np.random.seed(42)
    np.random.shuffle(indices)

    split = int(0.8 * dataset_size)
    train_indices, val_indices = indices[:split], indices[split:]

    train_dataset = Subset(train_set, train_indices)
    val_dataset = Subset(train_set, val_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=2)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)
    return train_loader, val_loader, test_loader

def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()

    if args.linear_probe:
        print("-" * 60)
        print("Linear Probe on ViT-b-32")
        print("-" * 60)
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32')

        # for param in model.parameters():
        #     param.requires_grad = False

        num_features = 768
        num_classes = 10

        model = CLIPWithLinearProbe(clip_model=model,
                               num_classes=num_classes, dropout=0.5)
        print(model)
        # dropout_prob = 0.5
        # model.fc = nn.Sequential(
        #     nn.Dropout(dropout_prob),
        #     nn.Linear(num_features, num_classes)
        # )
        criterion = nn.CrossEntropyLoss()
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        train_loader, val_loader, test_loader = prepare_dataset(args, preprocess)
        # text = tokenizer(["a photo of " + x for x in ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]])
        optimizer = optim.Adam(model.linear_head.parameters(), lr=1e-3, weight_decay=1e-4)
        text = tokenizer(["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])
        engine.linear_probe_train(model, text, train_loader, val_loader, optimizer, criterion, device=torch.device('cpu'))

    if args.zero_shot:
        print("-" * 60)
        print("Zero Shot")
        print("-" * 60)
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32')
        print(model)
        model.eval()
        train_loader, val_loader, test_loader = prepare_dataset(args, preprocess)
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        # text = tokenizer(["a photo of " + x for x in ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]])
        text = tokenizer(["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])
        engine.eval(model, text, test_loader)


if __name__ == "__main__":
    main()