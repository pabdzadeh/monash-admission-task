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
from linear_probe_model import CLIPWithLinearProbeSimple, CLIPWithLinearProbeExact

cifar10_prompts = {
    "airplane": [
        "airplane",
        "Airplane",
        "a photo of an airplane",
        "a flying airplane",
        "a passenger airplane",
        "an aircraft"
    ],
    "automobile": [
        "automobile",
        "car",
        "a photo of a car",
        "a small car",
        "a sports car",
        "a vehicle"
    ],
    "bird": [
        "bird",
        "a bird",
        "a small bird",
        "a colorful bird",
        "a flying bird",
        "wild bird"
    ],
    "cat": [
        "cat",
        "a cat",
        "domestic cat",
        "a small cat",
        "a cute cat",
        "kitten"
    ],
    "deer": [
        "deer",
        "a deer",
        "a wild deer",
        "a forest deer",
        "a brown deer",
        "stag"
    ],
    "dog": [
        "dog",
        "a dog",
        "puppy",
        "a small dog",
        "a cute dog",
        "domestic dog"
    ],
    "frog": [
        "frog",
        "a frog",
        "a green frog",
        "a small frog",
        "amphibian frog",
        "pond frog"
    ],
    "horse": [
        "horse",
        "a horse",
        "a brown horse",
        "a white horse",
        "a running horse",
        "stallion"
    ],
    "ship": [
        "ship",
        "a ship",
        "a boat",
        "a large ship",
        "a sailing ship",
        "cargo ship"
    ],
    "truck": [
        "truck",
        "a truck",
        "a large truck",
        "a delivery truck",
        "a heavy truck",
        "lorry"
    ]
}


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
        "--linear_probe_type",
        help="type of linear probe to use: Simple, Exact",
        default='exact',
        type=str,
    )

    parser.add_argument(
        "--output_dir",
        help="checkpoints save dir",
        default=".",
        type=str,
    )

    parser.add_argument("--class_name_type",
        help="The index of class_name type from this list: [airplane, Airplane, a photo of an airplane, a flying airplane, a passenger airplane, an aircraft] similar changes for all class names in each index",
        default=0,
        type=int,
    )

    parser.add_argument(
        "--class_name_prefix",
        help="The prefix to be added to all class names",
        default="",
        type=str,
    )

    parser.add_argument(
        "--class_name_postfix",
        help="The postfix to be added to all class names",
        default="",
        type=str,
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        help="checkpoint path",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--pretrained_model",
        help="name of  pretrained variants of CLIP on ViT-b-32",
        default=None,
        type=str,
    )

    parser.add_argument(
        "--linear_probe",
        help="train linear probe on ViT-b-32",
        default=False,
        type=bool,
    )

    parser.add_argument(
        "--zero_shot",
        help="Zero-shot transformer classification",
        default=False,
        type=bool,
    )

    parser.add_argument(
        "--train_epochs",
        help="number of training epochs for linear probe model",
        default=100,
        type=int,
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
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=args.pretrained_model)

        num_classes = 10


        train_loader, val_loader, test_loader = prepare_dataset(args, preprocess)

        if args.linear_probe_type == 'simple':
            model = CLIPWithLinearProbeSimple(clip_model=model,
                                        num_classes=num_classes, dropout=0.5)
            print(model)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.linear_head.parameters(), lr=1e-3, weight_decay=1e-4)
            engine.linear_probe_train(model, train_loader, val_loader, optimizer, criterion, device=torch.device('cuda'),
                                       checkpoint_folder=args.output_dir, resume_from=args.resume_from_checkpoint, total_epochs=args.train_epochs)

        if args.linear_probe_type == 'exact':
            model = CLIPWithLinearProbeExact(clip_model=model,
                                                num_classes=num_classes, device=torch.device('cuda'))
            best_val_acc = model.fit(train_loader, val_loader, max_iter=1000)
            print("Validation accuracy after fit:", best_val_acc)

        print("-" * 60)
        print("2nd Phase :Evaluate the Trained Model")
        print("-" * 60)
        # evaluate the trained model
        if args.linear_probe_type == 'exact':
            preds, y_true = model.predict(val_loader)
        if args.linear_probe_type == 'simple':
            engine.eval_linear_prob(model, test_loader)

    if not args.linear_probe or args.zero_shot:
        print("-" * 60)
        print("Zero Shot")
        print("-" * 60)
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=args.pretrained_model)
        print(model)
        model.eval()
        train_loader, val_loader, test_loader = prepare_dataset(args, preprocess)
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        text = tokenizer([cifar10_prompts[x][args.class_name_type] for x in cifar10_prompts.keys()])
        engine.eval(model, text, test_loader)


if __name__ == "__main__":
    main()