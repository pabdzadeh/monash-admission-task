import argparse
import sys
import torchvision
import torchvision.transforms as transforms
import torch
import engine
import open_clip


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
        "--zero_shot",
        help="evaluation mode",
        default=True,
        type=bool,
    )

    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()

def prepare_dataset(args):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)
    return train_loader, test_loader

def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()

    train_loader, test_loader = prepare_dataset(args)

    if (args.zero_shot):
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32')
        model.eval()
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        text = tokenizer(["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])
        engine.eval(model, preprocess, text, test_loader)





if __name__ == "__main__":
    main()