import argparse
import TrainModel
import os
import torch

def parse_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", type=bool, default=True)
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument("--resume", type=bool, default=True)
    parser.add_argument("--seed", type=int, default=19970318)

    parser.add_argument("--trainset", type=str, default="./HDR_train/")
    parser.add_argument("--testset", type=str, default="./HDR_tests/")

    parser.add_argument("--results_savepath", type=str,
                        default="./results/")
    parser.add_argument("--layer", type=int, default=8)

    parser.add_argument('--ckpt_path', default='./Checkpoint/', type=str,
                          metavar='PATH', help='path to checkpoints')
    parser.add_argument('--ckpt', default='', type=str, help='name of the checkpoint to load')

    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--image_size", type=int, default=512, help='None means random resolution')

    parser.add_argument("--max_epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--decay_interval", type=int, default=500)
    parser.add_argument("--decay_ratio", type=float, default=0.1)

    parser.add_argument("--epochs_per_eval", type=int, default=20)
    parser.add_argument("--epochs_per_save", type=int, default=20)

    return parser.parse_args()

def main(cfg):
    t = TrainModel .Trainer(cfg)
    if cfg.train:
        t.fit()
    else:
        t.eval()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    config = parse_config()

    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)
    main(config)





