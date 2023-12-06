import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F
import torchvision
import models
import utils
import tabulate
from torch.utils.data import random_split
import pandas as pd

parser = argparse.ArgumentParser(description='SGD-SWA-ASWA training')
parser.add_argument('--dir', type=str, default='.', required=False, help='training directory (default: None)')

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=16, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default='VGG16', required=False, metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--optim', type=str, default='SGD', help='dataset name (default: CIFAR10)')

parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')

parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=25, metavar='N', help='save frequency (default: 25)')
parser.add_argument('--eval_freq', type=int, default=1, metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--lr_init', type=float, default=0.1, metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')

parser.add_argument('--swa', action='store_true', help='swa usage flag (default: off)')
parser.add_argument('--swa_start', type=float, default=1, metavar='N', help='SWA start epoch number (default: 161)')
parser.add_argument('--swa_lr', type=float, default=0.05, metavar='LR', help='SWA LR (default: 0.05)')
parser.add_argument('--swa_c_epochs', type=int, default=1, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')

parser.add_argument('--aswa', action='store_true', help='aswa usage flag (default: off)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--val_ratio', type=float, default='0.1')

# TODO: add device
args = parser.parse_args()

print('Preparing directory %s' % args.dir)
os.makedirs(args.dir, exist_ok=True)

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

print('Loading dataset:', args.dataset)
assert 1.0 > args.val_ratio > 0.0
if args.dataset == "CIFAR10":
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                             transform=model_cfg.transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                            transform=model_cfg.transform_test)

    train_size = int(len(train_set) * (1 - args.val_ratio))
    val_size = len(train_set) - train_size
elif args.dataset == "CIFAR100":
    train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                              transform=model_cfg.transform_train)
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True,
                                             transform=model_cfg.transform_test)

    train_size = int(len(train_set) * (1 - args.val_ratio))
    val_size = len(train_set) - train_size
else:
    print("Incorred dataset", args.dataset)

num_classes = max(train_set.targets) + 1
train_set, val_set = random_split(train_set, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

print(f"|Train|:{len(train_set)} |Val|:{len(val_set)} t|Test|:{len(test_set)}")

loaders = {
    'train': torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    ),
    'val': torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    ),
    'test': torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

}

print('Preparing model')
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('SWA and ASWA training')
swa_model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
swa_n = 0

aswa_model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
aswa_model.load_state_dict(swa_model.state_dict())
aswa_ensemble_weights = [0]

model.to(device)
swa_model.to(device)
aswa_model.to(device)


def schedule(epoch):
    t = (epoch) / (args.swa_start if args.swa else args.epochs)
    lr_ratio = args.swa_lr / args.lr_init if args.swa else 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return args.lr_init * factor


criterion = F.cross_entropy
if args.optim == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_init, momentum=args.momentum, weight_decay=args.wd)
elif args.optim == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init)
else:
    print("NNN")
    exit(1)

# Not tested
if args.resume is not None:
    print('Resume training from %s' % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if args.swa:
        swa_state_dict = checkpoint['swa_state_dict']
        if swa_state_dict is not None:
            swa_model.load_state_dict(swa_state_dict)
        swa_n_ckpt = checkpoint['swa_n']
        if swa_n_ckpt is not None:
            swa_n = swa_n_ckpt

start_epoch = 0
df = []
for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()
    # (Adjust LR)
    if args.optim == "SGD" and epoch > 0:
        lr = schedule(epoch)
        utils.adjust_learning_rate(optimizer, lr)
    else:
        lr = args.lr_init

    # (.) Store the epoch info
    epoch_res = dict()

    # (.) Running model over the training data
    train_res = utils.train_epoch(loaders['train'], model, criterion, optimizer, device)
    # (.) Compute BN update before checking val performance
    utils.bn_update(loaders['train'], model)
    val_res = utils.eval(loaders['val'], model, criterion, device)
    test_res = utils.eval(loaders['test'], model, criterion, device)

    epoch_res["Running"] = {"train": train_res, "val": val_res, "test": test_res}

    epoch_res["SWA"] = {"train": {"loss": "-", "accuracy": "-"}, "val": {"loss": "-", "accuracy": "-"},
                        "test": {"loss": "-", "accuracy": "-"}}
    epoch_res["ASWA"] = {"train": {"loss": "-", "accuracy": "-"}, "val": {"loss": "-", "accuracy": "-"},
                         "test": {"loss": "-", "accuracy": "-"}}

    if args.swa and (epoch + 1) >= args.swa_start and (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0:
        print("here")
        # (1) Apply SWA
        utils.moving_average(swa_model, model, 1.0 / (swa_n + 1.0))
        swa_n += 1.0

        # (2) Apply ASWA
        # (2.1) Compute BN update before checking val performance
        utils.bn_update(loaders['train'], aswa_model)
        current_val = utils.eval(loaders['val'], aswa_model, criterion, device)
        # (2.2) Save current ASWA model
        current_aswa_state_dict = aswa_model.state_dict()
        aswa_state_dict = aswa_model.state_dict()

        # (2.3) Perform provisional param epdate
        for k, params in model.state_dict().items():
            # aswa_state_dict[k] = (aswa_state_dict[k] * aswa_n + params) / (1 + aswa_n)
            aswa_state_dict[k] = (aswa_state_dict[k] * sum(aswa_ensemble_weights) + params) / (
                    1 + sum(aswa_ensemble_weights))

        # (2.4) Test provisional ASWA
        aswa_model.load_state_dict(aswa_state_dict)
        utils.bn_update(loaders['train'], aswa_model)
        prov_val = utils.eval(loaders['val'], aswa_model, criterion, device)

        if prov_val["accuracy"] >= current_val["accuracy"]:
            aswa_ensemble_weights.append(1.0)
        else:
            aswa_model.load_state_dict(current_aswa_state_dict)

    # Compute validation performances to report
    # if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
    utils.bn_update(loaders['train'], swa_model)
    utils.bn_update(loaders['train'], aswa_model)

    epoch_res["SWA"] = {
        "train": utils.eval(loaders['train'], swa_model, criterion, device),
        "val": utils.eval(loaders['val'], swa_model, criterion, device),
        "test": utils.eval(loaders['test'], swa_model, criterion, device)}

    epoch_res["ASWA"] = {
        "train": utils.eval(loaders['train'], aswa_model, criterion, device),
        "val": utils.eval(loaders['val'], aswa_model, criterion, device),
        "test": utils.eval(loaders['test'], aswa_model, criterion, device)}

    """
    if (epoch + 1) % args.save_freq == 0:
        utils.save_checkpoint(
            args.dir,
            epoch + 1,
            state_dict=model.state_dict(),
            swa_state_dict=swa_model.state_dict() if args.swa else None,
            swa_n=swa_n if args.swa else None,
            aswa_state_dict=aswa_model.state_dict() if args.aswa else None,
            aswa_ensemble_weights=aswa_ensemble_weights if args.aswa else None,
            optimizer=optimizer.state_dict())
    """
    time_ep = time.time() - time_ep

    columns = ["ep", "time", "lr", "train_loss", "train_acc",
               "test_acc", "swa_train_acc", "swa_val_acc", "swa_test_acc",
               "aswa_train_acc", "aswa_val_acc", "aswa_test_acc"]
    values = [epoch + 1, time_ep, lr,
              epoch_res["Running"]["train"]["loss"], epoch_res["Running"]["train"]["accuracy"],
              epoch_res["Running"]["test"]["accuracy"],
              epoch_res["SWA"]["train"]["accuracy"],
              epoch_res["SWA"]["val"]["accuracy"],
              epoch_res["SWA"]["test"]["accuracy"],
              epoch_res["ASWA"]["train"]["accuracy"],
              epoch_res["ASWA"]["val"]["accuracy"],
              epoch_res["ASWA"]["test"]["accuracy"]]

    df.append(values)
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='4.4f')

    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)

if args.epochs % args.save_freq != 0:
    utils.save_checkpoint(
        args.dir,
        args.epochs,
        state_dict=model.state_dict(),
        swa_state_dict=swa_model.state_dict() if args.swa else None,
        swa_n=swa_n if args.swa else None,
        optimizer=optimizer.state_dict()
    )

df = pd.DataFrame(df, columns=columns)  # .save_csv("demir.csv")

df.to_csv(f"{args.dir}/results.csv")

utils.bn_update(loaders['train'], model)
print("Running model Train: ", utils.eval(loaders['train'], model, criterion, device))
print("Runing model Val:", utils.eval(loaders['val'], model, criterion, device))
print("Running model Test:", utils.eval(loaders['test'], model, criterion, device))

if args.swa:
    utils.bn_update(loaders['train'], swa_model)
    print("SWA Train: ", utils.eval(loaders['train'], swa_model, criterion, device))
    print("SWA Val:", utils.eval(loaders['val'], swa_model, criterion, device))
    print("SWA Test:", utils.eval(loaders['test'], swa_model, criterion, device))

if args.aswa:
    utils.bn_update(loaders['train'], aswa_model)
    print("ASWA Train: ", utils.eval(loaders['train'], aswa_model, criterion, device))
    print("ASWA Val:", utils.eval(loaders['val'], aswa_model, criterion, device))
    print("ASWA Test:", utils.eval(loaders['test'], aswa_model, criterion, device))
