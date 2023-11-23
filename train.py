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
parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--dir', type=str, default='.', required=False, help='training directory (default: None)')

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
#parser.add_argument('--data_path', type=str, default='.', required=False, metavar='PATH',                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=1024, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
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
parser.add_argument('--swa_start', type=float, default=161, metavar='N', help='SWA start epoch number (default: 161)')
parser.add_argument('--swa_lr', type=float, default=0.05, metavar='LR', help='SWA LR (default: 0.05)')
parser.add_argument('--swa_c_epochs', type=int, default=1, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')

parser.add_argument('--aswa', action='store_true', help='aswa usage flag (default: off)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

# TODO: add device
args = parser.parse_args()

print('Preparing directory %s' % args.dir)
os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

print('Loading dataset:', args.dataset)
"""
ds = getattr(torchvision.datasets, args.dataset)
path = os.path.join(args.data_path, args.dataset.lower())
train_set = ds(path, train=True, download=True, transform=model_cfg.transform_train)
test_set = ds(path, train=False, download=True, transform=model_cfg.transform_test)
"""
if args.dataset=="CIFAR10":
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=model_cfg.transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=model_cfg.transform_train)
    
    train_size = int(len(train_set)*0.7)
    val_size=len(train_set)-train_size
else:
    print("Not implemented")
    exit(1)
num_classes = max(train_set.targets) + 1
train_set, val_set = random_split(train_set, [train_size, val_size],generator=torch.Generator().manual_seed(args.seed))

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
model.to(device)

if args.swa:
    print('SWA training')
    swa_model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
    swa_model.to(device)
    swa_n = 0

if args.aswa:
    print('ASWA training')
    aswa_model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
    aswa_model.to(device)
    aswa_n = 0

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
if args.optim=="SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_init,momentum=args.momentum,weight_decay=args.wd)
elif args.optim=="Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init)
else:
    print("NNN")
    exit(1)

start_epoch = 0
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

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'val_loss', 'val_acc', 'time']
if args.swa:
    columns.extend(['swa_val_loss', 'swa_val_acc'])
    swa_res = {'loss': None, 'accuracy': None}

if args.aswa:
    columns.extend(['aswa_val_loss', 'aswa_val_acc'])    
    aswa_res = {'loss': -1, 'accuracy': -1}

utils.save_checkpoint(
    args.dir,
    start_epoch,
    state_dict=model.state_dict(),
    swa_state_dict=swa_model.state_dict() if args.swa else None,
    swa_n=swa_n if args.swa else None,
    aswa_state_dict=aswa_model.state_dict() if args.aswa else None,
    aswa_n=aswa_n if args.aswa else None,
    optimizer=optimizer.state_dict()
)

for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()
    # (Adjust LR)
    if args.optim=="SGD":
        lr = schedule(epoch)
        utils.adjust_learning_rate(optimizer, lr)
    else:
        lr = args.lr_init

    # A single iteration over training dataset
    train_res = utils.train_epoch(loaders['train'], model, criterion, optimizer, device)

    #if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
    val_res = utils.eval(loaders['val'], model, criterion)
    #else:
    #    test_res = {'loss': None, 'accuracy': None}

    if args.swa and (epoch + 1) >= args.swa_start and (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0:
        utils.moving_average(swa_model, model, 1.0 / (swa_n + 1))
        swa_n += 1
        if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
            utils.bn_update(loaders['train'], swa_model)
            swa_res = utils.eval(loaders['val'], swa_model, criterion)
        else:
            swa_res = {'loss': None, 'accuracy': None}
    
    if args.aswa:
        if val_res["accuracy"]> aswa_res["accuracy"]:
            # hard update
            aswa_model.load_state_dict(model.state_dict())
            aswa_n=1
            aswa_res= {k:v for k,v in val_res.items()}
            torch.save(aswa_model.state_dict(), f"{args.dir}/aswa.pt")
        else:
            # Load model
            aswa_model.load_state_dict(torch.load(f"{args.dir}/aswa.pt"))
            # Perform Provisional Soft Update
            utils.moving_average(aswa_model, model, 1.0 / (aswa_n + 1))
            utils.bn_update(loaders['train'], aswa_model)
            # Provisional val performance
            temp_aswa_res = utils.eval(loaders['val'], aswa_model, criterion)

            if temp_aswa_res["accuracy"] > aswa_res["accuracy"]:
                "Soft Update"
                aswa_n +=1
                aswa_res = {k:v for k,v in temp_aswa_res.items()}
                torch.save(aswa_model.state_dict(), f"{args.dir}/aswa.pt")
            else:
                "Reject Update"
                aswa_model.load_state_dict(torch.load(f"{args.dir}/aswa.pt"))


    if (epoch + 1) % args.save_freq == 0:
        utils.save_checkpoint(
            args.dir,
            epoch + 1,
            state_dict=model.state_dict(),
            swa_state_dict=swa_model.state_dict() if args.swa else None,
            swa_n=swa_n if args.swa else None,
            aswa_state_dict=aswa_model.state_dict() if args.aswa else None,
            aswa_n=aswa_n if args.aswa else None,
            optimizer=optimizer.state_dict())

    time_ep = time.time() - time_ep
    
    columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'val_loss', 'val_acc', 'time']


    values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'], val_res['loss'], val_res['accuracy'], time_ep]
    
    if args.swa:
        columns.extend(['swa_val_loss', 'swa_val_acc'])
        values.extend([swa_res['loss'], swa_res['accuracy']])

    
    if args.aswa:
        columns.extend(['aswa_val_loss', 'aswa_val_acc'])
        values.extend([aswa_res['loss'], aswa_res['accuracy']])  


    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
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


print("Running model Train: ",utils.eval(loaders['train'], model, criterion))
print("Runing model Val:", utils.eval(loaders['val'], model, criterion))
print("Running model Test:",utils.eval(loaders['test'], model, criterion))

if args.swa:
    print("SWA Train: ",utils.eval(loaders['train'], swa_model, criterion))     
    print("SWA Val:", utils.eval(loaders['val'], swa_model, criterion))     
    print("SWA Test:",utils.eval(loaders['test'], swa_model, criterion))

if args.aswa:
    aswa_model.load_state_dict(torch.load(f"{args.dir}/aswa.pt"))
    print("ASWA Train: ",utils.eval(loaders['train'], aswa_model, criterion))     
    print("ASWA Val:", utils.eval(loaders['val'], aswa_model, criterion))     
    print("ASWA Test:",utils.eval(loaders['test'], aswa_model, criterion))

