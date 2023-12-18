#PreResNet

#CIFAR10 
model="WideResNet28x10"
path="CIFAR10$model"

mkdir -p $path/FirstRun
mkdir $path/SecondRun
mkdir $path/ThirdRun

n_epochs=1000

# 1% of the training dataset will be used as a validation dataset
if [false]; then
v_ratio=0.01
name="Original_CIFAR10_WideResNet28x10_1000E_Val_001_ratio"
# Run 1
torchrun --standalone --nproc_per_node=gpu ddp_train.py --dir="$path/FirstRun/$name" --dataset=CIFAR10 --model=$model --epochs=$n_epochs --lr_init=0.1 --wd=3e-4 --swa --aswa --swa_start=126 --swa_lr=0.01 --val_ratio=$v_ratio --eval_freq 1 > "$path/FirstRun/$name.log"
# Run 2
torchrun --standalone --nproc_per_node=gpu ddp_train.py --dir="$path/SecondRun/$name" --dataset=CIFAR10 --model=$model --epochs=$n_epochs --lr_init=0.1 --wd=3e-4 --swa --aswa --swa_start=126 --swa_lr=0.01 --val_ratio=$v_ratio --eval_freq 1 > "$path/SecondRun/$name.log"
# Run 3
torchrun --standalone --nproc_per_node=gpu ddp_train.py --dir="$path/ThirdRun/$name" --dataset=CIFAR10 --model=$model --epochs=$n_epochs --lr_init=0.1 --wd=3e-4 --swa --aswa --swa_start=126 --swa_lr=0.01 --val_ratio=$v_ratio --eval_freq 1 > "$path/ThirdRun/$name.log"
else
# 5% of the training dataset will be used as a validation dataset
v_ratio=0.05
name="Original_CIFAR10_WideResNet28x10_1000E_Val_005_ratio"
# 5%
torchrun --standalone --nproc_per_node=gpu ddp_train.py --dir="$path/FirstRun/$name" --dataset=CIFAR10 --model=$model --epochs=$n_epochs --lr_init=0.1 --wd=3e-4 --swa --aswa --swa_start=126 --swa_lr=0.01 --val_ratio=$v_ratio --eval_freq 1 > "$path/FirstRun/$name.log"
# 5%
torchrun --standalone --nproc_per_node=gpu ddp_train.py --dir="$path/SecondRun/$name" --dataset=CIFAR10 --model=$model --epochs=$n_epochs --lr_init=0.1 --wd=3e-4 --swa --aswa --swa_start=126 --swa_lr=0.01 --val_ratio=$v_ratio --eval_freq 1 > "$path/SecondRun/$name.log"
# 5%2
torchrun --standalone --nproc_per_node=gpu ddp_train.py --dir="$path/ThirdRun/$name" --dataset=CIFAR10 --model=$model --epochs=$n_epochs --lr_init=0.1 --wd=3e-4 --swa --aswa --swa_start=126 --swa_lr=0.01 --val_ratio=$v_ratio --eval_freq 1 > "$path/ThirdRun/$name.log"
fi
