#PreResNet

#CIFAR10  or CIFAR100
dataset="CIFAR100"
model="VGG16"
path="$dataset""$model"

mkdir -p $path/FirstRun
mkdir $path/SecondRun
mkdir $path/ThirdRun

n_epochs=1000

# 5% of the training dataset will be used as a validation dataset
v_ratio=005
name="Original_$dataset""_$model""_1000E_Val_""$v_ratio""_ratio"
v_ratio=0.05
# 5%
torchrun --standalone --nproc_per_node=gpu ddp_train.py --dir="$path/FirstRun/$name" --dataset=$dataset --model=$model --epochs=$n_epochs --lr_init=0.1 --wd=3e-4 --swa --aswa --swa_start=126 --swa_lr=0.01 --val_ratio=$v_ratio --eval_freq 1 > "$path/FirstRun/$name.log"
# 5%
torchrun --standalone --nproc_per_node=gpu ddp_train.py --dir="$path/SecondRun/$name" --dataset=$dataset --model=$model --epochs=$n_epochs --lr_init=0.1 --wd=3e-4 --swa --aswa --swa_start=126 --swa_lr=0.01 --val_ratio=$v_ratio --eval_freq 1 > "$path/SecondRun/$name.log"
# 5%
torchrun --standalone --nproc_per_node=gpu ddp_train.py --dir="$path/ThirdRun/$name" --dataset=$dataset --model=$model --epochs=$n_epochs --lr_init=0.1 --wd=3e-4 --swa --aswa --swa_start=126 --swa_lr=0.01 --val_ratio=$v_ratio --eval_freq 1 > "$path/ThirdRun/$name.log"



model="PreResNet164"
path="$dataset""$model"

mkdir -p $path/FirstRun
mkdir $path/SecondRun
mkdir $path/ThirdRun


# 5% of the training dataset will be used as a validation dataset
v_ratio=005
name="Original_$dataset""_$model""_1000E_Val_""$v_ratio""_ratio"
v_ratio=0.05
# 5%
torchrun --standalone --nproc_per_node=gpu ddp_train.py --dir="$path/FirstRun/$name" --dataset=$dataset --model=$model --epochs=$n_epochs --lr_init=0.1 --wd=3e-4 --swa --aswa --swa_start=126 --swa_lr=0.05 --val_ratio=$v_ratio --eval_freq 1 > "$path/FirstRun/$name.log"
# 5%
torchrun --standalone --nproc_per_node=gpu ddp_train.py --dir="$path/SecondRun/$name" --dataset=$dataset --model=$model --epochs=$n_epochs --lr_init=0.1 --wd=3e-4 --swa --aswa --swa_start=126 --swa_lr=0.05 --val_ratio=$v_ratio --eval_freq 1 > "$path/SecondRun/$name.log"
# 5
torchrun --standalone --nproc_per_node=gpu ddp_train.py --dir="$path/ThirdRun/$name" --dataset=$dataset --model=$model --epochs=$n_epochs --lr_init=0.1 --wd=3e-4 --swa --aswa --swa_start=126 --swa_lr=0.05 --val_ratio=$v_ratio --eval_freq 1 > "$path/ThirdRun/$name.log"




model="WideResNet28x10"
path="$dataset""$model"

mkdir -p $path/FirstRun
mkdir $path/SecondRun
mkdir $path/ThirdRun


# 5% of the training dataset will be used as a validation dataset
v_ratio=005
name="Original_$dataset""_$model""_1000E_Val_""$v_ratio""_ratio"
v_ratio=0.05
# 5%
torchrun --standalone --nproc_per_node=gpu ddp_train.py --dir="$path/FirstRun/$name" --dataset=$dataset --model=$model --epochs=$n_epochs --lr_init=0.1 --wd=5e-4 --swa --aswa --swa_start=126 --swa_lr=0.05 --val_ratio=$v_ratio --eval_freq 1 > "$path/FirstRun/$name.log" 
# 5%
torchrun --standalone --nproc_per_node=gpu ddp_train.py --dir="$path/SecondRun/$name" --dataset=$dataset --model=$model --epochs=$n_epochs --lr_init=0.1 --wd=5e-4 --swa --aswa --swa_start=126 --swa_lr=0.05 --val_ratio=$v_ratio --eval_freq 1 > "$path/SecondRun/$name.log"
# 5  
torchrun --standalone --nproc_per_node=gpu ddp_train.py --dir="$path/ThirdRun/$name" --dataset=$dataset --model=$model --epochs=$n_epochs --lr_init=0.1 --wd=5e-4 --swa --aswa --swa_start=126 --swa_lr=0.05 --val_ratio=$v_ratio --eval_freq 1 > "$path/ThirdRun/$name.log"
