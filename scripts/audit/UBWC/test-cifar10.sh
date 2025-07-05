gpus=$1
attack=$2

python audit_main.py \
    --gpus ${gpus} \
    --epochs 60 \
    --lr 0.05 \
    --momentum 0.9 \
    --optim "sgd" \
    --bs 128 \
    --wd 1e-3 \
    --eval_rounds 10 \
    --test_bs 128 \
    --model "ResNet18" \
    --dataset "cifar10-imagefolder" \
    --image_size 32 \
    --seed 235 \
    --save_dir "./final/" \
    --save_model \
    --mode "test" \
    --audit_method "UBWC" \
    --audit_config_path "config/UBWC/cifar10-resnet18.yaml" \
    --attack_method ${attack}\
    --attack_config_path "config/attack/${attack}/cifar10.yaml" \
    --pre_train_path "./final/UBWC/noattack/cifar10-imagefolder/ResNet18/train/2025-03-29-14:39:18/model_last_epochs_59.pth"
    # --pre_train_path "final/DVBW/noattack/cifar10-imagefolder/ResNet18/train/best/model_last_epochs_89.pth"