gpus=$1
attack=$2

python audit_main.py \
    --gpus ${gpus} \
    --epochs 90 \
    --lr 0.03 \
    --momentum 0.9 \
    --optim "sgd" \
    --bs 128 \
    --wd 1e-3 \
    --eval_rounds 10 \
    --test_bs 512 \
    --model "ResNet18" \
    --dataset "cifar10-imagefolder" \
    --image_size 32 \
    --seed 666 \
    --save_dir "./final/" \
    --save_model \
    --mode "test" \
    --audit_method "UBWP" \
    --audit_config_path "config/UBWP/cifar10-resnet18.yaml" \
    --attack_method ${attack}\
    --attack_config_path "config/attack/${attack}/cifar10.yaml" \
    --pre_train_path "./final/UBWP/noattack/cifar10-imagefolder/ResNet18/train/2025-03-28-07:55:18/model_last_epochs_89.pth"
    # --pre_train_path "final/UBWP/noattack/cifar10-imagefolder/ResNet18/train/best/model_last_epochs_89.pth"