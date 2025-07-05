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
    --audit_method "Rapid" \
    --audit_config_path "config/Rapid/cifar10-resnet18.yaml" \
    --attack_method ${attack}\
    --attack_config_path "config/attack/${attack}/cifar10.yaml" \
    --pre_train_path "final/Rapid/noattack/cifar10-imagefolder/ResNet18/train/2025-03-30-15:11:01/model_last_epochs_89.pth"
    # final/ML_data_auditing/noattack/cifar10-imagefolder/ResNet18/train/2025-03-24-23:26:15/model_last_epochs_89.pth