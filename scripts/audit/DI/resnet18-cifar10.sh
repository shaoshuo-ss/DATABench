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
    --mode "train" \
    --audit_method "DI" \
    --audit_config_path "config/DI/cifar10-resnet18.yaml" \
    --attack_method ${attack}\
    --attack_config_path "config/attack/${attack}/cifar10.yaml"