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
    --seed 238 \
    --save_dir "./final/" \
    --save_model \
    --mode "train" \
    --audit_method "UBWC" \
    --audit_config_path "config/UBWC/cifar10-resnet18.yaml" \
    --attack_method ${attack}\
    --attack_config_path "config/attack/${attack}/cifar10.yaml" \
    --reprocessing