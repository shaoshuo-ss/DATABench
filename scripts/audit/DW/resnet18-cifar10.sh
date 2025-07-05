gpus=$1
attack=$2
CUDA_VISIBLE_DEVICES=0 \
python audit_main.py \
    --gpus ${gpus} \
    --epochs 1000 \
    --lr 0.001 \
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
    --audit_method "DW" \
    --audit_config_path "config/DW/cifar10-resnet18.yaml" \
    --attack_method ${attack}\
    --attack_config_path "config/attack/${attack}/cifar10.yaml" \
    --reprocessing