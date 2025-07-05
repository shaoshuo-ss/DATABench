gpus=$1
attack=$2


python audit_main.py \
    --gpus ${gpus} \
    --epochs 120 \
    --lr 0.03 \
    --momentum 0.9 \
    --optim "sgd" \
    --bs 512 \
    --wd 0.01 \
    --eval_rounds 10 \
    --test_bs 512 \
    --model "ResNet18" \
    --dataset "imagenet100" \
    --image_size 224 \
    --seed 666 \
    --save_dir "./final/" \
    --save_model \
    --mode "train" \
    --audit_method "UBWP" \
    --audit_config_path "config/UBWP/imagenet100-resnet18.yaml" \
    --attack_method ${attack}\
    --attack_config_path "config/attack/${attack}/imagenet100.yaml" \
    --reprocessing