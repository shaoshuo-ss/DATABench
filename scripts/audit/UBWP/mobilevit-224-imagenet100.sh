gpus=$1
attack=$2

python audit_main.py \
    --gpus ${gpus} \
    --epochs 150 \
    --lr 0.03 \
    --momentum 0.9 \
    --optim "sgd" \
    --bs 64 \
    --wd 5e-4 \
    --eval_rounds 5 \
    --test_bs 512 \
    --model "MobileViT" \
    --dataset "imagenet100" \
    --image_size 224 \
    --seed 666 \
    --save_dir "./final/" \
    --save_model \
    --mode "train" \
    --audit_method "UBWP" \
    --audit_config_path "config/UBWP/imagenet100-swinvit.yaml" \
    --attack_method ${attack}\
    --attack_config_path "config/attack/${attack}/imagenet100.yaml" \
    --reprocessing