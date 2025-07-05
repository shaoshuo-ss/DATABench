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
    --eval_rounds 10 \
    --test_bs 512 \
    --model "MobileViT" \
    --dataset "imagenet100" \
    --image_size 224 \
    --seed 666 \
    --save_dir "./results/" \
    --save_model \
    --mode "test" \
    --audit_method "Zmark" \
    --audit_config_path "config/Zmark/imagenet100-mobilevit.yaml" \
    --attack_method ${attack} \
    --attack_config_path "config/attack/${attack}/imagenet100.yaml" \
    --pre_train_path "results/Zmark/noattack/imagenet100/MobileViT/train/2025-04-09-17:21:57/model_best.pth"
    # /data/home/Mengren/Attack-Dataset-Auditing/results/Zmark/noattack/imagenet100/MobileViT/train/2025-04-08-00:34:01/model_best.pth model_last_epochs_149.pth
    # /data/home/Mengren/Attack-Dataset-Auditing/results/Zmark/noattack/imagenet100/MobileViT/train/2025-04-09-17:21:57/model_best.pth
    # /data/home/Mengren/Attack-Dataset-Auditing/results/Zmark/noattack/imagenet100/MobileViT/train/2025-04-09-17:21:57/model_best.pth