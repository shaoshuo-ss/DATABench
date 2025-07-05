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
    --test_bs 64 \
    --model "MobileViT" \
    --dataset "imagenet100" \
    --image_size 224 \
    --seed 666 \
    --save_dir "./results/" \
    --save_model \
    --mode "test" \
    --audit_method "MIA" \
    --audit_config_path "config/MIA/imagenet100-mobilevit.yaml" \
    --attack_method ${attack} \
    --attack_config_path "config/attack/${attack}/imagenet100.yaml" \
    --pre_train_path "results/MIA/advtraining/imagenet100/MobileViT/train/2025-04-11-13:30:48/model_best.pth" \
    # results/MIA/noattack/imagenet100/MobileViT/train/2025-04-06-15:36:17/model_best.pth
    # /data/home/Mengren/Attack-Dataset-Auditing/results/MIA/autoencoder/imagenet100/MobileViT/train/2025-04-07-17:53:33/model_best.pth
    # adv hybrid /data/home/Mengren/Attack-Dataset-Auditing/results/MIA/advtraining/imagenet100/MobileViT/train/2025-04-11-13:29:00/model_best.pth
    # adv fgsm /data/home/Mengren/Attack-Dataset-Auditing/results/MIA/advtraining/imagenet100/MobileViT/train/2025-04-11-13:30:48/model_best.pth
    # autoencoder /data/home/Mengren/Attack-Dataset-Auditing/results/MIA/autoencoder/imagenet100/MobileViT/train/2025-04-10-11:12:25/model_best.pth
    # /data/home/Mengren/Attack-Dataset-Auditing/results/MIA/noattack/imagenet100/MobileViT/train/2025-04-06-15:36:17/model_last_epochs_149.pth