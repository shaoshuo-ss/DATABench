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
    --audit_method "MIA" \
    --audit_config_path "config/MIA/cifar10-resnet18.yaml" \
    --attack_method ${attack}\
    --attack_config_path "config/attack/${attack}/cifar10.yaml" \
    --pre_train_path "final/noaudit/noattack/cifar10-imagefolder/ResNet18/train/2025-03-24-20:36:37/model_last_epochs_89.pth" \
    # "final/noaudit/noattack/cifar10-imagefolder/ResNet18/train/2025-03-24-20:36:37/model_last_epochs_89.pth"
    # MIA/noattack/cifar10-imagefolder/ResNet18/train/2025-04-03-13:50:20
    # MIA/autoencoder/cifar10-imagefolder/ResNet18/train/2025-04-03-16:58:29
    # MIA/advtraining/cifar10-imagefolder/ResNet18/train/2025-04-03-15:31:56 fgsm
    # MIA/advtraining/cifar10-imagefolder/ResNet18/train/2025-04-03-15:31:29 gybrid
    # MIA/dpsgd/cifar10-imagefolder/ResNet18/train/2025-04-03-19:13:27  dpsgd 32
    # MIA/dpsgd/cifar10-imagefolder/ResNet18/train/2025-04-03-20:19:54 dpsgd 64
    # MIA/guassianfilter/cifar10-imagefolder/ResNet18/train/2025-04-03-20:20:40
    # MIA/medianfilter/cifar10-imagefolder/ResNet18/train/2025-04-03-21:28:22 mdeian
    # MIA/synthesis/cifar10-imagefolder/ResNet18/train/2025-04-03-23:33:34  synthesis
    # MIA/waveletfilter/cifar10-imagefolder/ResNet18/train/2025-04-06-11:16:48
    # /data/home/Mengren/Attack-Dataset-Auditing/final/MIA/noattack/cifar10-imagefolder/ResNet18/train/2025-04-03-13:50:20/model_best.pth