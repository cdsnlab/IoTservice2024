# ViT + TENT Aug
python main.py -c ta --model=resnet50_bn_torch --method=tentaug --exp_name=

# ResNet50 + DeYO Aug
python main.py -c ta --model=resnet50_bn_torch --method=deyo_aug --exp_name=

# ViT + TENT Aug
python main.py -c ta --model=vitbase_timm --method=tentaug --exp_name=

# ViT + DeYO Aug
python main.py -c ta --model=vitbase_timm --method=deyo_aug --exp_name=

python main.py -c ta --model=resnet50_gn_timm --method=deyo_aug --exp_name=