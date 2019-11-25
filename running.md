ENV:
python: 3.6
pytorch: 1.1.0
cuda: 9.0.176

SCRIPT:
export DB_ROOT=/home/yuxiaoming/luoronghua/datasets
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m dirtorch.extract_features --dataset "ImageList('/home/yuxiaoming/luoronghua/datasets/mirflickr.log')" --checkpoint model/Resnet101-TL-MAC.pt --output features/mirflickr_features.npy --whiten Landmarks_clean --whitenp 0.25  --trfs "Resize(550), Resize(800), Resize(1050)" --gpu "0,1,2"

python -m dirtorch.extract_features --dataset "ImageList('/home/yuxiaoming/luoronghua/datasets/holiday.log')" --checkpoint model/Resnet101-TL-MAC.pt --output features/holiday_features.npy --whiten Landmarks_clean --whitenp 0.25 --trfs "Resize(550), Resize(800), Resize(1050)" --gpu 3
