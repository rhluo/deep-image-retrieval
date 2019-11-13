ENV:
python: 3.6
pytorch: 1.1.0
cuda: 9.0.176

SCRIPT:
export DB_ROOT=/home/yuxiaoming/luoronghua/datasets
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m dirtorch.extract_features --dataset "ImageList('/home/yuxiaoming/luoronghua/datasets/holiday.log')" --checkpoint model/Resnet101-TL-MAC.pt --output features/holiday_features.npy --whiten Landmarks_clean --whitenp 0.25 --gpu 0 1 2 3
