# ConvNeXt variant (TAFE-Net*):
# CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh configs/tafenet/tafenet_convnext_rtm.py 4
# SegFormer variant (TAFE-Net):
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh configs/tafenet/tafenet_segformer_rtm.py 4
