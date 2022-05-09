# DCCI
Content-Aware Adaptive Device-Cloud Collaborative Inference for Object Detection

## Dependencies
There are some dependencies for running this
1. python >= 3.6
2. pytorch >= 1.7
3. torchvision >= 0.19


## How to compress large models
1. Load initialization weights for basic training
```azure
python train.py --cfg cfg/yolov4.cfg --data data/voc.data --weights weights/yolov4.weights --epochs 100 --batch-size 16
```
2. Based on the optimal weights trained in step 1, perform sparse training
```azure
python train.py --cfg cfg/yolov4.cfg --data data/voc.data --weights weights/best.pt --epochs 299 --batch-size 64 -sr --s 0.0001 --prune 1
```
3. Channel-level pruning using sparsely trained models
```azure
python channel_pruning.py --cfg cfg/yolov4.cfg --data data/voc.data --weights weights/best.pt --global_percent 0.9 --layer_keep 0.01
```

4. Layer-level pruning using channel-pruned models
```azure
python layer_prune.py --cfg cfg/light_weight_channel.cfg --data data/voc.data --weights weights/light_weight_channel.weight --rl_epochs 80 --batch-size 32
```
5. Fine-tune recovery accuracy
```azure
python train.py --cfg cfg/light_weight_layer.cfg --data data/voc.data --weights weights/light_weight_layer.weights --epochs 100 --batch-size 32
```
