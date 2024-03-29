
## CIFAR-10
#-a densenet --depth 13 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint checkpoints/cifar10/densenet-bc-14-12
#### AlexNet
```
python cifar.py -a alexnet --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/alexnet 
```


#### VGG19 (BN)
```
python cifar.py -a vgg11 --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar10/vgg11
```
#### ResNet-20
```
python cifar.py -a resnet --depth 20 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-20 

```
#### ResNet-110
```
python cifar.py -a resnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-110 
```

#### ResNet-1202
```
python cifar.py -a resnet --depth 1202 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/resnet-1202 
```

#### PreResNet-110
```
python cifar.py -a preresnet --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar10/preresnet-110 
```

#### ResNeXt-29, 8x64d
```
python cifar.py -a resnext --depth 29 --cardinality 8 --widen-factor 4 --schedule 150 225 --wd 5e-4 --gamma 0.1 --checkpoint checkpoints/cifar10/resnext-8x64d 
```
#### ResNeXt-29, 16x64d
```
python cifar.py -a resnext --depth 29 --cardinality 16 --widen-factor 4 --schedule 150 225 --wd 5e-4 --gamma 0.1 --checkpoint checkpoints/cifar10/resnext-16x64d 
```

#### WRN-28-10-drop
```
python cifar.py -a wrn --depth 28 --depth 28 --widen-factor 10 --drop 0.3 --epochs 200 --schedule 60 120 160 --wd 5e-4 --gamma 0.2 --checkpoint checkpoints/cifar10/WRN-28-10-drop
```

#### DenseNet-BC (L=100, k=12)
**Note**: 
* DenseNet use weight decay value `1e-4`. Larger weight decay (`5e-4`) if harmful for the accuracy (95.46 vs. 94.05) 
* Official batch size is 64. But there is no big difference using batchsize 64 or 128 (95.46 vs 95.11).

```
python cifar.py -a densenet --depth 10 --growthRate 12 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint checkpoints/cifar10/densenet-bc-14-12
```

#### DenseNet-BC (L=190, k=40) 
```
python cifar.py -a densenet --depth 190 --growthRate 40 --train-batch 64 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint checkpoints/cifar10/densenet-bc-L190-k40
```

