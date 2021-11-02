# TensorFlow2 AdaIN implementation 
This is an implementation of [AdaIN](https://arxiv.org/abs/1703.06868) paper in TensorFlow 2. The script contains only the training algorithm.
# How to use it.
1. Download vgg19 normalized from [here](https://github.com/elleryqueenhomels/arbitrary_style_transfer)

2. Insert the paths to wikiart and train2014 datasets or other datasets.
```python
style_path = "/path/to/wikiart"
content_path = "/path/to/train2014"
```
3. You can change the global variables which are the training parameters, if you want.
```python 
# Batch Size
BATCH_SIZE = 8 
# Image Size
IMAGE_SIZE = [256, 256, 3] 
# Learning rate
LEARNING_RATE = 1e-4
# Learning rate decay rate, we need this so
LR_DECAY_RATE = 5e-5
# Decay step
DECAY_STEPS = 1.0
# Style loss weight
STYLE_LOSS_WEIGHT = 2
```
4. After these changes you can run to execute the script and start the training.
```cmd
python3 AdaIn.py 
```

