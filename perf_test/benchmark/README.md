### Usage: distributed_train_dmo.sh <ps|wk> task_index

**_Note: Network bandwidth matters a lot in terms of images/sec, so use 100Gb network(10.100.1.42 instead of 'bigisland
') whenever possible!_**
______________________________________________________________________
### start 2 workers on bigisland (2 GPUs): ### 
WORKERS='bigisland:54321,bigisland:12345'

PS='bigisland:6543'

#### term 0:
```
distributed_train_dmo.sh ps  0  #2 0 not used for ps
```
#### term 1:
```
export CUDA_VISIBLE_DEVICES=0
distributed_train_dmo.sh wk  0  #2 workers, worker 0
```
#### term 2:
```
export CUDA_VISIBLE_DEVICES=1
distributed_train_dmo.sh wk  1  #2 workers, worker 1
```
______________________________________________________________________
### start one ps on bigisland + worker on bigisland + one worker on maui
 
WORKERS='bigisland:54321,maui:54321'

PS='bigisland:6543'

#### term 0:
```
bigisland#: distributed_train_dmo.sh ps  0   
```
#### term 1:
```
distributed_train_dmo.sh wk  0
```
#### term 2:
```
maui: distributed_train_dmo.sh wk  1
```

#### To resize image for resnet(say 224 -> 64), change models/resnet_model.py  
```
class ResnetModel():
...
  __init__(...)
    super(ResnetModel, self).__init__(model, 224...
to:
    super(ResnetModel, self).__init__(model, 64..
```
Note:
    - As of branch cnn_tf_v1.13_compatible functions like add_backbone_saver/load_backbone_model have Not been implemented, so options of tf_cnn_benchmarks.py like --backbone_model_path can not function!
    
Note: 
    Use bmp instead of jpeg, inference(and maybe trainning as well) is 2~3x faster. To use bmp in inference:
    1. create bmp tfrecord file (see evernote notes)
    2. make changes to benchmarks/scripts/tf_cnn_benchmarks/preprocessing.py:
        . in BaseImagePreprocessor::__init__(...) Self.train = ""
        . change tf.image.decode_jpeg() to tf.image.decode_bmp()

