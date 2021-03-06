# TensorFlow benchmarks
This repository contains various TensorFlow benchmarks. Currently, it consists of two projects:

1. [scripts/tf_cnn_benchmarks](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks): The TensorFlow CNN benchmarks contain benchmarks for several convolutional neural networks.
2. [scripts/keras_benchmarks](https://github.com/tensorflow/benchmarks/tree/master/scripts/keras_benchmarks): The Keras benchmarks contain benchmarks for several models using Keras. Note this project is deprecated and unmaintained.

## 3.  NOTES: all code changes is on branch cnn_tf_v1.10_compatible as current dmo driver now work with tf later than v1.10.1

### code changes made to support dmo 

### model could be resnet50, inception3, vgg16, and alexnet 

### must set --data_name=imagenet to run training

### Bugs in branch cnn_tf_v1.10_compatible:
        no --save_model_steps and --save_model_secs doesn't work!

### New flags added in tensorflow/benchmarks/scripts/tf_cnn_benchmarks/benchmark_cnn.py, marked as 'hsj'
    --save_model_steps  #if == 0, skip checkpointing, used with 'forward_only' to do inference.
    --enable_dmo

### Run bigisland_run_bench_hdfs.sh as user 'dmo'

### distributed training: benchmark_cnn.py:      # First worker will be 'chief'
### distributed training: benchmark_cnn.py:      # Description of "The method for managing variables": parameter_server, replicated, ...
    - replicated, independent only apply to local mode
    - hdfs results w/ bigisland:1 + maui:0 (images/sec)
        .parameter_server:          215/s/node, 430/s total
        .distributed_replicated:    280/s/node, 560/s total
        .distributed_all_reduce:    340/s total
        .horovod:                   362/s total

### BUG? for horovod must set HOROVOD_DEVICE='cpu', if 'gpu', got msg: "NCCL INFO NET : Using interface docker0:172.17.0.1<0>", while interfaces like docker0 is not routed, hence hangs. https://github.com/uber/horovod/blob/master/docs/running.md

### --batch_size=0 : let tf decide the batch_size to use

### --cross_replica_sync used to control syn/async training https://github.com/tensorflow/benchmarks/issues/29, where True (sync) is the default.
   
###  cross_replica_sync must be True if --variable_update = cross_replica_sync/cross_replica_sync/cross_replica_sync (line ~1290, benchmark_cnn.py)

### --sync_on_finish: Enable/disable whether the devices are synced after each step. (not used in test)

-----------------------------------------------------------------------------------
### distributed_single.sh is according to https://software.intel.com/en-us/articles/boosting-deep-learning-training-inference-performance-on-xeon-and-xeon-phi.  However workers spend a very long time at 'Running warm up'. It's nothing comparable to GPU training at all!  2018/10//19

