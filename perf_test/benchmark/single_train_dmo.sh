#!/bin/bash

cd /home/cephagent/benchmarks/scripts/tf_cnn_benchmarks

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
num_gpus=8

DMO_SOCK_PATH='dmo.daemon.sock.0'
dmocli -socket_path ${DMO_SOCK_PATH} -action rmdir -path resnet_model_dir

#must set --data_name=imagenet to run training
/home/cephagent/anaconda3/bin/python tf_cnn_benchmarks.py \
		--enable_dmo \
		--use_fp16=True \
		--save_model_steps=200 \
		--device=gpu \
		--data_format=NCHW \
		--num_gpus=${num_gpus} \
		--batch_size=512 \
		--optimizer=momentum \
		--model=resnet50 \
		--variable_update=replicated \
		--data_dir=dmo://$DMO_SOCK_PATH/imagenet/tfrecord \
		--data_name=imagenet \
		--num_batches=500 \
		--train_dir=dmo://$DMO_SOCK_PATH/resnet_model_dir

