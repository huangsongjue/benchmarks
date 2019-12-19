#!/bin/bash

cd ../../scripts/tf_cnn_benchmarks

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
num_gpus=8

if [ $# != 1 ]; then
	echo "usage: single_inference_dmo.sh <dmo|base>"
	exit -1
fi


if [ $1 == 'base' ]; then
	data_dir='/mnt/data/tfrecord'
	enable_dmo=False
elif [ $1 == 'dmo' ]; then
	DMO_SOCK_PATH='dmo.daemon.sock.0'
	data_dir=dmo://$DMO_SOCK_PATH/imagenet/tfrecord
	enable_dmo=True
fi
	 
# (resnet50/resnet101/resnet152/resnet50_v1.5/resnet101_v1.5/resnet152_v1.5/resnet50_v2/resnet101_v2/resnet152_v2) see models/resnet_model.py 
#must set --data_name=imagenet to run training
time -p /home/cephagent/anaconda3/bin/python  tf_cnn_benchmarks.py \
		--enable_dmo=$enable_dmo \
		--optimizer=momentum \
		--data_dir=$data_dir \
		--data_name=imagenet \
		--forward_only=true \
		--num_gpus=${num_gpus} \
		--batch_size=2048 \
		--num_batches=200 \
		--model=resnet50 \
		--use_fp16=true  \
		--fp16_vars=True \
		--trt_mode=INT8 \
		--freeze_when_forward_only=True
		#--num_epochs=10 
		#--backbone_model_path
		#--allow_growth=True \

		


#--forward_only=True --batch_size=${BATCH_SIZE} --model=${MODEL} --num_epochs=10 --optimizer=momentum --distortions=True --display_every 10 -- num_gpus=${NUM_GPUS} --data_dir=./test_data/fake_tf_record_data/ --data_name=imagenet
