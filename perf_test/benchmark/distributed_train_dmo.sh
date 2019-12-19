#!/bin/bash

echo "usage: distributed_train_dmo.sh <ps|wk> num_tasks task_index"
echo "***************"

if [ $1 == 'ps' ]; then
  JOB_NAME='ps'
#  PARAM_DEV='cpu'
  PARAM_DEV='cpu'
elif [ $1 == 'wk' ]; then
  JOB_NAME='worker'
  PARAM_DEV='gpu'
else
  echo "Error: job_name must be <ps|wk>"
  exit -1
fi

MODEL='resnet50'
SAVE_MODEL_STEPS=200
#WORKERS='bigisland:5432,maui:5432'    #the first one will be chief
#WORKERS='asgnode029:54321,asgnode029:12345,asgnode019:54321'
#WORKERS='asgnode029:54321,asgnode019:54321'
#PS='asgnode029:6543'
WORKERS='10.50.0.32:54321,10.50.0.19:54321'
PS='10.50.0.32:6543'

cd /home/cephagent/benchmarks/scripts/tf_cnn_benchmarks

TASK_INDEX=$3

if [ ${JOB_NAME} == 'ps' ]; then
   export CUDA_VISIBLE_DEVICES=''
   TASK_INDEX=0
fi

num_gpus=8
# --sync_on_finish: Enable/disable whether the devices are synced after each step.
# --use_fp16

# --num_shards, --shard_idx

DMO_SOCK_PATH='dmo.daemon.sock.0'
if [ $JOB_NAME == 'ps' ]; then
    dmocli -socket_path ${DMO_SOCK_PATH} -action rmdir -path resnet50_model_dir
fi

/home/cephagent/anaconda3/bin/python tf_cnn_benchmarks.py \
		--enable_dmo \
                --use_fp16=True \
                --fp16_vars=True \
		--task_index=${TASK_INDEX} \
		--job_name=${JOB_NAME} \
                --ps_hosts=${PS}\
                --worker_hosts=${WORKERS} \
		--local_parameter_device=${PARAM_DEV} \
		--save_model_steps=${SAVE_MODEL_STEPS} \
		--num_gpus=${num_gpus} \
		--batch_size=0 \
		--model=${MODEL} \
		--variable_update=distributed_replicated \
		--data_dir=dmo://$DMO_SOCK_PATH/imagenet/tfrecord \
		--data_name=imagenet \
		--num_batches=1000 \
		--train_dir=dmo://$DMO_SOCK_PATH/${MODEL}_model_dir
		#--train_dir=/tmp/${MODEL}_model_dir

