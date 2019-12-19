### usage: distributed_train_dmo.sh <ps|wk> task_index

Note: Network bandwidth matters a lot in terms of images/sec, so use 100Gb network(10.100.1.42 instead of 'bigisland
') if possible!
______________________________________________________________________
### start 2 workers on bigisland (2 GPUs): ### 
WORKERS='bigisland:54321,bigisland:12345'
PS='bigisland:6543'

#term 0
distributed_train_dmo.sh ps  0  #2 0 not used for ps

#term 1
export CUDA_VISIBLE_DEVICES=0
distributed_train_dmo.sh wk  0  #2 workers, worker 0

#term 2
export CUDA_VISIBLE_DEVICES=1
distributed_train_dmo.sh wk  1  #2 workers, worker 1
______________________________________________________________________
### start one ps on bigisland + worker on bigisland + one worker on maui
WORKERS='bigisland:54321,maui:54321'
PS='bigisland:6543'

#term 0
bigisland#: distributed_train_dmo.sh ps  0   

#term 1
distributed_train_dmo.sh wk  0

#term 2
maui: distributed_train_dmo.sh wk  1
