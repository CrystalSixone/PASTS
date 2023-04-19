name=20230419_r2r_speaker_resnet_validate
DATA_ROOT=../datasets

train_alg=dagger

features=resnet
# features=vitbase
# features=clip768

ngpus=1
seed=0

outdir=${DATA_ROOT}/R2R/
speaker_path=${DATA_ROOT}/R2R/speaker/20230418_r2r_speaker_resnet_lr5e-5/ckpts/best_both_bleu

flag="--root_dir ${DATA_ROOT}
      --dataset r2r
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert    
      --train valid_speaker  
      --name ${name}

      --batch_size 64

      --features ${features}
      --speaker_ckpt_path ${speaker_path}
      --use_drop
      "

# train
CUDA_VISIBLE_DEVICES='0' python -u r2r/main_nav.py $flag
      

# test
# CUDA_VISIBLE_DEVICES='4' python r2r/main_nav.py $flag  \
#       --tokenizer bert \
#       --resume_file /data/ssd0/w61/tjg_duet/datasets/R2R/navigator/20220614_r2r_ft_transpeaker_vit_original/ckpts/best_val_unseen \
#       --submit --test