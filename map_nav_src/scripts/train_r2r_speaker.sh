name=20230419_r2r_speaker_vit_lr5e-5_continue
DATA_ROOT=../datasets

train_alg=dagger

# features=resnet
features=vitbase
# features=clip768

ngpus=1
seed=0

outdir=${DATA_ROOT}/R2R/
speaker_path=${DATA_ROOT}/R2R/speaker/20220612_transpeaker_duet_vit/state_dict/best_both_bleu
aug_path=${DATA_ROOT}/R2R/annotations/prevalent_aug.json
resume_file=${DATA_ROOT}/R2R/navigator/vit/model_step_340000.pt

flag="--root_dir ${DATA_ROOT}
      --dataset r2r
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert    
      --train speaker  
      --name ${name}

      --batch_size 64
      --lr 5e-5
      --iters 80000
      --log_every 500

      --features ${features}
      --use_drop

      --resume_file /home/ubuntu/mycodes/PASTS/datasets/R2R/speaker/20230418_r2r_speaker_vit_lr5e-5/ckpts/best_both_bleu
      "

# train
CUDA_VISIBLE_DEVICES='0' python -u r2r/main_nav.py $flag
      

# test
# CUDA_VISIBLE_DEVICES='4' python r2r/main_nav.py $flag  \
#       --tokenizer bert \
#       --resume_file /data/ssd0/w61/tjg_duet/datasets/R2R/navigator/20220614_r2r_ft_transpeaker_vit_original/ckpts/best_val_unseen \
#       --submit --test