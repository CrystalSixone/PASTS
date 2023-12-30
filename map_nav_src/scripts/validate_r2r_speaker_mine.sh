name=speaker_clip768_validate
DATA_ROOT=../datasets

train_alg=dagger

features=clip768

ngpus=1
seed=0

outdir=${DATA_ROOT}/R2R/
speaker_path=${DATA_ROOT}/R2R/speaker/20231226_speaker_vit_clip768/ckpts/best_both_bleu

flag="--root_dir ${DATA_ROOT}
      --dataset r2r
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert    
      --train valid_speaker  
      --name ${name}

      --batch_size 64
      --dropout 0
      --speaker_dropout 0.2
      --featdropout 0

      --features ${features}
      --speaker_ckpt_path ${speaker_path}
      --compute_coco
      "

CUDA_VISIBLE_DEVICES='0' python -u r2r/main_nav.py $flag
