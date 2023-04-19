name=speaker_vit_validate
DATA_ROOT=../datasets

train_alg=dagger

features=vitbase

ngpus=1
seed=0

outdir=${DATA_ROOT}/R2R/
speaker_path=${DATA_ROOT}/R2R/speaker/speaker_vit/ckpts/best_both_bleu

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
      "

CUDA_VISIBLE_DEVICES='0' python -u r2r/main_nav.py $flag
