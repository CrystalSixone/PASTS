name=20231229_speaker_clip768_debugScaledDotProduct
DATA_ROOT=../datasets

train_alg=dagger

features=clip768

ngpus=1
seed=0

outdir=${DATA_ROOT}/R2R/

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
      --use_drop

      --features ${features}
      "

# train
CUDA_VISIBLE_DEVICES='4' python -u r2r/main_nav.py $flag