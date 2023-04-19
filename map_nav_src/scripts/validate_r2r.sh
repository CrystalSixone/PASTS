name=DUET_bt_validate
DATA_ROOT=../datasets

train_alg=dagger

features=vitbase
ft_dim=768

obj_features=vitbase
obj_ft_dim=768

ngpus=1
seed=0


outdir=${DATA_ROOT}/R2R/
speaker_path=${DATA_ROOT}/R2R/speaker/speaker_vit/ckpts/best_both_bleu
aug_path=${DATA_ROOT}/R2R/annotations/prevalent_aug_train_enc.json
resume_file=${DATA_ROOT}/R2R/follower/finetuned/ckpts/best_val_unseen

flag="--root_dir ${DATA_ROOT}
      --dataset r2r
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert    
      --train valid_follower  
      --name ${name}

      --enc_full_graph
      --graph_sprels
      --fusion dynamic

      --expert_policy spl
      --train_alg ${train_alg}
      
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      
      --max_action_len 15
      --max_instr_len 200

      --batch_size 8
      --lr 1e-5
      --iters 200000
      --log_every 1000
      --optim adamW

      --features ${features}
      --image_feat_size ${ft_dim}
      --feature_size ${ft_dim}
      --angle_feat_size 4

      --ml_weight 0.2   

      --feat_dropout 0.4
      --dropout 0.5
      
      --gamma 0.9

      --resume_file ${resume_file}
      --submit
      "

CUDA_VISIBLE_DEVICES='0' python -u r2r/main_nav.py $flag
