import os,sys
root_path = os.getcwd()
current_path = os.path.join(root_path,'map_nav_src')
sys.path.append(root_path)
sys.path.append(current_path)
import json
import time
import numpy as np
from collections import defaultdict

import torch
from tensorboardX import SummaryWriter

from transformers import DistilBertTokenizer
import pandas as pd

from utils.misc import set_random_seed
from utils.logger import write_to_record_file, print_progress, timeSince
from utils.distributed import init_distributed, is_default_gpu
from utils.distributed import all_gather, merge_dist_results

from utils.data import ImageFeaturesDB, SpeakerTokenizer
from r2r.data_utils import construct_instrs, construct_sub_instrs
from r2r.env import R2RNavBatch
from r2r.parser import parse_args
from r2r.eval_utils import Bleu_Scorer

from models.vlnbert_init import get_tokenizer
from r2r.agent import GMapNavAgent

from r2r.PASTS import Speaker

def build_dataset(args, rank=0, is_test=False):
    # Load tokenizers for BERT and speaker
    try:
        TRAIN_VOCAB = root_path + '/r2r/train_vocab.txt'
        with open(TRAIN_VOCAB) as f:
            vocab = [word.strip() for word in f.readlines()]
    except Exception:
        TRAIN_VOCAB = root_path + '/map_nav_src/r2r/train_vocab.txt'
        with open(TRAIN_VOCAB) as f:
            vocab = [word.strip() for word in f.readlines()]
    bert_tok = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    speaker_tok = SpeakerTokenizer(vocab=vocab, encoding_length=args.maxInput)
    
    # Load dataset
    feat_db = ImageFeaturesDB(args.img_ft_file, args.image_feat_size)

    dataset_class = R2RNavBatch

    if 'speaker' in args.train:
        load_instr = construct_sub_instrs
        anno_dir = args.fg_anno_dir
    else:
        load_instr = construct_instrs
        anno_dir = args.anno_dir

    if args.aug is not None:
        aug_instr_data = load_instr(
            anno_dir, args.dataset, [args.aug], 
            tokenizer=speaker_tok, max_instr_len=args.max_instr_len,
            is_test=is_test
        )
        aug_env = dataset_class(
            feat_db, aug_instr_data, args.connectivity_dir, 
            batch_size=args.batch_size, angle_feat_size=args.angle_feat_size, 
            seed=args.seed+rank, sel_data_idxs=None, name='aug', 
            args=args
        )
    else:
        aug_env = None

    train_instr_data = load_instr(
        anno_dir, args.dataset, ['train'], 
        tokenizer=speaker_tok, max_instr_len=args.max_instr_len,
        is_test=is_test
    )
    train_env = dataset_class(
        feat_db, train_instr_data, args.connectivity_dir,
        batch_size=args.batch_size, 
        angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
        sel_data_idxs=None, name='train', args=args
    )

    # val_env_names = ['val_train_seen']
    if args.submit and args.dataset != 'r4r':
        val_env_names = ['val_seen']
    else:
        if 'speaker' in args.train:
            val_env_names = ['val_seen', 'val_unseen']
        else:
            val_env_names = ['val_train_seen', 'val_seen', 'val_unseen']
    if args.dataset == 'r4r' and (not args.test):
        val_env_names[-1] == 'val_unseen_sampled'
    
    if args.submit and args.dataset != 'r4r':
        val_env_names.append('val_unseen', 'test')
        
    val_envs = {}
    for split in val_env_names:
        val_instr_data = load_instr(
            anno_dir, args.dataset, [split], 
            tokenizer=speaker_tok, max_instr_len=args.max_instr_len,
            is_test=is_test
        )
        val_env = dataset_class(
            feat_db, val_instr_data, args.connectivity_dir, batch_size=args.batch_size, 
            angle_feat_size=args.angle_feat_size, seed=args.seed+rank,
            sel_data_idxs=None if args.world_size < 2 else (rank, args.world_size), name=split,
            args=args
        )   # evaluation using all objects
        val_envs[split] = val_env

    return train_env, val_envs, aug_env, bert_tok, speaker_tok

def train_follower(args, train_env, val_envs, aug_env=None, rank=-1, bert_tok=None, speaker_tok=None):
    default_gpu = is_default_gpu(args)

    if default_gpu:
        with open(os.path.join(args.log_dir, 'training_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        writer = SummaryWriter(log_dir=args.log_dir)
        record_file = os.path.join(args.log_dir, 'train.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    agent_class = GMapNavAgent
    listner = agent_class(args, train_env, rank=rank, tok=bert_tok)

    if args.use_speaker:
        speaker = Speaker(args,train_env,speaker_tok)
        print("Load the speaker from %s." % args.speaker_ckpt_path)
        speaker.load(args.speaker_ckpt_path)
        print("Load speaker model successully.")
    else:
        speaker = None

    # resume file
    start_iter = 0
    if args.resume_file is not None:
        start_iter = listner.load(os.path.join(args.resume_file))
        if default_gpu:
            write_to_record_file(
                "\nLOAD the model from {}, iteration ".format(args.resume_file, start_iter),
                record_file
            )
       
    # first evaluation
    if args.eval_first:
        loss_str = "validation before training"
        for env_name, env in val_envs.items():
            listner.env = env
            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            preds = listner.get_results()
            # gather distributed results
            preds = merge_dist_results(all_gather(preds))
            if default_gpu:
                score_summary, _ = env.eval_metrics(preds)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
        if default_gpu:
            write_to_record_file(loss_str, record_file)
        # return

    start = time.time()
    if default_gpu:
        write_to_record_file(
            '\nListener training starts, start iteration: %s' % str(start_iter), record_file
        )

    best_val = {'val_unseen': {"spl": 0., "sr": 0., "state":""}}
    if args.dataset == 'r4r':
        best_val = {'val_unseen_sampled': {"spl": 0., "sr": 0., "state":""}}
    
    for idx in range(start_iter, start_iter+args.iters, args.log_every):
        listner.logs = defaultdict(list)
        interval = min(args.log_every, args.iters-idx)
        iter = idx + interval

        # Train for log_every interval
        if aug_env is None:
            listner.env = train_env
            listner.train(interval, feedback=args.feedback)  # Train interval iters
        else:
            if args.accumulate_grad:
                jdx_length = len(range(interval // 2))
                for jdx in range(interval // 2):
                    listner.zero_grad()
                    # Train with GT data
                    listner.env = train_env
                    listner.accumulate_gradient(args.feedback)
                    listner.env = aug_env

                    # Train with Augmented data
                    listner.accumulate_gradient(args.feedback, speaker=speaker)
                    listner.optim_step()

                    if default_gpu:
                        print_progress(jdx, jdx_length, prefix='Progress:', suffix='Complete', bar_length=50)
            else:
                jdx_length = len(range(interval // 2))
                for jdx in range(interval // 2):
                    # Train with GT data
                    listner.env = train_env
                    listner.train(1, feedback=args.feedback)

                    # Train with Augmented data
                    listner.env = aug_env
                    listner.train(1, feedback=args.feedback)

                    if default_gpu:
                        print_progress(jdx, jdx_length, prefix='Progress:', suffix='Complete', bar_length=50)

        if default_gpu:
            # Log the training stats to tensorboard
            total = max(sum(listner.logs['total']), 1)          # RL: total valid actions for all examples in the batch
            length = max(len(listner.logs['critic_loss']), 1)   # RL: total (max length) in the batch
            critic_loss = sum(listner.logs['critic_loss']) / total
            policy_loss = sum(listner.logs['policy_loss']) / total
            RL_loss = sum(listner.logs['RL_loss']) / max(len(listner.logs['RL_loss']), 1)
            IL_loss = sum(listner.logs['IL_loss']) / max(len(listner.logs['IL_loss']), 1)
            entropy = sum(listner.logs['entropy']) / total
            writer.add_scalar("loss/critic", critic_loss, idx)
            writer.add_scalar("policy_entropy", entropy, idx)
            writer.add_scalar("loss/RL_loss", RL_loss, idx)
            writer.add_scalar("loss/IL_loss", IL_loss, idx)
            writer.add_scalar("total_actions", total, idx)
            writer.add_scalar("max_length", length, idx)
            write_to_record_file(
                "\ntotal_actions %d, max_length %d, entropy %.4f, IL_loss %.4f, RL_loss %.4f, policy_loss %.4f, critic_loss %.4f" % (
                    total, length, entropy, IL_loss, RL_loss, policy_loss, critic_loss),
                record_file
            )

        # Run validation
        loss_str = "iter {}".format(iter)
        for env_name, env in val_envs.items():
            listner.env = env

            # Get validation distance from goal under test evaluation conditions
            listner.test(use_dropout=False, feedback='argmax', iters=None)
            preds = listner.get_results()
            preds = merge_dist_results(all_gather(preds))

            if default_gpu:
                score_summary, _ = env.eval_metrics(preds)
                loss_str += ", %s " % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                    writer.add_scalar('%s/%s' % (metric, env_name), score_summary[metric], idx)

                # select model by spl
                if env_name in best_val:
                    if score_summary['spl'] >= best_val[env_name]['spl']:
                        best_val[env_name]['spl'] = score_summary['spl']
                        best_val[env_name]['sr'] = score_summary['sr']
                        best_val[env_name]['state'] = 'Iter %d %s' % (iter, loss_str)
                        listner.save(idx, os.path.join(args.ckpt_dir, "best_%s" % (env_name)))
                
        
        if default_gpu:
            listner.save(idx, os.path.join(args.ckpt_dir, "latest_dict"))

            write_to_record_file(
                ('%s (%d %d%%) %s' % (timeSince(start, float(iter)/args.iters), iter, float(iter)/args.iters*100, loss_str)),
                record_file
            )
            write_to_record_file("BEST RESULT TILL NOW", record_file)
            for env_name in best_val:
                write_to_record_file(env_name + ' | ' + best_val[env_name]['state'], record_file)


def valid_follower(args, train_env, val_envs, rank=-1):
    default_gpu = is_default_gpu(args)

    agent_class = GMapNavAgent
    agent = agent_class(args, train_env, rank=rank)

    if args.resume_file is not None:
        print("Loaded the listener model at iter %d from %s" % (
            agent.load(args.resume_file), args.resume_file))

    if default_gpu:
        with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(args.log_dir, 'valid.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    for env_name, env in val_envs.items():
        prefix = 'submit' if args.detailed_output is False else 'detail'
        if os.path.exists(os.path.join(args.pred_dir, "%s_%s.json" % (prefix, env_name))):
            continue
        agent.logs = defaultdict(list)
        agent.env = env

        iters = None
        start_time = time.time()
        agent.test(
            use_dropout=False, feedback='argmax', iters=iters)
        print(env_name, 'cost time: %.2fs' % (time.time() - start_time))
        preds = agent.get_results(detailed_output=args.detailed_output)
        preds = merge_dist_results(all_gather(preds))

        if default_gpu:
            if 'test' not in env_name:
                score_summary, _ = env.eval_metrics(preds)
                loss_str = "Env name: %s" % env_name
                for metric, val in score_summary.items():
                    loss_str += ', %s: %.2f' % (metric, val)
                write_to_record_file(loss_str+'\n', record_file)

            if args.submit:
                json.dump(
                    preds,
                    open(os.path.join(args.pred_dir, "%s_%s.json" % (prefix, env_name)), 'w'),
                    sort_keys=True, indent=4, separators=(',', ': ')
                )


def train_speaker(args, train_env, val_envs, speaker_tok=None):
    default_gpu = is_default_gpu(args)

    if default_gpu:
        with open(os.path.join(args.log_dir, 'training_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        writer = SummaryWriter(log_dir=args.log_dir)
        record_file = os.path.join(args.log_dir, 'train.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    speaker = Speaker(args, train_env, speaker_tok)
    bleu_scorer = Bleu_Scorer()
    best_bleu = defaultdict(lambda: 0)
    best_loss = defaultdict(lambda: 1232)
    best_both_bleu = 0

    # resume file
    start_iter = 0
    if args.resume_file is not None:
        start_iter = speaker.load(os.path.join(args.resume_file))
        if default_gpu:
            write_to_record_file(
                "\nLOAD the model from {}, iteration ".format(args.resume_file, start_iter),
                record_file
            )

    if default_gpu:
        write_to_record_file(
            '\nSpeaker training starts, start iteration: %s' % str(start_iter), record_file
        )

    for idx in range(start_iter, start_iter+args.iters, args.log_every):
        interval = min(args.log_every, args.iters - idx)
        iter = idx + interval

        # Train for log_every interval
        speaker.env = train_env
        train_loss, progress_loss = speaker.train(interval)   # Train interval iters
        writer.add_scalar("loss/train_loss",train_loss,idx)
        writer.add_scalar("loss/train_progress_loss",progress_loss,idx)

        write_to_record_file("\n****** Iter: %d" % idx,record_file)
        write_to_record_file("Train loss: %.2f" %train_loss, record_file)

        # Evaluation
        current_valseen_bleu4 = 0
        current_valunseen_bleu4 = 0
        best_data_seen, best_data_unseen = [], []
        for env_name, env in val_envs.items():
            write_to_record_file("............ Evaluating %s ............." % env_name, record_file)
            speaker.env = env
            path2inst, loss, word_accu, sent_accu = speaker.valid()
            write_to_record_file(f"Evaluation loss:{loss}",record_file)
            path_id = list(path2inst.keys())[0]
            print("Inference: ", speaker_tok.decode_sentence(path2inst[path_id]['inst']))
            print("GT: ", env.gt_insts[path_id])

            data = []
            # Compute BLEU
            for path_id in path2inst.keys():
                inference_sentences = speaker_tok.decode_sentence(path2inst[path_id]['inst'])
                gt_sentences = env.gt_insts[path_id]
                temp_data = {
                    "path_id":path_id,
                    "Inference": [inference_sentences],
                    "Ground Truth":gt_sentences,
                }
                data.append(temp_data)
            if env_name == 'val_seen':
                best_data_seen = data
            elif env_name == 'val_unseen':
                best_data_unseen = data
            precisions = bleu_scorer.compute_scores(data)
            bleu_score = precisions[3] # bleu4
            
            if env_name == 'val_seen':
                current_valseen_bleu4 = bleu_score
            elif env_name == 'val_unseen':
                current_valunseen_bleu4 = bleu_score
                if current_valunseen_bleu4 + current_valseen_bleu4 >= best_both_bleu:
                    best_both_bleu = current_valunseen_bleu4 + current_valseen_bleu4
                    write_to_record_file('Save the model with val_seen BEST env bleu %0.4f and val_unseen BEST env bleu %0.4f' % (current_valseen_bleu4,current_valunseen_bleu4),record_file)
                    speaker.save(idx, os.path.join(args.ckpt_dir, 'best_both_bleu'))

                    # Save the prediction results
                    pd_data_seen = pd.DataFrame(best_data_seen)
                    pd_data_unseen = pd.DataFrame(best_data_unseen)
                    pd_data_seen.to_csv(os.path.join(args.pred_dir,"{}_val_seen.csv".format(args.name)))
                    pd_data_unseen.to_csv(os.path.join(args.pred_dir,"{}_val_unseen.csv".format(args.name)))
                    write_to_record_file('save the result to {}'.format(args.pred_dir),record_file)

            # Tensorboard log
            writer.add_scalar("bleu/%s" % (env_name), precisions[0], idx)
            writer.add_scalar("loss/%s" % (env_name), loss, idx)
            writer.add_scalar("word_accu/%s" % (env_name), word_accu, idx)
            writer.add_scalar("sent_accu/%s" % (env_name), sent_accu, idx)
            writer.add_scalar("bleu4/%s" % (env_name), precisions[3], idx)

            if bleu_score > best_bleu[env_name]:
                best_bleu[env_name] = bleu_score
                write_to_record_file('Save the model with %s BEST env bleu %0.4f' % (env_name, bleu_score),record_file)
                if env_name == 'val_unseen':
                    speaker.save(idx, os.path.join(args.ckpt_dir, 'best_unseen_bleu'))

            if loss < best_loss[env_name]:
                best_loss[env_name] = loss
                write_to_record_file('Save the model with %s BEST env loss %0.4f' % (env_name, loss),record_file)
            
            # Screen print out
            write_to_record_file("Bleu 1: %0.4f Bleu 2: %0.4f, Bleu 3 :%0.4f,  Bleu 4: %0.4f" % tuple(precisions[:4]),record_file)
            if len(best_bleu)!=0:
                write_to_record_file("Best bleu: %0.4f, Best loss: %0.4f" % (best_bleu[env_name],best_loss[env_name]),record_file)

def valid_speaker(args, val_envs, speaker_tok, write_result=True, compute_coco=False):
    import tqdm
    default_gpu = is_default_gpu(args)
    speaker = Speaker(args, val_envs, speaker_tok)
    speaker.model.eval()
    speaker.load(args.speaker_ckpt_path)
    bleu_scorer = Bleu_Scorer()
    
    if default_gpu:
        with open(os.path.join(args.log_dir, 'validation_args.json'), 'w') as outf:
            json.dump(vars(args), outf, indent=4)
        record_file = os.path.join(args.log_dir, 'validation.txt')
        write_to_record_file(str(args) + '\n\n', record_file)

    if compute_coco:
        # need pycocoevalcap
        from r2r.eval_utils import All_Scorer

    # Evaluation
    for env_name, env in val_envs.items():
        if env_name == 'train':
            continue
        print("............ Evaluating %s ............." % env_name)
        save_data = []
        speaker.env = env
        path2inst, loss, word_accu, sent_accu = speaker.valid(wrapper=tqdm.tqdm)
        for path_id in path2inst.keys():
            inference_sentences = speaker_tok.decode_sentence(path2inst[path_id]['inst'])
            gt_sentences = env.gt_insts[path_id]
            temp_data = {
                "path_id":path_id,
                "Inference": [inference_sentences],
                "Ground Truth":gt_sentences,
            }
            save_data.append(temp_data)
        precisions = bleu_scorer.compute_scores(save_data) # [0,1,2,3]
        bleu_score = precisions[4]
        write_to_record_file('Bleu:{}\tBleu_1:{}\tBleu_4:{}'.format(bleu_score,precisions[0],precisions[3]), record_file)

        if write_result:
            pd_data = pd.DataFrame(save_data)
            save_path = os.path.join(args.pred_dir,f'{args.name}_{env_name}_valid.csv')
            pd_data.to_csv(save_path)
            print('save the result to {}'.format(save_path))
        
        if compute_coco:
            # Calculate BLEU, ROUGE_L, SPICE, CIDEr
            inference = {}
            ground_truth = {}
            for item in save_data:
                inference[item['path_id']] = item['Inference']
                ground_truth[item['path_id']] = item['Ground Truth']
            all_scorer = All_Scorer(inference, ground_truth)
            total_scores = all_scorer.compute_scores()
            write_to_record_file('*****DONE*****', record_file)
            for key,value in total_scores.items():
                write_to_record_file('{}:{}'.format(key,value), record_file)

def main():
    args = parse_args()

    if args.world_size > 1:
        rank = init_distributed(args)
        torch.cuda.set_device(args.local_rank)
    else:
        rank = 0

    set_random_seed(args.seed + rank)
    train_env, val_envs, aug_env, bert_tok, speaker_tok = build_dataset(args, rank=rank, is_test=args.test)

    if args.train == 'follower':
        train_follower(args, train_env, val_envs, aug_env=aug_env, rank=rank, bert_tok=bert_tok, speaker_tok=speaker_tok)
    elif args.train == 'valid_follower':
        valid_follower(args, train_env, val_envs, rank=rank)
    elif args.train == 'speaker':
        train_speaker(args, train_env, val_envs, speaker_tok)
    elif args.train == 'valid_speaker':
        valid_speaker(args, val_envs, speaker_tok, compute_coco=args.compute_coco)

if __name__ == '__main__':
    main()
