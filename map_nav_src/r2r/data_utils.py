import os
import json
import numpy as np
import ast

def load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=True, use_fg=False):
    data = []
    for split in splits:
        if "/" not in split:    # the official splits
            if use_fg:
                filepath = os.path.join(anno_dir, 'FGR2R_%s.json' % (split))
            else:
                filepath = os.path.join(anno_dir, '%s_%s_enc.json' % (dataset.upper(), split))
            with open(filepath) as f:
                new_data = json.load(f)

            if split == 'val_train_seen':
                new_data = new_data[:50]

            if not is_test:
                if dataset == 'r4r' and split == 'val_unseen':
                    ridxs = np.random.permutation(len(new_data))[:200]
                    new_data = [new_data[ridx] for ridx in ridxs]
        else:   # augmented data
            print('\nLoading augmented data %s ...' % os.path.basename(split))
            with open(split) as f:
                new_data = json.load(f)
        # Join
        data += new_data
    return data

def construct_instrs(anno_dir, dataset, splits, tokenizer, max_instr_len=512, is_test=True):
    data = []
    for i, item in enumerate(load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=is_test)):
        # Split multiple instructions into separate entries
        for j, instr in enumerate(item['instructions']):
            new_item = dict(item)
            new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
            new_item['instruction'] = instr
            new_item['instr_encoding'] = item['instr_encodings'][j][:max_instr_len]
            del new_item['instructions']
            del new_item['instr_encodings']
            data.append(new_item)
    return data

def construct_sub_instrs(anno_dir, dataset, splits, tokenizer, max_instr_len=512, is_test=True):
    data = []
    for i, item in enumerate(load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=is_test, use_fg=True)):
        # Split multiple instructions into separate entries
        for j, instr in enumerate(item['instructions']):
            new_item = dict(item)
            new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
            new_item['instruction'] = instr
            new_item['instr_encoding'] = tokenizer.encode_sentence(instr)
            del new_item['instructions']

            if j < len(ast.literal_eval(item['new_instructions'])): # some sub instructions are shorter than gt instr.
                # add punctuation and EOS into the sub instructions
                sub_instr = ast.literal_eval(item['new_instructions'])[j]
                sub_chunk_view = item['chunk_view'][j]
                speaker_progress = [-1] 
                total_count = 0
                for i,sub_instr_item in enumerate(sub_instr):
                    progress = float((sub_chunk_view[i][-1]) / sub_chunk_view[-1][-1])
                    for word in sub_instr_item:
                        speaker_progress.append(progress)
                        total_count += 1

                # speaker_progress.append(1.0) # EOS
                speaker_progress.extend([1.0,1.0,1.0]) # Multiple EOS for buffering
                progress_end_mask = [0]*(max_instr_len - len(speaker_progress)) 
                speaker_progress = speaker_progress + progress_end_mask
                new_item['sub_instr'] = sub_instr
                new_item['sub_chunk_view'] = sub_chunk_view
                new_item['speaker_progress'] = speaker_progress
            else:
                continue 

            data.append(new_item)
    return data