import os
import torch
import torch.nn as nn
import numpy as np
from r2r.parser import parse_args
args = parse_args()

import r2r.speaker_utils as utils

import models.PASTS_model as speaker_model 
from utils.logger import print_progress

import torch.nn.functional as F
from torch.autograd import Variable

class Speaker():
    env_actions = {
      'left': ([0.0],[-1.0], [0.0]), # left
      'right': ([0], [1], [0]), # right
      'up': ([0], [0], [1]), # up
      'down': ([0], [0],[-1]), # down
      'forward': ([1], [0], [0]), # forward
      '<end>': ([0], [0], [0]), # <end>
      '<start>': ([0], [0], [0]), # <start>
      '<ignore>': ([0], [0], [0])  # <ignore>
    }

    def __init__(self, args, env, tok):
        self.args = args
        self.env = env
        self.tok = tok
        self.feature_size = args.image_feat_size
        self.tok.finalize()

        # Initialize the model
        self.model = speaker_model.PASTS(
            feature_size=self.feature_size+args.speaker_angle_size,
            hidden_size=args.h_dim,
            word_size=args.wemb, 
            tgt_vocab_size=self.tok.vocab_size()
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=args.lr) 

        # Evaluation
        self.softmax_loss = torch.nn.CrossEntropyLoss(ignore_index=self.tok.word_to_index['<PAD>'])
        self.valid_loss = torch.nn.CrossEntropyLoss(ignore_index=self.tok.word_to_index['<PAD>'],reduction='none')
        self.angle_loss = torch.nn.MSELoss()
        self.progress_loss = torch.nn.MSELoss()

    def train(self, iters):
        losses = 0
        for i in range(iters):
            self.env.reset()
            self.optimizer.zero_grad()                
            
            loss, progress_loss = self.teacher_forcing(train=True)

            losses += loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 20.)
            self.optimizer.step()

            print_progress(i, iters, prefix='Progress:', suffix='Complete', bar_length=50)

        return losses, progress_loss

    def get_insts(self, wrapper=(lambda x: x)):
        # Get the caption for all the data
        self.env.reset_epoch(shuffle=True) 
        path2inst = {}
        total = self.env.size()

        noise=None
        for _ in wrapper(range(total // self.env.batch_size + 1)):  # Guarantee that all the data are processed
            obs = self.env.reset()
            insts = self.infer_batch(featdropmask=noise)

            path_ids = [ob['path_id'] for ob in obs]  # Gather the path ids
            for path_id, inst in zip(path_ids, insts):
                if path_id not in path2inst:
                    path2inst[path_id] = {}
                    path2inst[path_id]['inst'] = self.tok.shrink(inst)  # Shrink the words

        return path2inst

    def valid(self, *aargs, **kwargs):
        path2inst = self.get_insts(*aargs, **kwargs) # record the whole val-seen/val-unseen dataset

        # Calculate the teacher-forcing metrics
        self.env.reset_epoch(shuffle=True)
        N = 3     # Set the iter to 1 if the fast_train (o.w. the problem occurs)
        metrics = np.zeros(3)
        for i in range(N):
            self.env.reset()
            metrics += np.array(self.teacher_forcing(train=False))
        metrics /= N
        # metrics: [loss, word_accu, sent_accu]

        return (path2inst, *metrics)

    def make_equiv_action(self, a_t, obs, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        def take_action(i, name):
            if type(name) is int:       # Go to the next view
                self.env.env.sims[i].makeAction([name], [0], [0])
            else:                       # Adjust
                self.env.env.sims[i].makeAction(*self.env_actions[name])

        for i, ob in enumerate(obs):
            action = a_t[i]
            if action != -1:            # -1 is the <stop> action
                select_candidate = ob['candidate'][action]
                src_point = ob['viewIndex']
                trg_point = select_candidate['pointId']
                src_level = (src_point ) // 12  # The point idx started from 0
                trg_level = (trg_point ) // 12
                while src_level < trg_level:    # Tune up
                    take_action(i, 'up')
                    src_level += 1
                while src_level > trg_level:    # Tune down
                    take_action(i, 'down')
                    src_level -= 1
                while self.env.env.sims[i].getState()[0].viewIndex != trg_point:    # Turn right until the target
                    take_action(i, 'right')
                assert select_candidate['viewpointId'] == \
                       self.env.env.sims[i].getState()[0].navigableLocations[select_candidate['idx']].viewpointId
                take_action(i, select_candidate['idx'])

                state = self.env.env.sims[i].getState()[0]
                if traj is not None:
                    traj[i]['path'].append((state.location.viewpointId, state.heading, state.elevation))

    def _teacher_action(self, obs, ended, tracker=None):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = args.ignoreid
            else:
                for k, candidate in enumerate(ob['candidate']):
                    if candidate['viewpointId'] == ob['teacher']:   # Next view point
                        a[i] = k
                        break
                else:   # Stop here
                    assert ob['teacher'] == ob['viewpoint']         # The teacher action should be "STAY HERE"
                    a[i] = len(ob['candidate'])
        return torch.from_numpy(a).cuda()

    def _candidate_variable(self, obs, actions, use_angle=False):
        candidate_feat = np.zeros((len(obs), self.feature_size + args.speaker_angle_size), dtype=np.float32)
        if use_angle:
            candidate_angle = np.zeros((len(obs), args.angle_output),dtype=np.float32)

        for i, (ob, act) in enumerate(zip(obs, actions)):
            if act == -1:  # Ignore or Stop --> Just use zero vector as the feature
                pass
            else:
                c = ob['candidate'][act]
                candidate_feat[i, :] = c['speaker_feature'] 
        return torch.from_numpy(candidate_feat).cuda()
   
    def _feature_variable(self, obs):
        ''' Extract precomputed features into variable. '''
        features = np.empty((len(obs), args.views, self.feature_size + args.speaker_angle_size), dtype=np.float32)
        for i, ob in enumerate(obs):
            features[i, :, :] = ob['speaker_feature']  # Image feat
        return Variable(torch.from_numpy(features), requires_grad=False).cuda()

    def from_shortest_path(self, viewpoints=None, get_first_feat=False):
        obs = self.env._get_obs()
        ended = np.array([False] * len(obs)) # Indices match permuation of the model, not env
        length = np.zeros(len(obs), np.int64)
        img_feats = []
        can_feats = []

        first_feat = np.zeros((len(obs), self.feature_size+args.speaker_angle_size), np.float32)

        for i, ob in enumerate(obs):
            first_feat[i, -args.speaker_angle_size:] = utils.angle_feature(ob['heading'], ob['elevation'])
            # get the heading and elevation of the first viewpoint
        first_feat = torch.from_numpy(first_feat).cuda()
        index = 0
        while not ended.all():
            index += 1
            if viewpoints is not None:
                for i, ob in enumerate(obs):
                    viewpoints[i].append(ob['viewpoint'])
            img_feats.append(self._feature_variable(obs))
            teacher_action = self._teacher_action(obs, ended) # [batch_size, ] the index of action
            teacher_action = teacher_action.cpu().numpy()
            for i, act in enumerate(teacher_action):
                if act < 0 or act == len(obs[i]['candidate']):  # Ignore or Stop
                    teacher_action[i] = -1                      # Stop Action

            can_feats.append(self._candidate_variable(obs, teacher_action))
            # already contain the relavent heading and elevation information.
            self.make_equiv_action(teacher_action, obs)
            length += (1 - ended)
            ended[:] = np.logical_or(ended, (teacher_action == -1))
            obs = self.env._get_obs()
        img_feats = torch.stack(img_feats, 1)  
        can_feats = torch.stack(can_feats, 1) 

        if get_first_feat:
            return (img_feats, can_feats, first_feat), length
        else:
            return (img_feats, can_feats), length

    def gt_words(self, obs):
        seq_tensor = np.array([ob['instr_encoding'] for ob in obs])
        progress_tensor = []
        for i,ob in enumerate(obs):
            progress_tensor.append(list(map(float,ob['speaker_progress'][:args.maxInput]))) 
        progress_tensor = torch.Tensor(progress_tensor)
        return torch.from_numpy(seq_tensor).cuda(), progress_tensor.cuda()

    def teacher_forcing(self, train=True, features=None, insts=None, for_listener=False):
        if train:
            self.model.train()
        else:
            self.model.eval()

        # Get Image Input & Encode
        if features is not None:
            assert insts is not None
            (img_feats, can_feats), lengths = features
            batch_size = len(lengths)
        else:
            obs = self.env._get_obs()
            batch_size = len(obs)
            (img_feats, can_feats), lengths = self.from_shortest_path()    # Image Feature (from the shortest path)
        
        noise = False

        # Get Language Input
        if insts is None:
            insts, progress = self.gt_words(obs)

        ctx_mask = utils.length2mask(lengths) 
        ctx_mask = ctx_mask.cuda()
            
        logits, progress_logits, _, _, _ = self.model(can_feats, img_feats, insts, ctx_mask=ctx_mask,already_dropfeat=noise)
        progress_logits = progress_logits.squeeze()

        # Calculate loss of word prediction
        logits = logits.permute(0, 2, 1)
        loss = self.softmax_loss(
            input  = logits[:, :, :-1],         # -1 for aligning
            target = insts[:, 1:]               # "1:" to ignore the word <BOS>
        )
        
        # Calculate loss of speaker progress monitor
        progress_loss = self.progress_loss(
            input = progress_logits[:,:-1],
            target = progress[:,1:] # to ignore the beginning
        )
        lamda = args.lamda
        w = args.w
        total_loss = lamda * loss + (1-lamda) * w * progress_loss

        if train:   
            return total_loss, progress_loss*w
        else:
            # Evaluation
            _, predict = logits.max(dim=1)                                  # BATCH, LENGTH
            gt_mask = (insts != self.tok.word_to_index['<PAD>'])
            correct = (predict[:, :-1] == insts[:, 1:]) * gt_mask[:, 1:]    # Not pad and equal to gt
            correct, gt_mask = correct.type(torch.LongTensor), gt_mask.type(torch.LongTensor)
            word_accu = correct.sum().item() / gt_mask[:, 1:].sum().item()     # Exclude <BOS>
            sent_accu = (correct.sum(dim=1) == gt_mask[:, 1:].sum(dim=1)).sum().item() / batch_size  # Exclude <BOS>
            return total_loss.item(), word_accu, sent_accu

    def infer_batch(self, sampling=False, train=False, featdropmask=None):
        ''' Used to infer the instruction
        '''
        if train:
            self.model.train()
        else:
            self.model.eval()

        # Image Input for the Encoder
        obs = self.env._get_obs()
        batch_size = len(obs)
        viewpoints_list = [list() for _ in range(batch_size)]

        # Get feature
        (img_feats, can_feats), lengths = self.from_shortest_path(viewpoints=viewpoints_list)
        ctx_mask = utils.length2mask(lengths)
        # This code block is only used for the featdrop.
        if featdropmask is not None:
            img_feats[..., :-args.speaker_angle_size] *= featdropmask 
            can_feats[..., :-args.speaker_angle_size] *= featdropmask
    
        # Encoder
        with torch.no_grad():
            enc_inputs, enc_outputs, _ = self.model.encoder(can_feats, img_feats,already_dropfeat=(featdropmask is not None))
        batch_size = can_feats.size()[0]

        # Decoder
        words = []
        log_probs = []
        entropies = []
        ended = np.zeros(batch_size, np.bool)
        word = np.ones(batch_size, np.int64) * self.tok.word_to_index['<BOS>']   # First word is <BOS>
        word = torch.from_numpy(word).reshape(-1, 1).cuda()

        next_word = np.ones(batch_size, np.int64) * self.tok.word_to_index['<BOS>']
        next_word = torch.from_numpy(next_word).reshape(-1,1).cuda()

        for i in range(args.maxDecode):
            # Decode Step
            with torch.no_grad():
                dec_outputs, _, _ = self.model.decoder(word, enc_outputs, ctx_mask=ctx_mask) 
                logits = self.model.projection(dec_outputs) 

            # Select the word
            logits[:,:, self.tok.word_to_index['<UNK>']] = -float("inf")          # No <UNK> in infer
            if sampling:
                probs = F.softmax(logits, -1)
                m = torch.distributions.Categorical(probs)
                word = m.sample()
                log_prob = m.log_prob(word)
                if train:
                    log_probs.append(log_prob)
                    entropies.append(m.entropy())
                else:
                    log_probs.append(log_prob.detach())
                    entropies.append(m.entropy().detach())
            else:
                values, prob = logits.max(dim=-1,keepdim=True) 
            next_word = prob[:,-1,0] 

            # Append the word
            next_word[ended] = self.tok.word_to_index['<PAD>']
            word = torch.cat([word.detach(),next_word.unsqueeze(-1)],-1) 

            # End?
            ended = np.logical_or(ended, next_word.cpu().numpy() == self.tok.word_to_index['<EOS>'])
            if ended.all():
                break

        if train and sampling:
            return np.stack(words, 1), torch.stack(log_probs, 1), torch.stack(entropies, 1)
        else:
            return word.cpu().numpy()

    def save(self, epoch, path):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
            states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
        all_tuple = [("PASTS", self.model, self.optimizer)]
        for param in all_tuple:
            create_state(*param)
        torch.save(states, path)

    def load(self, path):
        ''' Loads parameters (but not training state) '''
        print("Load the speaker's state dict from %s" % path)
        states = torch.load(path)
        def recover_state(name, model, optimizer):
            state = model.state_dict()
            state.update(states[name]['state_dict'])
            model.load_state_dict(state,strict=True) 
            if args.loadOptim:
                optimizer.load_state_dict(states[name]['optimizer'],strict=True)
        all_tuple = [("PASTS", self.model, self.optimizer)]
        for param in all_tuple:
            recover_state(*param)
        print('load epoch:{}'.format(states['PASTS']['epoch']-1))
        return states['PASTS']['epoch'] - 1


