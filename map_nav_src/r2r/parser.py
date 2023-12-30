import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--root_dir', type=str, default='')
    parser.add_argument('--dataset', type=str, default='r2r', choices=['r2r'])
    parser.add_argument('--output_dir', type=str, default='', help='experiment id')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--tokenizer', choices=['bert', 'xlm'], default='bert')

    parser.add_argument('--act_visited_nodes', action='store_true', default=False)
    parser.add_argument('--fusion', default='dynamic',choices=['global', 'local', 'avg', 'dynamic'])
    parser.add_argument('--expl_sample', action='store_true', default=False)
    parser.add_argument('--expl_max_ratio', type=float, default=0.6)
    parser.add_argument('--expert_policy', default='spl', choices=['spl', 'ndtw'])

    # distributional training (single-node, multiple-gpus)
    parser.add_argument('--world_size', type=int, default=1, help='number of gpus')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument("--node_rank", type=int, default=0, help="Id of the node")
    
    # General
    parser.add_argument('--iters', type=int, default=200000, help='training iterations')
    parser.add_argument('--log_every', type=int, default=1000)
    parser.add_argument('--eval_first', action='store_true', default=False)

    # Data preparation
    parser.add_argument('--max_instr_len', type=int, default=200)
    parser.add_argument('--max_action_len', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--ignoreid', type=int, default=-100, help='ignoreid for action')
    
    # Load the model from
    parser.add_argument("--resume_file", default=None, help='path of the trained model')
    parser.add_argument("--resume_optimizer", action="store_true", default=False)

    # Augmented Paths from
    parser.add_argument("--aug", default=None)
    parser.add_argument('--bert_ckpt_file', default=None, help='init vlnbert')

    # Listener Model Config
    parser.add_argument("--ml_weight", type=float, default=0.20)
    parser.add_argument('--entropy_loss_weight', type=float, default=0.01)

    parser.add_argument("--features", type=str, default='vitbase')

    parser.add_argument('--fix_lang_embedding', action='store_true', default=False)
    parser.add_argument('--fix_pano_embedding', action='store_true', default=False)
    parser.add_argument('--fix_local_branch', action='store_true', default=False)

    parser.add_argument('--num_l_layers', type=int, default=9)
    parser.add_argument('--num_pano_layers', type=int, default=2)
    parser.add_argument('--num_x_layers', type=int, default=4)

    parser.add_argument('--enc_full_graph', default=True, action='store_true')
    parser.add_argument('--graph_sprels', action='store_true', default=True)

    # Dropout Param
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--feat_dropout', type=float, default=0.4)

    # Submision configuration
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument("--submit", action='store_true', default=False)
    parser.add_argument('--no_backtrack', action='store_true', default=False)
    parser.add_argument('--detailed_output', action='store_true', default=False)

    # Training Configurations
    parser.add_argument(
        '--optim', type=str, default='adamW',
        choices=['rms', 'adam', 'adamW', 'sgd']
    )    # rms, adam
    parser.add_argument('--lr', type=float, default=0.00001, help="the learning rate")
    parser.add_argument('--decay', dest='weight_decay', type=float, default=0.)
    parser.add_argument(
        '--feedback', type=str, default='sample',
        help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``'
    )
    parser.add_argument('--epsilon', type=float, default=0.1, help='')

    # Model hyper params:
    parser.add_argument("--angle_feat_size", type=int, default=4)
    parser.add_argument('--image_feat_size', type=int, default=768)
    parser.add_argument('--obj_feat_size', type=int, default=0)
    parser.add_argument('--views', type=int, default=36)

    # # A2C
    parser.add_argument("--gamma", default=0.9, type=float, help='reward discount factor')
    parser.add_argument(
        "--normalize", dest="normalize_loss", default="total", 
        type=str, help='batch or total'
    )
    parser.add_argument('--train_alg', 
        choices=['imitation', 'dagger'], 
        default='dagger'
    )

    # Speaker
    parser.add_argument("--speaker", default=None)
    parser.add_argument('--maxDecode', type=int, default=120, help="max input instruction")
    parser.add_argument('--use_speaker',default=False,action='store_true')
    parser.add_argument('--use_drop',action='store_true',default=False)
    parser.add_argument('--speaker_dropout', type=float, default=0.2)
    parser.add_argument('--w',default=10.,type=float)
    parser.add_argument('--lamda',default=0.8,type=float)
    parser.add_argument('--wemb',type=int,default=256)
    parser.add_argument('--h_dim', type=int, default=512)
    parser.add_argument('--aemb', type=int, default=64)
    parser.add_argument('--proj', type=int, default=512)
    parser.add_argument('--proj_hidden',default=1024,type=int) 
    parser.add_argument("--feature_size",default=640,type=int)
    parser.add_argument("--accumulateGrad", dest='accumulate_grad', action='store_const', default=False, const=True)
    parser.add_argument('--featdropout', type=float, default=0.3)
    parser.add_argument('--maxInput', type=int, default=80, help="max input instruction")
    parser.add_argument('--maxAction', type=int, default=15, help='Max Action sequence')
    parser.add_argument("--loadOptim",action="store_const", default=False, const=True)
    parser.add_argument("--speaker_angle_size",type=int,default=128)
    parser.add_argument("--maxSrclen",type=int,default=7)
    parser.add_argument("--speaker_ckpt_path",type=str,default='') # The loading path of speaker
    parser.add_argument('--speaker_layer_num',default=6,type=int) 
    parser.add_argument('--speaker_head_num',default=6,type=int)
    parser.add_argument('--compute_coco', action='store_true', default=False)

    # others
    parser.add_argument("--train",type=str,required=True) # follower, speaker, valid_follower, valid_speaker
    parser.add_argument("--name",type=str,default='debug')

    args, _ = parser.parse_known_args()

    args = postprocess_args(args)

    return args


def postprocess_args(args):
    ROOTDIR = args.root_dir

    # Setup input paths
    ft_file_map = {
        'vitbase': 'pth_vit_base_patch16_224_imagenet.hdf5',
        'clip640':'CLIP-ResNet-50x4-views.tsv',
        'resnet': 'ResNet-152-imagenet.tsv',
        'clip768': 'CLIP-ViT-B-16-views.hdf5'
    }
    args.img_ft_file = os.path.join(ROOTDIR, 'R2R', 'features', ft_file_map[args.features])
    if args.features == 'vitbase':
        args.img_type = 'hdf5'
        args.feature_size = args.image_feat_size = 768
    elif args.features == 'clip640':
        args.img_type = 'tsv'
        args.feature_size = args.image_feat_size = 640
    elif args.features == 'resnet':
        args.img_type = 'tsv'
        args.faeture_size = args.image_feat_size = 2048
    elif args.features == 'clip768':
        args.img_type = 'hdf5'
        args.feature_size = args.image_feat_size = 768

    args.connectivity_dir = os.path.join(ROOTDIR, 'R2R', 'connectivity')
    args.scan_data_dir = os.path.join(ROOTDIR, 'Matterport3D', 'v1_unzip_scans')

    args.anno_dir = os.path.join(ROOTDIR, 'R2R', 'annotations')
    args.fg_anno_dir = os.path.join(ROOTDIR, 'R2R', 'annotations','FG')

    # Build paths
    if args.train == 'speaker':
        args.output_dir = os.path.join(args.output_dir,'speaker',args.name)
    elif args.train == 'follower':
        args.output_dir = os.path.join(args.output_dir,'follower',args.name)
    elif args.train == 'valid_speaker':
        args.output_dir = os.path.join(args.output_dir,'valid_speaker',args.name)
    elif args.train == 'valid_follower':
        args.output_dir = os.path.join(args.output_dir,'valid_follower',args.name)

    # Build paths
    args.ckpt_dir = os.path.join(args.output_dir, 'ckpts')
    args.log_dir = os.path.join(args.output_dir, 'logs')
    args.pred_dir = os.path.join(args.output_dir, 'preds')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.pred_dir, exist_ok=True)

    return args

