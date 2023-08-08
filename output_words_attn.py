
'''
Author: Anonymous submission 
Time: 2023-03-12 18:03 
'''

import random
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader 

from models.text_gt_v import TextGT_v 

from data_utils import SentenceDataset, build_tokenizer, build_embedding_matrix 
from prepare_vocab import VocabHelp 

from visualization.visualize import draw 

def setup_seed(seed): 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 

def words_attn_of_one_sentence(opt): 
    tokenizer = build_tokenizer(
        fnames=[opt.dataset_file['train'], opt.dataset_file['test']], 
        max_length=opt.max_length, 
        data_file='{}/{}_tokenizer.dat'.format(opt.vocab_dir, opt.dataset))
    embedding_matrix = build_embedding_matrix(
        vocab=tokenizer.vocab, 
        embed_dim=opt.embed_dim, 
        data_file='{}/{}d_{}_embedding_matrix.dat'.format(opt.vocab_dir, str(opt.embed_dim), opt.dataset))

    post_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_post.vocab')    # position
    pos_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_pos.vocab')      # POS
    dep_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_dep.vocab')      # deprel
    pol_vocab = VocabHelp.load_vocab(opt.vocab_dir + '/vocab_pol.vocab')      # polarity

    # opt.tok_size = len(token_vocab) 
    opt.post_size = len(post_vocab) 
    opt.pos_size = len(pos_vocab) 
    opt.deprel_size = len(dep_vocab) 

    vocab_help = (post_vocab, pos_vocab, dep_vocab, pol_vocab)
    model = opt.model_class(embedding_matrix, opt) 
    model.load_state_dict(torch.load('./state_dict/the-dict-file-of-a-trained-model')) 
    model.eval() 
    testset = SentenceDataset(opt.dataset_file['test'], tokenizer, opt=opt, vocab_help=vocab_help)
    test_dataloader = DataLoader(dataset=testset, batch_size=1) 
    one_sample_batch = list(iter(test_dataloader))[opt.sentence_id-1] 
    inputs = [one_sample_batch[col] for col in opt.inputs_cols] 
    targets = one_sample_batch['polarity'] 
    words_attn = model.get_words_attn(inputs).mean(dim=0)[:one_sample_batch['length'], :one_sample_batch['length']] # (L, L) 
    # print(words_attn) 

    test_cases_file = open('./test_cases.txt', 'r') 
    lines = test_cases_file.readlines() 

    draw(lines[opt.sentence_id-1], words_attn) 


def main():
    model_classes = {
        'text-gt-v': TextGT_v 
    } 

    vocab_dirs = {
        'restaurant': './dataset/Restaurants_corenlp',
        'laptop': './dataset/Laptops_corenlp',
        'twitter': './dataset/Tweets_corenlp',
        'rest16': './dataset/Restaurants16', 
    } 
    
    
    dataset_files = {
        'restaurant': {
            'train': './dataset/Restaurants_corenlp/train.json',
            'test': './dataset/Restaurants_corenlp/test.json',
        },
        'laptop': {
            'train': './dataset/Laptops_corenlp/train.json',
            'test': './dataset/Laptops_corenlp/test.json'
        },
        'twitter': {
            'train': './dataset/Tweets_corenlp/train.json',
            'test': './dataset/Tweets_corenlp/test.json',
        }, 
        'rest16': { 
            'train': './dataset/Restaurants16/train.json', 
            'test': './dataset/Restaurants16/test.json', 
        } 
    }
    
    input_colses = { 
        'text-gt': ['text', 'aspect', 'pos', 'head', 'deprel', 'post', 'mask', 'length', 'adj']
    } 
    
    # Hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='text-gt-v', type=str, help=', '.join(model_classes.keys()))
    parser.add_argument('--dataset', default='restaurant', type=str, help=', '.join(dataset_files.keys()))
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--post_dim', type=int, default=60, help='Position embedding dimension.')
    parser.add_argument('--pos_dim', type=int, default=30, help='Pos embedding dimension.') 
    parser.add_argument('--deprel_dim', type=int, default=30, help='Dependent relation embedding dimension.') 
    parser.add_argument('--hidden_dim', type=int, default=60, help='GCN mem dim.') 
    parser.add_argument('--num_layers', type=int, default=4, help='Num of GCN layers.') 
    parser.add_argument('--polarities_dim', default=3, type=int, help='3') 

    parser.add_argument('--input_dropout', type=float, default=0.7, help='Input dropout rate.')
    parser.add_argument('--lower', default=True, help='Lowercase all words.')

    # no need to specified, may incur error 
    parser.add_argument('--directed', default=False, help='directed graph or undirected graph')
    parser.add_argument('--add_self_loop', default=True) 

    parser.add_argument('--use_rnn', action='store_true') 
    parser.add_argument('--bidirect', default=True, help='Do use bi-RNN layer.')
    parser.add_argument('--rnn_hidden', type=int, default=60, help='RNN hidden state size.')
    parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
    parser.add_argument('--rnn_dropout', type=float, default=0.1, help='RNN dropout rate.') 
    
    parser.add_argument('--max_length', default=85, type=int) 
    parser.add_argument('--device', default=None, type=str, help='cpu, cuda')
    parser.add_argument('--seed', default=1000, type=int)
    parser.add_argument('--vocab_dir', type=str, default='./dataset/Tweets_corenlp') 
    parser.add_argument('--pad_id', default=0, type=int) 

    # TextGT specific hyper-parameters 
    parser.add_argument('--graph_conv_type', type=str, default='gin', choices=['eela', 'gcn', 'gin', 'gat']) 
    parser.add_argument('--graph_conv_attention_heads', default=4, type=int) 
    parser.add_argument('--graph_conv_attn_dropout', type=float, default=0.0) 
    parser.add_argument('--attention_heads', default=4, type=int) 
    parser.add_argument('--attn_dropout', type=float, default=0.1) 
    parser.add_argument('--ffn_dropout', type=float, default=0.3) 
    parser.add_argument('--norm', type=str, default='ln', choices=['ln', 'bn']) 
    parser.add_argument('--max_position', type=int, default=9) 

    parser.add_argument('--sentence_id', default=0, type=int) 
    opt = parser.parse_args()
    	
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset] 
    opt.inputs_cols = input_colses[opt.model_name] 

    opt.vocab_dir = vocab_dirs[opt.dataset] 

    # set random seed
    setup_seed(opt.seed) 

    words_attn_of_one_sentence(opt) 


if __name__ == '__main__':
    main()
