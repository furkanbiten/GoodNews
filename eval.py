from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import numpy as np

import time
import os
from six.moves import cPickle

import opts
import models
from dataloader import *
from dataloaderraw import *
import eval_utils
import argparse
import misc.utils as utils
import torch

# Input arguments and options
parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--model_path', type=str, default='./save/show_attend_tell/model-best.pth',
                help='path to model to evaluate')
parser.add_argument('--cnn_model_path', type=str, default='./save/show_attend_tell/model-cnn-best.pth',
                help='path to cnn model to evaluate')
parser.add_argument('--infos_path', type=str, default='./save/show_attend_tell/infos_-best.pkl',
                help='path to infos to evaluate')
parser.add_argument('--sen_embed_path', type=str, default='./data/articles_full_avg.h5', #'./data/sen_embed/articles_compact_TBB.h5'
                help='path to sentence embedding in data folder')
# Basic options
parser.add_argument('--save_name', type=str, default='default', help='the name for saving the output file')
parser.add_argument('--batch_size', type=int, default=32,
                help='if > 0 then overrule, otherwise load from checkpoint.')
parser.add_argument('--num_images', type=int, default=-1,
                help='how many images to use when periodically evaluating the loss? (-1 = all)')
parser.add_argument('--language_eval', type=int, default=1,
                help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-caption code from Github.')
parser.add_argument('--dump_images', type=int, default=0,
                help='Dump images into vis/imgs folder for vis? (1=yes,0=no)')
parser.add_argument('--dump_json', type=int, default=1,
                help='Dump json with predictions into vis folder? (1=yes,0=no)')
parser.add_argument('--dump_path', type=int, default=0,
                help='Write image paths along with predictions into vis json? (1=yes,0=no)')

# Sampling options
parser.add_argument('--sample_max', type=int, default=1,
                help='1 = sample argmax words. 0 = sample from distributions.')
parser.add_argument('--beam_size', type=int, default=1,
                help='used when sample_max = 1, indicates number of beams in beam search. Usually 2 or 3 works well. More is not better. Set this to 1 for faster runtime but a bit worse performance.')
parser.add_argument('--temperature', type=float, default=0.0,
                help='temperature when sampling from distributions (i.e. when sample_max = 0). Lower = "safer" predictions.')
# For evaluation on a folder of images:
parser.add_argument('--image_folder', type=str, default='', #/home/abiten/Desktop/Thesis/europana/test/
                help='If this is nonempty then will predict on the images in this folder path')
parser.add_argument('--image_root', type=str, default='',
                help='In case the image paths have to be preprended with a root path to an image folder')
# For evaluation on MSCOCO images from some split:
parser.add_argument('--input_label_h5', type=str, default='./data/data_news_label.h5',
                help='path to the h5file containing the preprocessed label')
parser.add_argument('--input_image_h5', type=str, default='',
                help='path to the h5file containing the preprocessed image')
parser.add_argument('--input_json', type=str, default='data/data_news.json',
                help='path to the json file containing additional info and vocab. empty = fetch from model checkpoint.')
parser.add_argument('--split', type=str, default='test', 
                help='if running on MSCOCO images, which split to use: val|test|train')
parser.add_argument('--coco_json', type=str, default='', 
                help='if nonempty then use this file in DataLoaderRaw (see docs there). Used only in MSCOCO test evaluation, where we have a specific json file of only test set images.')
# misc
parser.add_argument('--sentence_embed_method', type=str, default='fc',
                        help='choose which method to use, available options are conv, conv_deep, fc, bnews, fc_max')
parser.add_argument('--id', type=str, default='', 
                help='an id identifying this run/job. used only if language_eval = 1 for appending to intermediate files')
parser.add_argument('--return_attention', type=bool, default=True,
                help='This should only be run when sentence attention architecture is used. When set to True, '
                     'it will write the attention weights for article and images to json')

opt = parser.parse_args()

# Load infos
with open(opt.infos_path) as f:
    infos = cPickle.load(f)

# override and collect parameters
if len(opt.input_label_h5) == 0:
    opt.input_label_h5 = infos['opt'].input_label_h5
if len(opt.input_image_h5) == 0:
    opt.input_image_h5 = infos['opt'].input_image_h5
if len(opt.input_json) == 0:
    opt.input_json = infos['opt'].input_json
if opt.batch_size == 0:
    opt.batch_size = infos['opt'].batch_size
if len(opt.id) == 0:
    opt.id = infos['opt'].id
ignore = ["id", "batch_size", "beam_size", "start_from"] # , "language_eval"
for k in vars(infos['opt']).keys():
    if k not in ignore:
        if k in vars(opt):
            pass
            # assert vars(opt)[k] == vars(infos['opt'])[k], k + ' option not consistent'
        else:
            vars(opt).update({k: vars(infos['opt'])[k]}) # copy over options from model

vocab = infos['vocab'] # ix -> word mapping

# Setup the model
cnn_model = utils.build_cnn(opt)
cnn_model.load_state_dict(torch.load(opt.cnn_model_path))
cnn_model.cuda()
cnn_model.eval()
model = models.setup(opt)
model.load_state_dict(torch.load(opt.model_path))
model.cuda()
model.eval()
crit = utils.LanguageModelCriterion()
opt.seq_per_img = 1
# opt.sentence_embed = './data/data_news_compact_lda_label.h5'
# opt.sentence_embed = './data/sen_embed/articles_compact_avg.h5'
# Create the Data Loader instance
if len(opt.image_folder) == 0:
  loader = DataLoader(opt)
else:
  loader = DataLoaderRaw({'folder_path': opt.image_folder, 
                            'coco_json': opt.coco_json,
                            'batch_size': opt.batch_size})
# When eval using provided pretrained model, the vocab may be different from what you have in your cocotalk.json
# So make sure to use the vocab in infos file.
loader.ix_to_word = infos['vocab']


# Set sample options
loss, split_predictions, lang_stats = eval_utils.eval_split(cnn_model, model, crit, loader, 
    vars(opt), return_attention=opt.return_attention)

print('loss: ', loss)
if lang_stats:
  print(lang_stats)

if opt.dump_json == 1:
    # dump the json
    json.dump(split_predictions, open('vis/vis_'+vars(infos['opt'])['caption_model']+
                                      '_'+vars(opt)['save_name']+'.json', 'w'))
