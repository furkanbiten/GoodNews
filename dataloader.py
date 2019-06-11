from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from joblib import Parallel, delayed
import json
import h5py
import os
import warnings
warnings.filterwarnings('ignore')
import tables
# import multiprocessing
# from multiprocessing import Process
# from multiprocessing.dummy import Pool as ThreadPool
# from pathos.multiprocessing import Pool
# import pathos.pools as pp
import numpy as np
import random
import torch
from torchvision import transforms as trn

preprocess = trn.Compose([
        #trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# def unwrap_self(args):
#     cls, arg = args
#     return cls.get_batch_one(arg)
def func(*args, **kwargs):
    return DataLoader.get_batch_one(args, kwargs)

def get_img(args):
    h5_image_file, ix = args
    temp_h5 = tables.open_file(h5_image_file, mode='r')
    img = np.array(temp_h5.root.images[ix, :, :, :])
    img_batch = preprocess(torch.from_numpy(img.astype('float32') / 255.0)).numpy()
    temp_h5.close()
    return img_batch

def get_sen_embed(args):
    h5_sen_file, sen_ix = args
    temp_h5 = h5py.File(h5_sen_file, mode='r')
    sen_embed = np.stack(temp_h5['average'][sen_ix, :]).transpose()
    temp_h5.close()
    return sen_embed

def combine(args):
    h5_image_file, ix, h5_sen_file, sen_ix = args
    arg1 = h5_image_file, ix
    arg2 = h5_sen_file, sen_ix
    img = get_img(arg1)
    sen = get_sen_embed(arg2)
    return img, sen

class DataLoader:
    
    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.seq_per_img = self.opt.seq_per_img
        self.num_thread = 1
        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)
        print('vocab size is ', self.vocab_size)


        # open the hdf5 file
        print('DataLoader loading h5 file: ', opt.input_label_h5, opt.input_image_h5)
        # self.h5_label_file = h5py.File(self.opt.input_label_h5, mode='r')
        self.h5_label_file = tables.open_file(self.opt.input_label_h5, driver="H5FD_CORE")
        # self.h5_image_file = h5py.File(self.opt.input_image_h5, mode='r')
        self.h5_image_file = tables.open_file(self.opt.input_image_h5, mode='r')
        if 'sentence_embed' in opt:
            if opt.sentence_embed:
                self.h5_sen_embed_file = h5py.File(self.opt.sentence_embed, mode='r')
                # self.h5_sen_embed_file = tables.open_file(self.opt.sentence_embed, mode='r')
                self.sen_embed_keys = json.load(open(self.opt.sentence_embed.split('.h5')[0] + '_keys.json'))
                # self.sen_embed_file = da.from_array(self.h5_sen_embed_file['average'],
                #                     chunks=(self.h5_sen_embed_file['average'].shape[0], 300, ))
        else:
            self.opt.sentence_embed = False

        # extract image size from dataset
        # images_size = self.h5_image_file['images'].shape
        images_size = self.h5_image_file.root.images.shape
        assert len(images_size) == 4, 'images should be a 4D tensor'
        assert images_size[2] == images_size[3], 'width and height must match'
        self.num_images = images_size[0]
        self.num_channels = images_size[1]
        self.max_image_size = images_size[2]
        print('read %d images of size %dx%dx%d' %(self.num_images,
                    self.num_channels, self.max_image_size, self.max_image_size))

        # load in the sequence data
        # seq_size = self.h5_label_file['labels'].shape
        seq_size = self.h5_label_file.root.labels.shape
        self.seq_length = seq_size[1]
        print('max sequence length in data is', self.seq_length)
        # load the pointers in full to RAM (should be small enough)
        self.label_start_ix = np.array(self.h5_label_file.root.label_start_ix)
        # self.label_start_ix = np.array(self.h5_label_file['label_start_ix'])
        self.label_end_ix = np.array(self.h5_label_file.root.label_end_ix)
        # self.label_end_ix = np.array(self.h5_label_file['label_end_ix'])

        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': []}
        # TODO: for nytimes dataset
        self.id_to_keys = {i['id']: i['file_path'].split('/')[1].split('_')[0] for i in self.info['images']}
        # TODO: for breakinNews
        # self.id_to_keys = {i['id']: i['id'] for i in self.info['images']}


        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif opt.train_only == 0: # restval
                self.split_ix['train'].append(ix)

        print('assigned %d images to split train' % len(self.split_ix['train']))
        print('assigned %d images to split val' % len(self.split_ix['val']))
        print('assigned %d images to split test' % len(self.split_ix['test']))

        self.iterators = {'train': 0, 'val': 0, 'test': 0}
        self.shuffle = {'train': np.random.permutation(np.arange(len(self.split_ix['train']))),
                        'val': np.arange(len(self.split_ix['val'])),
                        'test': np.arange(len(self.split_ix['test']))}
    def __len__(self):
        return len(self.split_ix['train'])

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length
    def get_batch_one(self, split):
        split_ix = self.split_ix[split]
        batch_size = 1
        img_batch = np.ndarray([batch_size, 3, 256, 256], dtype='float32')
        label_batch = np.zeros([batch_size * self.seq_per_img, self.seq_length + 2], dtype='int')
        mask_batch = np.zeros([batch_size * self.seq_per_img, self.seq_length + 2], dtype='float32')
        if self.opt.sentence_embed:
            sen_embed_batch = np.zeros(
                [batch_size * self.seq_per_img, self.opt.sentence_length + 1, self.opt.sentence_embed_size],
                dtype='float32')
        max_index = len(split_ix)
        infos = []
        b_id = self.shuffle[split][self.iterators[split]: self.iterators[split] + batch_size][0]
        self.iterators[split] += batch_size
        if self.iterators[split] >= max_index:
            np.random.shuffle(self.shuffle[split])
            self.iterators[split] = 0

        i=0
        ix = split_ix[b_id]

        # fetch image
        # img = self.load_image(self.image_info[ix]['filename'])
        # img = np.array(self.h5_image_file['images'][ix, :, :, :])
        img = np.array(self.h5_image_file.root.images[ix, :, :, :])
        img_batch[i] = preprocess(torch.from_numpy(img.astype('float32') / 255.0)).numpy()

        # fetch the sequence labels
        ix1 = self.label_start_ix[ix] - 1  # label_start_ix starts from 1
        ix2 = self.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1  # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        if ncap < self.seq_per_img:
            # we need to subsample (with replacement)
            seq = np.zeros([self.seq_per_img, self.seq_length], dtype='int')
            for q in range(self.seq_per_img):
                ixl = random.randint(ix1, ix2)
                # seq[q, :] = self.h5_label_file['labels'][ixl, :self.seq_length]
                seq[q, :] = self.h5_label_file.root.labels[ixl, :self.seq_length]
        else:
            ixl = random.randint(ix1, ix2 - self.seq_per_img + 1)
            # seq = self.h5_label_file['labels'][ixl: ixl + self.seq_per_img, :self.seq_length]
            seq = self.h5_label_file.root.labels[ixl: ixl + self.seq_per_img, :self.seq_length]

        label_batch[i * self.seq_per_img: (i + 1) * self.seq_per_img, 1: self.seq_length + 1] = seq

        # record associated info as well
        info_dict = {}
        info_dict['id'] = self.info['images'][ix]['id']
        info_dict['file_path'] = self.info['images'][ix]['file_path']
        # fetch sen_embed
        if self.opt.sentence_embed:
            # for q in range(self.seq_per_img):
            key = self.id_to_keys[info_dict['id']]
            sen_ix = self.sen_embed_keys.index(key)
            sen_embed = np.stack(self.h5_sen_embed_file['average'][sen_ix, :]).transpose()
            # sen_embed = np.stack(self.h5_sen_embed_file.root.average[sen_ix, :]).transpose()
            sen_embed_batch[i, :len(sen_embed), :] = sen_embed
        infos.append(info_dict)

        # generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 2, label_batch)))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1

        data = {}
        if self.opt.sentence_embed:
            data['sen_embed'] = sen_embed_batch

        data['images'] = img_batch
        data['labels'] = label_batch
        data['masks'] = mask_batch
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(split_ix)}
        data['infos'] = infos

        return data

    def __call__(self, split):
        return self.get_batch_one(split)


    def get_batch(self, split, batch_size=None):
        split_ix = self.split_ix[split]
        batch_size = batch_size or self.batch_size

        # img_batch = np.ndarray([batch_size, 3, 256, 256], dtype='float32')
        label_batch = np.zeros([batch_size * self.seq_per_img, self.seq_length + 2], dtype='int')
        mask_batch = np.zeros([batch_size * self.seq_per_img, self.seq_length + 2], dtype='float32')
        # if self.opt.sentence_embed:
        #     sen_embed_batch = np.zeros(
        #         [batch_size * self.seq_per_img, self.opt.sentence_length + 1, self.opt.sentence_embed_size],
        #         dtype='float32')
        max_index = len(split_ix)
        wrapped = False
        infos = []
        # temp_img_h5 = tables.open_file(self.opt.input_image_h5, mode='r')
        # Parallel(n_jobs=self.num_thread, verbose=0, backend="loky")(map(delayed(self.get_batch_one),
        #                                                                           range(self.batch_size)))
        batch_ids = self.shuffle[split][self.iterators[split]: self.iterators[split]+batch_size].tolist()
        self.iterators[split] += batch_size
        if self.iterators[split] >= max_index:
            if split=='train':
                np.random.shuffle(self.shuffle[split])
            self.iterators[split] = 0
            wrapped = True
            if len(batch_ids) != batch_size:
                leftover = batch_size - len(batch_ids)
                batch_ids.extend(self.shuffle[split][self.iterators[split]: self.iterators[split]+leftover])
                self.iterators[split] += leftover

        #combine
        if self.opt.sentence_embed:
            keys = [self.id_to_keys[self.info['images'][i]['id']] for i, b_id in enumerate(batch_ids)]
            sen_ixs = [self.sen_embed_keys.index(key) for key in keys]

            combined = Parallel(n_jobs=self.num_thread, verbose=0, backend="loky")(
                map(delayed(combine), [(self.opt.input_image_h5, split_ix[b_id], self.opt.sentence_embed, s
                                        ) for b_id, s in zip(batch_ids, sen_ixs)]))
            img_batch = [c[0] for c in combined]
            sen = [c[1] for c in combined]
            if vars(self.opt).get('sentence_embed_method', None) == 'fc' or \
                    vars(self.opt).get('sentence_embed_method', None) == 'fc_max':
                sen_embed_batch = [np.pad(a, ((0, self.opt.sentence_length + 1 - len(a)), (0, 0)),
                                          'constant', constant_values=0) for a in sen]
            else:
                sen_embed_batch = [np.pad(a, ((0, self.opt.sentence_length - len(a)), (0, 0)),
                                          'constant', constant_values=0) if len(a)<self.opt.sentence_length else a[:self.opt.sentence_length] for a in sen]
                sen_embed_batch = np.array(sen_embed_batch, dtype=np.float32)
        else:
            # combined = Parallel(n_jobs=self.num_thread, verbose=0, backend="loky")(
            #     map(delayed(get_img), [(self.opt.input_image_h5, split_ix[b_id]) for b_id in batch_ids]))
            img_batch = [get_img((self.opt.input_image_h5, split_ix[b_id])) for b_id in batch_ids]

        img_batch = np.array(img_batch)

        for i, b_id in enumerate(batch_ids):
        # for i in range(batch_size):
        #     ri = self.iterators[split]
        #     ri_next = ri + 1
        #     if ri_next >= max_index:
        #         np.random.shuffle(self.shuffle[split])
        #         ri_next = 0
        #         wrapped = True
        #     self.iterators[split] = ri_next
        #     ix = split_ix[ri]
            ix = split_ix[b_id]

            # fetch image
            # img = self.load_image(self.image_info[ix]['filename'])
            # img = np.array(self.h5_image_file['images'][ix, :, :, :])
            # img = np.array(self.h5_image_file.root.images[ix, :, :, :])
            # img_batch[i] = preprocess(torch.from_numpy(img.astype('float32')/255.0)).numpy()

            # fetch the sequence labels
            ix1 = self.label_start_ix[ix] - 1 #label_start_ix starts from 1
            ix2 = self.label_end_ix[ix] - 1
            ncap = ix2 - ix1 + 1 # number of captions available for this image
            assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

            if ncap < self.seq_per_img:
                # we need to subsample (with replacement)
                seq = np.zeros([self.seq_per_img, self.seq_length], dtype = 'int')
                for q in range(self.seq_per_img):
                    ixl = random.randint(ix1,ix2)
                    # seq[q, :] = self.h5_label_file['labels'][ixl, :self.seq_length]
                    seq[q, :] = self.h5_label_file.root.labels[ixl, :self.seq_length]
            else:
                ixl = random.randint(ix1, ix2 - self.seq_per_img + 1)
                # seq = self.h5_label_file['labels'][ixl: ixl + self.seq_per_img, :self.seq_length]
                seq = self.h5_label_file.root.labels[ixl: ixl + self.seq_per_img, :self.seq_length]

            label_batch[i * self.seq_per_img : (i + 1) * self.seq_per_img, 1 : self.seq_length + 1] = seq

            # record associated info as well
            info_dict = {}
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix]['file_path']
            # fetch sen_embed
            # if self.opt.sentence_embed:
            #     # for q in range(self.seq_per_img):
            #     key = self.id_to_keys[info_dict['id']]
            #     sen_ix = self.sen_embed_keys.index(key)
            #     sen_embed = np.stack(self.h5_sen_embed_file['average'][sen_ix, :]).transpose()
            #     # sen_embed = np.stack(self.h5_sen_embed_file.root.average[sen_ix, :]).transpose()
            #     sen_embed_batch[i, :len(sen_embed), :] = sen_embed
            infos.append(info_dict)

        # generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum()+2, label_batch)))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1

        data = {}
        if self.opt.sentence_embed:
            data['sen_embed'] = sen_embed_batch

        data['images'] = img_batch
        data['labels'] = label_batch
        data['masks'] = mask_batch 
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(split_ix), 'wrapped': wrapped}
        data['infos'] = infos

        return data

    def reset_iterator(self, split):
        self.iterators[split] = 0
