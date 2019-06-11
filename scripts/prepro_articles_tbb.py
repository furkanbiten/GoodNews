import tqdm
import h5py
# import spacy
import numpy as np
import json
# import argparse
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.extmath import randomized_svd

def open_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_h5(file_save, data):
    f_lb = h5py.File(file_save, "w")
    dt = h5py.special_dtype(vlen=np.dtype('float64'))
    ds = f_lb.create_dataset('average', (len(data), 300, ), dtype=dt)
    for i,d in tqdm.tqdm(enumerate(data)):
        ds[i] = d
    f_lb.close()

if __name__ == '__main__':
    np.random.seed(42)
    data_com = h5py.File('../data/articles_full_WeightedAvg.h5')
    data_com = [np.stack(d) for d in tqdm.tqdm(data_com.get('average'))]
    lengths = [d.shape[1] for d in data_com]
    data_com = [d.reshape(-1, 300) for d in data_com]
    data_com = [i for d in data_com for i in d]
    # d = data_com[:sum(lengths[:100])]
    svd = TruncatedSVD(n_components=10, n_iter=7, random_state=42)
    svd.fit(data_com)
    pc = svd.components_

    data_com = data_com - np.dot(data_com, np.transpose(pc).dot(pc))
    new = []
    index = 0
    for l in lengths:
        new.append(data_com[index:index + l].reshape(300, -1))
        index += l
    keys = open_json('../data/articles_full_WeightedAvg_keys.json')
    json.dump(keys, open('../data/articles_full_TBB_keys.json', 'wb'))
    save_h5('../data/articles_full_TBB.h5', new)