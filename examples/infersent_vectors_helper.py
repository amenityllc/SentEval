# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from datetime import datetime

import pandas as pd
import torch
import io

# get models.py from InferSent repo
from InferSent.models import InferSent


# ------------------------------------------Clustering-----------------------------------------------------------------------


class InferSentVectorsHelper:
    def __init__(self, bsize=64, word_emb_dim=300, enc_lstm_dim=2048, pool_type='max', dpout_model=0.0, version=2,
                 model_path='../infersent/infersent2.pkl',
                 path_to_w2v='../fasttext/crawl-300d-2M.vec',
                 use_cuda=True):
        self.version = version
        self.dpout_model = dpout_model
        self.pool_type = pool_type
        self.enc_lstm_dim = enc_lstm_dim
        self.word_emb_dim = word_emb_dim
        self.bsize = bsize
        model = InferSent({'bsize': bsize, 'word_emb_dim': word_emb_dim, 'enc_lstm_dim': enc_lstm_dim,
                    'pool_type': pool_type, 'dpout_model': dpout_model, 'version': version})
        model.load_state_dict(torch.load(model_path))
        model.set_w2v_path(path_to_w2v)

        if not use_cuda:
            self.model = model
        else:
            self.model = model.cuda()

        self.first_call = True

    def get_vectors(self, sentence_list, use_phrases=False):
        start = datetime.now()
        if self.first_call:
            self.model.build_vocab(sentence_list, tokenize=True)
        else:
            self.model.update_vocab(sentence_list, tokenize=True)
        print('Done building vocabulary for InferSent in {} seconds'.format(int((datetime.now() - start).total_seconds())))

        start = datetime.now()
        vectors = self.model.encode(sentence_list, bsize=self.bsize, tokenize=True)
        print('Done InferSent encode in {} seconds'.format(int((datetime.now() - start).total_seconds())))
        return vectors

    def add_vectors(self, events: pd.DataFrame, use_phrases=False):
        events['vector'] = self.get_vectors(list(events['sentence'].values), use_phrases)
        return events


if __name__ == "__main__":
    infer_sent_vectors_helper = InferSentVectorsHelper(model_path='../infersent/infersent2.pkl',
                                                       path_to_w2v='../fasttext/crawl-300d-2M.vec',
                                                       use_cuda=True)

    # v = infer_sent_vectors_helper.get_vectors(['Revenue for the year ended December 31, 2017 increased $420.5 million compared to the same period in 2016.',
    #                                            'Cost of revenue for the year ended December 31, 2016 increased $269.3 million, or 148%, compared to the same period in 2015.'])
    # print(v)

    result = {}
    with io.open('../data/senteval_data/downstream/AMENITY/suggestions_labeled_data_with_seed_random_order_train.txt', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i > 1:
                text = line.strip().split('\t')
                id_a = text[1]
                id_b = text[2]
                sentence_a = text[3]
                sentence_b = text[4]
                result[id_a] = {'sentence': sentence_a}
                result[id_b] = {'sentence': sentence_b}

    sentences_list = [sentence['sentence'] for _, sentence in result.items()]
    vectors = infer_sent_vectors_helper.get_vectors(sentences_list)
    for sentence, vector in zip(sentences_list, vectors):
        id_dict = {id: sentence for id, sentence in result.items() if sentence == sentence}
        for id in id_dict:
            result[id]['vector'] = vector

    import numpy as np
    import json

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    with open('amenity_train_infersent_vectors.json', 'w') as out_file:
        json.dump(result, out_file, cls=NumpyEncoder)



