import sys
import io
import os
import json
import numpy as np
import logging
import requests
from urllib.parse import quote
from abc import ABC, abstractmethod

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data/senteval_data'
PATH_TO_RESULTS = '../results'

# BERT_URL = 'http://localhost:5000/bert/embeddings?usebasic=true&sentences='
BERT_URL = 'http://192.168.1.65:5000/bert/embeddings?usebasic=true&sentences='

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


# SentEval prepare and batcher
def prepare(params, samples):
    return


def get_bert_embeddings(batch):
    query = '|'.join(batch)
    query = quote(query)
    url = BERT_URL + query
    return requests.get(url).json()


class AbstractBertPooller(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_name(self):
        raise NotImplementedError()

    @abstractmethod
    def pool(self, results):
        raise NotImplementedError()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class BertEmbeddings:
    def __init__(self, bert_pooler, max_seq_size=64):
        self.pooller = bert_pooler
        self.max_seq_size = max_seq_size

    def batcher(self, params, batch):
        batch = [sent if sent != [] else ['.'] for sent in batch]
        batch = [' '.join(sent) for sent in batch]
        batch = [sent.replace('|', ' ') for sent in batch]

        results = get_bert_embeddings(batch)
        embeddings = self.pooller.pool(self.pooller, results)

        embeddings = np.vstack(embeddings)
        return embeddings

    def run(self):
        # Set params for SentEval
        params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5,
                           'classifier': {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                          'tenacity': 3, 'epoch_size': 2}}

        se = senteval.engine.SE(params_senteval, self.batcher, prepare)
        transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                          'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                          'SICKEntailment', 'SICKRelatedness', 'STSBenchmark']
                          # 'Length', 'WordContent', 'Depth', 'TopConstituents',
                          # 'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                          # 'OddManOut', 'CoordinationInversion']
        results = se.eval(transfer_tasks)
        print(results)

        if not os.path.exists(PATH_TO_RESULTS):
            os.mkdir(PATH_TO_RESULTS)

        with open(os.path.join(PATH_TO_RESULTS, 'bert_' + self.pooller.get_name(self.pooller) + '.json'), 'w') as out_file:
            json.dump(results, out_file, cls=NumpyEncoder)
