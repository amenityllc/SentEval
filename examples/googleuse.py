# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division

import os
import sys
import logging
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
tf.logging.set_verbosity(0)

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data/senteval_data'
PATH_TO_RESULTS = '../results'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# tensorflow session
session = tf.Session()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# SentEval prepare and batcher
def prepare(params, samples):
    return

def batcher(params, batch):
    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
    embeddings = params['google_use'](batch)
    return embeddings

def make_embed_fn(module):
  with tf.Graph().as_default():
    sentences = tf.placeholder(tf.string)
    embed = hub.Module(module)
    embeddings = embed(sentences)
    session = tf.train.MonitoredSession()
  return lambda x: session.run(embeddings, {sentences: x})

# Start TF session and load Google Universal Sentence Encoder
encoder = make_embed_fn("https://tfhub.dev/google/universal-sentence-encoder-large/3")

# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}
params_senteval['google_use'] = encoder

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark']
    results = se.eval(transfer_tasks)
    print(results)

    if not os.path.exists(PATH_TO_RESULTS):
        os.mkdir(PATH_TO_RESULTS)

    with open(os.path.join(PATH_TO_RESULTS, 'googleuse.json'), 'w') as out_file:
        json.dump(results, out_file, cls=NumpyEncoder)
