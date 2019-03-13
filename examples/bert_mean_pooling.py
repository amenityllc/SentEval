# from examples.bert_embeddings import BertEmbeddings, AbstractBertPooller
from bert_embeddings import BertEmbeddings, AbstractBertPooller
import numpy as np


class MeanPooller(AbstractBertPooller):
    def __init__(self):
        super().__init__()

    def get_name(self):
        return 'mean_pooler'

    def pool(self, results):
        embeddings = [result['embedding'] for result in results]
        tokens_list = [result['tokens'] for result in results]
        embeddings = [embedding[1: len(tokens) - 1] for embedding, tokens in zip(embeddings, tokens_list)]

        return np.array([np.mean(sentence_vectors, axis=0) for sentence_vectors in embeddings])


if __name__ == "__main__":
    bert_embeddings = BertEmbeddings(MeanPooller)
    bert_embeddings.run()
