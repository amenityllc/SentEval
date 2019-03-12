from bert_embeddings import BertEmbeddings, AbstractBertPooller


class TFIDFAttentivePooller(AbstractBertPooller):
    def __init__(self):
        super().__init__()

    def get_name(self):
        return 'tfidf_attentive_pooler'

    def pool(self, results):
        return [result['vector'] for result in results]


if __name__ == "__main__":
    bert_embeddings = BertEmbeddings(TFIDFAttentivePooller)
    bert_embeddings.run()
