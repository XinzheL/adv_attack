from torch.utils.data import Dataset
from typing import List, Iterator
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
class AllennlpDataset(Dataset):
    """
    An `AllennlpDataset` is created by calling `.read()` on a non-lazy `DatasetReader`.
    It's essentially just a thin wrapper around a list of instances.
    """

    def __init__(self, instances: List[Instance], vocab: Vocabulary = None):
        self.instances = instances
        self.vocab = vocab

    def __getitem__(self, idx) -> Instance:
        if self.vocab is not None:
            self.instances[idx].index_fields(self.vocab)
        return self.instances[idx]

    def __len__(self):
        return len(self.instances)

    def __iter__(self) -> Iterator[Instance]:
        """
        Even though it's not necessary to implement this because Python can infer
        this method from `__len__` and `__getitem__`, this helps with type-checking
        since `AllennlpDataset` can be considered an `Iterable[Instance]`.
        """
        yield from self.instances

    def index_with(self, vocab: Vocabulary):
        self.vocab = vocab

def collate_fn(batch):
    from allennlp.data import Batch
    batch = Batch(batch)
    return batch.as_tensor_dict(batch.get_padding_lengths())

            
def load_sst_data(split, READER_TYPE='None', pretrained_model = 'bert-base-uncased'):
    

    from allennlp_models.classification.dataset_readers.stanford_sentiment_tree_bank import StanfordSentimentTreeBankDatasetReader
    from allennlp.data.token_indexers import SingleIdTokenIndexer
    from allennlp.data.token_indexers import PretrainedTransformerIndexer
    from allennlp.data.tokenizers import PretrainedTransformerTokenizer
    
    # setup reader
    assert split in ['dev', 'train', 'test']
    file_path = f'https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/{split}.txt'
    
    if READER_TYPE == 'pretrained':
        #tokenizer = PretrainedTransformerTokenizer(model_name=pretrained_model, add_special_tokens=False)
        indexer = PretrainedTransformerIndexer(model_name=pretrained_model,) 
        tokenizer = indexer._allennlp_tokenizer
        reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class", tokenizer=tokenizer, token_indexers={"tokens": indexer})
    else: # READER_TYPE is None:
        indexer = SingleIdTokenIndexer(lowercase_tokens=True) # word tokenizer
        reader = StanfordSentimentTreeBankDatasetReader(granularity="2-class", token_indexers={"tokens": indexer})
    # load file
    instance_generator = reader.read(file_path) # return list of Instance object
    
    
    return reader, instance_generator


def load_snli_data(split, bert_model = 'bert-base-uncased'):
    from bert_snli import BertSnliReader
    from allennlp.data.token_indexers import PretrainedTransformerIndexer
    from allennlp.data.tokenizers import PretrainedTransformerTokenizer
    assert split in ['dev', 'train', 'test']
    
    # setup reader
    tokenizer = PretrainedTransformerTokenizer(model_name=bert_model, add_special_tokens=False)
    indexer = PretrainedTransformerIndexer(model_name=bert_model) 
    reader = BertSnliReader( tokenizer=tokenizer, token_indexers={"tokens": indexer})
    # import file
    file_path = f'https://allennlp.s3.amazonaws.com/datasets/snli/snli_1.0_{split}.jsonl'
    instance_generator = reader.read(file_path) # return list of Instance object
    
    
    return reader, instance_generator


