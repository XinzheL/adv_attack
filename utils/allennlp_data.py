from torch.utils.data import Dataset


import math
import random
from typing import Optional, List, Iterator

from overrides import overrides
import torch

from allennlp.common.util import lazy_groups_of
from allennlp.data.data_loaders.data_loader import DataLoader, allennlp_collate, TensorDict
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.vocabulary import Vocabulary
import allennlp.nn.util as nn_util


@DataLoader.register("my_simple", constructor="from_dataset_reader")
class MySimpleDataLoader(DataLoader):
    """
    A very simple `DataLoader` that is mostly used for testing.
    """

    def __init__(
        self,
        instances: List[Instance],
        batch_size: int,
        *,
        shuffle: bool = False,
        batches_per_epoch: Optional[int] = None,
        vocab: Optional[Vocabulary] = None,
    ) -> None:
        self.instances = instances
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batches_per_epoch = batches_per_epoch
        self.vocab = vocab
        self.cuda_device: Optional[torch.device] = None
        self._batch_generator: Optional[Iterator[TensorDict]] = None

    def __len__(self) -> int:
        return math.ceil(len(self.instances) / self.batch_size)

    @overrides
    def __iter__(self) -> Iterator[TensorDict]:
        if self.batches_per_epoch is None:
            yield from self._iter_batches()
        else:
            if self._batch_generator is None:
                self._batch_generator = self._iter_batches()
            for i in range(self.batches_per_epoch):
                try:
                    yield next(self._batch_generator)
                except StopIteration:  # data_generator is exhausted
                    self._batch_generator = self._iter_batches()  # so refresh it
                    yield next(self._batch_generator)

    def _iter_batches(self) -> Iterator[TensorDict]:
        if self.shuffle:
            random.shuffle(self.instances)
        for batch in lazy_groups_of(self.iter_instances(), self.batch_size):
            yield batch
            # tensor_dict = allennlp_collate(batch)
            # if self.cuda_device is not None:
            #     tensor_dict = nn_util.move_to_device(tensor_dict, self.cuda_device)
            # yield tensor_dict

    @overrides
    def iter_instances(self) -> Iterator[Instance]:
        for instance in self.instances:
            if self.vocab is not None:
                instance.index_fields(self.vocab)
            yield instance

    @overrides
    def index_with(self, vocab: Vocabulary) -> None:
        self.vocab = vocab
        for instance in self.instances:
            instance.index_fields(self.vocab)

    @overrides
    def set_target_device(self, device: torch.device) -> None:
        self.cuda_device = device

    @classmethod
    def from_dataset_reader(
        cls,
        reader: DatasetReader,
        data_path: str,
        batch_size: int,
        shuffle: bool = False,
        batches_per_epoch: Optional[int] = None,
    ) -> "SimpleDataLoader":
        instances = list(reader.read(data_path))
        return cls(instances, batch_size, shuffle=shuffle, batches_per_epoch=batches_per_epoch)


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

            
def load_sst_data(split, READER_TYPE='None', pretrained_model = 'bert-base-uncased', granularity = '2-class'):
    

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
        reader = StanfordSentimentTreeBankDatasetReader(granularity=granularity, tokenizer=tokenizer, token_indexers={"tokens": indexer})
    else: # READER_TYPE is None:
        indexer = SingleIdTokenIndexer(lowercase_tokens=True) # word tokenizer
        reader = StanfordSentimentTreeBankDatasetReader(granularity=granularity, token_indexers={"tokens": indexer})
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


