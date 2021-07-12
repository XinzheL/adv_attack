def load_data(task, split, MODEL_TYPE=None, max_length=None, min_length=None, 
                sample=0, distributed=False, num_worker=0):
     
    """ load data with its tokenizers and indexers
    Args:
        MODEL_TYPE: used to indicate whether using pretrained tokenizer and indexer
        max_length: for imdb
        min_length: 
        sample: for imdb
    """
    from allennlp_models.classification.dataset_readers.stanford_sentiment_tree_bank import StanfordSentimentTreeBankDatasetReader
    from allennlp.data.token_indexers import SingleIdTokenIndexer
    from allennlp.data.token_indexers import PretrainedTransformerIndexer
    from allennlp.data.tokenizers import PretrainedTransformerTokenizer, WhitespaceTokenizer
    from allennlp.data.data_loaders import MultiProcessDataLoader

    from my_library.dataset_readers import ImdbDatasetReader, AmazonDatasetReader, YelpDatasetReader
  
    
    
    if 'bert' in MODEL_TYPE:
        tokenizer = PretrainedTransformerTokenizer(model_name=MODEL_TYPE, add_special_tokens=True, max_length=max_length)
        indexer = PretrainedTransformerIndexer(model_name=MODEL_TYPE) 
    else:
        if task == 'bi_sst':
            
            tokenizer = None
        elif task == 'imdb' or 'amazon' or "yelp":
            tokenizer = WhitespaceTokenizer()
        indexer = SingleIdTokenIndexer(lowercase_tokens=False) # word tokenizer
    
    if distributed :
        manual_distributed_sharding = True
        manual_multiprocess_sharding = True
    else:
        manual_distributed_sharding = False
        manual_multiprocess_sharding = False

    if task == 'bi_sst':
        granularity = '2-class'
        reader = StanfordSentimentTreeBankDatasetReader(granularity=granularity, 
                                        tokenizer=tokenizer, token_indexers={"tokens": indexer},
                                        manual_distributed_sharding = manual_distributed_sharding,
                                        manual_multiprocess_sharding = manual_multiprocess_sharding,
                                        ) #distributed=distributed
    elif task == 'imdb':
        reader = ImdbDatasetReader(tokenizer=tokenizer, token_indexers={"tokens": indexer}, 
                                        sample_files=sample, min_len=min_length,
                                        manual_distributed_sharding = manual_distributed_sharding,
                                        manual_multiprocess_sharding = manual_multiprocess_sharding,
                                        distributed=distributed)
    elif task == 'amazon':
        reader = AmazonDatasetReader(tokenizer=tokenizer, token_indexers={"tokens": indexer}, 
                                        sample=sample, min_len=min_length,
                                        manual_distributed_sharding = manual_distributed_sharding,
                                        manual_multiprocess_sharding = manual_multiprocess_sharding,
                                        distributed=distributed)
    elif task == 'yelp':
        reader = YelpDatasetReader(tokenizer=tokenizer, token_indexers={"tokens": indexer}, 
                                        sample=sample, min_len=min_length,
                                        manual_distributed_sharding = manual_distributed_sharding,
                                        manual_multiprocess_sharding = manual_multiprocess_sharding,
                                        distributed=distributed)
                
    
    # load file
    if task == 'bi_sst':
        assert split in ['dev', 'train', 'test']
        file_path = f'https://s3-us-west-2.amazonaws.com/allennlp/datasets/sst/{split}.txt'
    elif task == 'imdb' or 'amazon' or 'yelp':
        assert split in [ 'train', 'test']
        file_path = split
    

    dataloader = MultiProcessDataLoader(reader, file_path, batch_size=2, max_instances_in_memory=10, num_workers=num_worker)
    
    
    return dataloader