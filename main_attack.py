
from utils.allennlp_model import load_sst_model, load_all_trained_models
from utils.allennlp_predictor import AttackPredictorForBiClassification
from utils.allennlp_data import load_sst_data
from allennlp.interpret.attackers import Hotflip
from utils.universal_attack import UniversalAttack
import os
from copy import deepcopy

def attack(label_filter, MODEL_TYPE, sst_granularity = 2):
    if sst_granularity == 2:
        MODELS_DIR = 'checkpoints/bi_sst/'
    elif sst_granularity == 5:
        MODELS_DIR = 'checkpoints/five_way_sst/'



    if 'finetuned_bert' in MODEL_TYPE:
        READER_TYPE= 'pretrained' 
        pretrained_model = 'bert-base-uncased'  
        EMBEDDING_TYPE = None 

    elif  'lstm' in MODEL_TYPE or 'cnn' in MODEL_TYPE :
        READER_TYPE= None 
        pretrained_model = None 
        EMBEDDING_TYPE = None 
        if 'w2v' in MODEL_TYPE:
            EMBEDDING_TYPE =  "w2v" 

   
    datareader, dev_data = load_sst_data('dev',\
            READER_TYPE=READER_TYPE, \
            pretrained_model = pretrained_model,
            granularity = str(sst_granularity)+'-class')
    _, test_data = load_sst_data('test',\
            READER_TYPE=READER_TYPE, \
            pretrained_model = pretrained_model,
            granularity = str(sst_granularity)+'-class')

        
    # load data and model
    vocab, model = load_sst_model(f"{MODELS_DIR}{MODEL_TYPE}/",  MODEL_TYPE=MODEL_TYPE)

    if MODEL_TYPE == 'finetuned_bert':
        vocab_namespace='tags'
    elif 'lstm' in MODEL_TYPE or 'cnn' in MODEL_TYPE:
        vocab_namespace='tokens'


    # predictor
    predictor = AttackPredictorForBiClassification(model, datareader)


    def non_target_attack(predictor, instances):
        """
        Args:
        dev_data [list(Instance)]
        """
        hotflip = Hotflip(predictor)
        # dev_dict = {}
        attack_output = []
        for instance in instances:
            # dev_dict['tokens'] = instance.fields['tokens']
            attack_output.append(hotflip.attack_from_json(instance.fields))

            
    def universal_attack(predictor, instances, test_data, vocab_namespace='tokens'):
        universal = UniversalAttack(predictor)
        loss_lst, metrics_lst, log_trigger_tokens = universal.attack_instances(instances, test_data=test_data, num_epoch=4, vocab_namespace=vocab_namespace, label_filter=label_filter)
        return loss_lst, metrics_lst, log_trigger_tokens



    # non_target_attack(predictor, instances)
    loss_lst, metrics_lst, log_trigger_tokens = universal_attack(predictor, list(dev_data), test_data=list(test_data), vocab_namespace=vocab_namespace)
    # save the result
    import pandas as pd
    result_df = pd.DataFrame({"accuracy": [ele for lst in metrics_lst for ele in lst], \
        "loss":  [ele for lst in loss_lst for ele in lst], \
        "iteration": range(len([ele for lst in loss_lst for ele in lst]))})

    # save triggers
    # t=log_trigger_tokens[i] -> [the, the, the] where type(the) is `Token`
    
    result_df['triggers'] = [str(t[0]) + '_' + str(t[1]) + '_' + str(t[2]) for t in log_trigger_tokens]

    result_df.to_csv(f'result_data/{MODEL_TYPE}_{str(label_filter)}.csv')




if __name__ == "__main__":
    MODEL_TYPES = ['cnn_tanh'] #   , 'finetuned_bert', 'lstm', 'lstm_w2v', 'cnn', 'cnn_w2v'
    LABELS = [0, 1]
    for MODEL_TYPE in MODEL_TYPES:
        for label_filter in LABELS:
            attack(label_filter = label_filter, MODEL_TYPE = MODEL_TYPE, sst_granularity = 2 )
    
    
    




    