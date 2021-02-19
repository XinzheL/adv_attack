

from utils.allennlp_model import load_sst_model
from utils.allennlp_predictor import AttackPredictorForBiClassification

from allennlp.interpret.attackers import Hotflip
from utils.universal_attack import UniversalAttack
import os
from copy import deepcopy

label_filter = 1
MODELS_DIR = 'checkpoints/five_way_sst/'
sst_granularity = 5
MODEL_TYPE = 'lstm' 


if MODEL_TYPE == 'finetuned_bert':
    READER_TYPE= 'pretrained' 
    pretrained_model = 'bert-base-uncased'  
    EMBEDDING_TYPE = None 

elif MODEL_TYPE =='lstm' or MODEL_TYPE=='cnn' :
    READER_TYPE= None 
    pretrained_model = None 
    EMBEDDING_TYPE = None # "w2v" 

from utils.allennlp_data import load_sst_data
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
elif MODEL_TYPE == 'lstm' or MODEL_TYPE=='cnn':
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

# save triggers and their transferability (accuracy on other models)
triggers = []
accs = {}
MODELS = {}
for M in os.listdir(MODELS_DIR):
    if M != MODEL_TYPE:
        if M == 'finetuned_bert':
            READER_TYPE= 'pretrained' # None # 
            pretrained_model = 'bert-base-uncased' # None # 
            EMBEDDING_TYPE = None # "w2v" # 

        elif M =='lstm' :
            READER_TYPE= None 
            pretrained_model = None 
            EMBEDDING_TYPE = "w2v" 

        # Load Model
        vocab, model = load_sst_model(f"{MODELS_DIR}{M}/",  MODEL_TYPE=M)

        _, test_data = load_sst_data('test',\
                                        READER_TYPE=READER_TYPE, \
                                        pretrained_model = pretrained_model,
                                        granularity = str(sst_granularity)+'-class')

        MODELS[M] = (deepcopy(model), deepcopy(vocab), deepcopy(list(test_data)))

for t in log_trigger_tokens:
    triggers.append(str(t[0]) + '_' + str(t[1]) + '_' + str(t[2]))

    # evaluate the transferability of other models
    #if t == log_trigger_tokens[-1]: # only evaluate on the last one
    for M in MODELS.keys():
        model, vocab, test_data = MODELS[M]
        #Evaluate with test data appended by triggers
        noisy_test_data = UniversalAttack.prepend_batch(
            UniversalAttack.filter_instances( \
                list(test_data), label_filter=label_filter, vocab=vocab
            ), \
            trigger_tokens=t, \
            vocab = vocab
        )

        acc, _ = UniversalAttack.evaluate_instances_cls(noisy_test_data, model, vocab, cuda_device=0)
        if M not in accs.keys():
            accs[M] = [acc]
        else:
            accs[M].append(acc)
            


result_df['triggers'] = triggers
for M in accs.keys():
    result_df[f'{M}_accuracy'] = accs[M]

# result_long_df = pd.melt(result_df , ['iteration'])
result_df.to_csv(f'result_data/{MODEL_TYPE}_{str(label_filter)}.csv')




    