# from utils.allennlp_model import train_sst_model
# train_sst_model(output_dir="output/")
# exit()

from utils.allennlp_data import load_sst_data
from utils.allennlp_model import load_sst_model
from utils.allennlp_predictor import AttackPredictorForBiClassification

from allennlp.interpret.attackers import Hotflip
from utils.universal_attack import UniversalAttack


CHOOSE_MODEL = 'lstm_w2v' # 'finetuned_bert'

# load data and model
if CHOOSE_MODEL == 'finetuned_bert':
    datareader, data_generator = load_sst_data('dev', READER_TYPE='pretrained', pretrained_model = 'bert-base-uncased')
    vocab, model = load_sst_model("output/",  Model_TYPE='pretrained')

    vocab_namespace='tags'
elif CHOOSE_MODEL == 'lstm_w2v':
    datareader, data_generator = load_sst_data('dev', READER_TYPE=None)
    vocab, model = load_sst_model("checkpoints/sst/lstm_w2v/",  Model_TYPE=None)

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

        
def universal_attack(predictor, instances, vocab_namespace='tokens'):
    universal = UniversalAttack(predictor)
    loss_lst, metrics_lst, log_trigger_tokens = universal.attack_instances(instances, num_epoch=4, vocab_namespace=vocab_namespace)
    return loss_lst, metrics_lst, log_trigger_tokens


instances = list(data_generator)
# non_target_attack(predictor, instances)
loss_lst, metrics_lst, log_trigger_tokens = universal_attack(predictor, instances, vocab_namespace=vocab_namespace)
triggers = []
for t in log_trigger_tokens:
    triggers.append(str(t[0]) + '_' + str(t[1]) + '_' + str(t[2]))

# save the result
import pandas as pd
result_df = pd.DataFrame({"accuracy": [ele for lst in metrics_lst for ele in lst], \
    "loss":  [ele for lst in loss_lst for ele in lst], \
    "triggers": triggers, \
    "iteration": range(len([ele for lst in loss_lst for ele in lst]))})

# result_long_df = pd.melt(result_df , ['iteration'])
result_df.to_csv(f'result_data/{CHOOSE_MODEL}.csv')




    