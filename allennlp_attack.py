from utils.allennlp_model import train_sst_model

train_sst_model(output_dir="output")

from utils.allennlp_data import load_sst_data
from utils.allennlp_model import load_sst_model
from utils.allennlp_predictor import AttackPredictorForBiClassification

from allennlp.interpret.attackers import Hotflip
from utils.universal_attack import UniversalAttack




# load data and model
datareader, data_generator = load_sst_data('dev') 
vocab, model = load_sst_model("checkpoints/sst/" )

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

        
def universal_attack(predictor, instances):
    universal = UniversalAttack(predictor)
    loss_lst, metrics_lst, log_trigger_tokens = universal.attack_instances(instances, num_epoch=4)
    return loss_lst, metrics_lst, log_trigger_tokens


instances = list(data_generator)
# non_target_attack(predictor, instances)
loss_lst, metrics_lst, log_trigger_tokens = universal_attack(predictor, instances)

eric_result_0 = [0.831,0.841, 0.121, 0.086] # neg(0) to pos(1): [the], [captivating, captivating, captivating] , [vividly, georgian-israeli, captivating]
eric_result_1 = [0.8761261261261262, 0.1373873873873874, 0.04504504504504504, 0.018018018018018018, 0.015765765765765764, 0.015765765765765764, 0.015765765765765764, 0.015765765765765764, 0.015765765765765764, 0.015765765765765764, 0.015765765765765764, 0.015765765765765764, 0.015765765765765764] # pos(1) to neg(0)
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

result_df = pd.DataFrame({"accuracy": [ele for lst in metrics_lst for ele in lst], "loss":  [ele for lst in loss_lst for ele in lst], "iteration": range(len([ele for lst in loss_lst for ele in lst]))})
# result_long_df = pd.melt(result_df , ['iteration'])
axes = sns.lineplot(data=result_df, x='iteration', y='accuracy', label='accuracy: neg->pos', color='teal')
plt.plot(eric_result_0, color='blue', linestyle='-', linewidth=2, alpha=0.3, label='accuracy (Wallace, 2020)')
axes.set_xticks(np.linspace(min(result_df.index), max(result_df.index), num=5, dtype=int))
axes.set_xticklabels(np.linspace(min(result_df.index)+1, max(result_df.index)+1, num=5 , dtype=int))
axes2 = axes.twinx()
sns.lineplot(data=result_df, x='iteration', y='loss', label='loss: neg->pos', ax=axes2, color='red')
axes.legend(loc="lower right")
axes2.legend(loc="upper right")
plt.title('Universal Perturbation for LSTM-based Model with First-order Approximation')
plt.show()




    