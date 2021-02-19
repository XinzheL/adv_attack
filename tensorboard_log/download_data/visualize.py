import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas
import os

dir_name = 'train' # 'validation' #


fig = plt.figure(figsize=(16, 6), dpi=100)
# axe for accuracy
ax1 = fig.add_subplot(1,1,1)
COLORS_FOR_METRICS = ['teal', 'orange', '#e87a59']
TRAIN_TYPES = ['standard', 'error_max', 'error_min']
for i, TRAIN_TYPE in enumerate(TRAIN_TYPES):
    if dir_name == 'train':
        if TRAIN_TYPE == 'standard':
            
            file_name = 'run-finetuned_bert_train-tag-epoch_metrics_accuracy.csv'
        else:
            file_name = 'run-finetuned_bert_' + TRAIN_TYPE + '_train-tag-epoch_metrics_accuracy.csv'
    elif dir_name == 'validation':
        if TRAIN_TYPE == 'standard':
            file_name = 'run-finetuned_bert_validation-tag-accuracy.csv'
        else:
            file_name = 'run-finetuned_bert_' + TRAIN_TYPE + '_validation-tag-accuracy.csv'

    if os.path.exists(f'{dir_name}/{file_name}'):
        result_df = pandas.read_csv(f'{dir_name}/{file_name}')
        print(result_df['Value'])
    else:
        print("Invalid File Path.")
        exit()    
    sns.lineplot(data=result_df, x='Step', y='Value', label=TRAIN_TYPE, color=COLORS_FOR_METRICS[i], ax=ax1, linestyle='-', linewidth=2,)
    
    



if dir_name == 'validation':
    ax1.set_xlabel('Epoch')
    ax1.set_xticks((1.0, 2.0, 3.0)) 
    ax1.set_xticklabels([1,2,3])
    ax1.legend(loc="center right")
elif dir_name == 'train':
    ax1.set_xlabel('Iteration')
    ax1.legend(loc="lower right")
ax1.set_ylabel('Accuracy')


plt.show()
