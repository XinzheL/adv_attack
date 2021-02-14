import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas

MODELS_TO_CHOOSE = ['lstm_w2v', 'finetuned_bert']

fig = plt.figure(figsize=(16, 6), dpi=100)
# axe for accuracy
ax1 = fig.add_subplot(1,1,1)
# ax1.set_xticks(np.linspace(min(result_df.index), max(result_df.index), num=5, dtype=int))
#     ax1.set_xticklabels(np.linspace(min(result_df.index)+1, max(result_df.index)+1, num=5 , dtype=int))
# axe for loss
ax2 = ax1.twinx()

COLORS_FOR_METRICS = ['teal', 'blue']
COLORS_FOR_LOSS = ['red', 'yellow']
ALPHAS = np.linspace(0.1, 1, num=len(MODELS_TO_CHOOSE), dtype=float)
for i, MODEL in enumerate(MODELS_TO_CHOOSE):
    result_df = pandas.read_csv(f'result_data/{MODEL}.csv')
    # eric_result_0 = [0.831,0.841, 0.121, 0.086] # neg(0) to pos(1): [the], [captivating, captivating, captivating] , [vividly, georgian-israeli, captivating]
    # eric_result_1 = [0.8761261261261262, 0.1373873873873874, 0.04504504504504504, 0.018018018018018018, 0.015765765765765764, 0.015765765765765764, 0.015765765765765764, 0.015765765765765764, 0.015765765765765764, 0.015765765765765764, 0.015765765765765764, 0.015765765765765764, 0.015765765765765764] # pos(1) to neg(0)
    
    sns.lineplot(data=result_df, x='iteration', y='accuracy', label=f'accuracy-{MODEL}', color='teal', alpha=ALPHAS[i], ax=ax1, linestyle='-', linewidth=2,)
    
    
    sns.lineplot(data=result_df, x='iteration', y='loss', label=f'loss-{MODEL}', color='red', alpha=ALPHAS[i], ax=ax2, linestyle='-', linewidth=2)

ax1.legend(loc="lower right")
ax2.legend(loc="upper right")
plt.title('Universal Perturbation(neg->pos) with First-order Approximation')
plt.show()
