import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas

result_df = pandas.read_csv('bert_result.csv')
eric_result_0 = [0.831,0.841, 0.121, 0.086] # neg(0) to pos(1): [the], [captivating, captivating, captivating] , [vividly, georgian-israeli, captivating]
eric_result_1 = [0.8761261261261262, 0.1373873873873874, 0.04504504504504504, 0.018018018018018018, 0.015765765765765764, 0.015765765765765764, 0.015765765765765764, 0.015765765765765764, 0.015765765765765764, 0.015765765765765764, 0.015765765765765764, 0.015765765765765764, 0.015765765765765764] # pos(1) to neg(0)
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
