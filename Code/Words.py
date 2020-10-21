'''----------------------------------------------------Title------------------------------------------------------------
1, 给qid和title列分别作分词，并count出现在各自列的出现频率，作为特征1和特征2
2, 计算qid切词在title中出现的频率比，作为特征3
3, 使用决策树模型和3个特征训练模型
---------------------------------------------------------------------------------------------------------------------'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, roc_auc_score, precision_score
pd.options.display.max_columns = 999

#loading the Boston Dataset
data = pd.read_csv('/Users/kyle/Documents/Virtual Intern/Tencent/data/words.csv', low_memory=False)
df = pd.DataFrame(data)
print(df)
df0, df1, df2 ,df3, df4, df5 = \
    df[df['label'] == 0], \
    df[df['label'] == 1], \
    df[df['label'] == 2], \
    df[df['label'] == 3], \
    df[df['label'] == 4], \
    df[df['label'] == 5]

df_words, df0_words, df1_words, df2_words, df3_words, df4_words, df5_words = \
    np.unique(df['qid']), \
    np.unique(df0['qid']), \
    np.unique(df1['qid']), \
    np.unique(df2['qid']), \
    np.unique(df3['qid']), \
    np.unique(df4['qid']), \
    np.unique(df5['qid'])

#Feature Engineering
# print(df_words[:10])
import string
def remove_punctuation(text):
    try: # python 2.x
        text = text.translate(None, string.punctuation)
    except: # python 3.x
        translator = text.maketrans('', '', '，？！《》.。、“”-+*/：……（）@:"＂,’‘— ')
        text = text.translate(translator)
    return text

qid_without_punc = []
title_without_punc = []
# Remove punctuation.
for i in range(len(df_words)):
    qid_without_punc.append(remove_punctuation(df_words[i]))
qid_without_punc = pd.DataFrame(qid_without_punc).rename(columns={0:"qid"})
qid_without_punc['qid'] = qid_without_punc['qid'].str.strip()
qid_without_punc = qid_without_punc['qid'].replace(' ', '')
# print(qid_without_punc)

for i in range(len(df['title'])):
    title_without_punc.append(remove_punctuation((df['title'][i])))
title_without_punc = pd.DataFrame(title_without_punc).rename(columns={0:"title"})
title_without_punc['title'] = title_without_punc['title'].str.strip()
title_without_punc = title_without_punc['title'].replace(' ', '')
# print(title_without_punc)

# 给qid作切词
import jieba
qid_cut_words = []
print(len(qid_without_punc))
for i in range(len(qid_without_punc)):
    cut_words = jieba.cut(qid_without_punc[i], cut_all=False)
    for j in cut_words:
        qid_cut_words.append(j)
qid_cut_words = pd.DataFrame(qid_cut_words).rename(columns={0:"title"})
qid_cut_unique = pd.DataFrame(pd.unique(qid_cut_words['title']))
print(qid_cut_unique)

# 给title作切词
title_cut_words = []
print(len(title_without_punc))
for i in range(len(title_without_punc)):
    cut_words2 = jieba.cut(title_without_punc[i], cut_all=False)
    for j in cut_words2:
        title_cut_words.append(j)
title_cut_words = pd.DataFrame(title_cut_words).rename(columns={0:"title"})
title_cut_unique = pd.DataFrame(pd.unique(title_cut_words['title']))
print(title_cut_unique)

