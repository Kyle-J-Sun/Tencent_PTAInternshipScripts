# '''----------------------------------------------------Title------------------------------------------------------------
#
# ---------------------------------------------------------------------------------------------------------------------'''
# # import pkuseg
# # seg = pkuseg.pkuseg()
# # text = seg.cut("我是孙旌凯，我是一名学生")
# # # print(text)
# #
# import jieba
# import jieba.posseg as pseg
# import jieba.analyse as anls
# #
# # # 全模式
# # seg_list = jieba.lcut("我是孙旌凯，我是一名学生", cut_all=True)
# # print(seg_list)
# # #精确模式
# # seg_list2 = jieba.lcut("我是孙旌凯，我是一名学生", cut_all = False)
# # print(seg_list2)
# # #搜索引擎模式
# # seg_list3 = jieba.lcut_for_search("我是孙旌凯，我是一名学生")
# # print(seg_list3)
# #
# # import string
# # print(string.whitespace)
#
# # s = "g5高速"
# # seg = anls.extract_tags(s, topK=20, withWeight=True)
# # print(seg)
# #
# import numpy as np
# # np.vectorize
# from scipy import stats
# list1 = ['出行', '推迟']
# list2 = ['爸爸', '国语', 'HHH', 'HHH', '国语版']
# title = 'G5京昆！'
# # #
# # def match_density(list1, list2, density_threthold = 1):
# #     match_index = []
# #     count = 0
# #     for j in list2 :
# #         for i in list1:
# #             if j == i:
# #                 match_index.append(list2.index(i))
# #                 count += 1
# #     match_density = []
# #     for i in range(len(match_index)):
# #         try:
# #             if match_index[i+1] - match_index[i] <= density_threthold:
# #                 match_density.append(1)
# #             else:
# #                 match_density.append(0)
# #         except:
# #             break
# #     if match_density != [] :
# #     # return max(set(match_density), key=match_density.count)  # get mode
# #         return sum(match_density) * 1. / len(match_density)
# #     else :
# #         return 0
# #
# # print(match_density(list1, list2, 1))
#
#
# # def match_order_score(list1, list2):
# #     words_matched = []
# #     for j in list2:
# #         for i in list1:
# #             if j == i:
# #                 words_matched.append(j)
# #     print(words_matched)
# #     checkpoint1 = 0
# #     checkpoint2 = []
# #     from itertools import combinations, product
# #     checkpoint1 = list(combinations(list1, 2))
# #     for i in range(len(words_matched)):
# #         try:
# #             checkpoint2.append(tuple([words_matched[i], words_matched[i + 1]]))
# #         except:
# #             break
# #
# #     matched_order = []
# #     for s in checkpoint1:
# #         for l in checkpoint2:
# #             if s == l:
# #                 matched_order.append(1)
# #             else:
# #                 matched_order.append(0)
# #     if matched_order == []:
# #         return 0
# #     else:
# #         return (matched_order.count(1) * 1. / len(matched_order)) * 10
# # print(match_order_score(list1, list2))
#
# # from itertools import combinations
# # l = [1,2,3,4,5]
# # print(list(combinations(l, 2)))
#
#
#
# # for o in list1:
# #     try:
# #         if match_order.index(o) == list1.index(o):
# #             final.append(1)
# #         else:
# #             final.append(0)
# #     except:
# #         break
# # if all(final) == True:
# #     order = 1
# # else:
# #     order = 0
# #     # return order
#
# # important_words = ['I', 'All', 'kids', 'cried', 'Kyle']
# # review = "All of my kids have cried nonstop when I tried to"
# #
# # important_words = [str(s) for s in important_words]
# #
# # a = []
# # for word in important_words:
# #     a.append(review.count(word))
# # print(a)
#
# # s = 'strong'
# # print(len(s))
# # import random
# # import string
# # letters = string.ascii_letters
# # print(random.choice(letters))
#
# # list1 = ['赤脚', '医生', '赤脚医生']
# # title = '邯郸冀南新区赤脚医生补助名单正在审核中'
# #
# # import jieba.analyse as anlys
# # def get_tfidf(list, sentence):
# #     store = anlys.extract_tags(sentence, topK= None, withWeight= True, allowPOS=(), withFlag=False)
# #     TFIDF = []
# #     for i in store:
# #         TFIDF.append(i)
# #     words_with_tfidf = dict(TFIDF)
# #     print(words_with_tfidf)
# #
# #     words_weight = []
# #     for i in list:
# #         for j in words_with_tfidf.keys():
# #             if i == j:
# #                 words_weight.append(words_with_tfidf[j])
# #     return sum(words_weight)
# #
# # print(get_tfidf(list1,title))
# #
# #
# #
# # def get_textrank(list, sentence):
# #     store = anls.textrank(sentence, topK=None, withWeight = True, allowPOS = ('ns', 'nl' , 'ng', 'vx','n', 'v', 'vn', 'vd', 'nr1', 'nr2'),withFlag=False)
# #     textrank = []
# #     for i in store:
# #         textrank.append(i)
# #     words_with_textrank = dict(textrank)
# #
# #     words_rank_weight = []
# #     for i in list:
# #         for j in words_with_textrank.keys():
# #             if i == j:
# #                 words_rank_weight.append(words_with_textrank[j])
# #     return sum(words_rank_weight)
# # #
# print(get_textrank(list1,title))

import csv2es as c2e
help(c2e)
