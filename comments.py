# -*- coding: utf-8 -*-
"""
comment数据聚类
目标：实现单个音乐的comments集合

流程：
    1. 停词分词
    2. 建立词向量
        使用tfidf对文本的单词排序，保留排序前50
        建立生词表，对全部文本编码，制作 [0,1,1,1,0,0...]
    3. k-means
    4. 簇标签生成

结果：

    对文档进行聚类，形成文档的聚类。簇标签
"""

import nltk
from nltk.corpus import stopwords
import re
from sklearn.cluster import KMeans
import os
import pickle
import math
import numpy as np


# 读取路径下的全部文件,读取为一个词袋模型，key=filename，val=data

def readdata(path):
    file = os.walk(path)
    comment_dic = {}
    for path, dir_list, file_list in file:
        for file_name in file_list:
            if file_name[-3:] == "txt":
                comment_data = lrc2data(path, file_name)  # 获取list格式的数据
                print(file_name)
                # 在此处进行tfidf操作，val储存筛选后的结果
                val = tfidf(comment_data)
                comment_dic[file_name] = val
                print("finish", file_name)

    return comment_dic


def lrc2data(path, file_name):
    # 打开文件
    file = open(path + "/" + file_name, "r", encoding="utf-8")
    # 读取文件全部内容
    lrc_list = file.readlines()
    lrc_data = []

    ##stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    new_words = ['it', 'is', 'are', '...', 'my', 'you', 'this', 'that', 'when', 'http', 'https',
                 'song', 'songs', 'soundcloud.com']
    for w in new_words:
        stopwords.append(w)
    # stopwords.extend(new_words)
    ##

    # 遍历所有元素,干掉方括号
    for i in lrc_list:
        lrc_word = i.strip().split("]")
        # print(i,"str(lrc_word[-1])",str(lrc_word[-1]),"str(lrc_word[-1])")
        if str(lrc_word[-1]) != "":
            # 对于筛选出的句子，剔除掉全部非英文字符
            rstr = r"[\=\(\)\,\/\\\:\*\?\"\<\>\|\' ']"
            text = re.sub(rstr, " ", lrc_word[-1])
            token = nltk.word_tokenize(text)
            # print("token",token)
            filtered = [w.lower() for w in token if (w not in stopwords and len(w) > 1)]
            filtered = [w for w in filtered if w not in stopwords]
            # print(filtered)
            lrc_data.append(filtered)
    return lrc_data


# TF-IDF
# 输入：一个txt处理后的文档，[[1w,,3d,1r,1c],[21,d cs,qds,q],..]
# 输出：[], 保留前50个，可以小于50
def tfidf(data):
    #
    def tf(data):
        return 1

    def idf(data):
        res = []
        idfs = {}
        d = 0.0
        for doc in data:
            d += 1
            counted = []
            for word in doc:
                if not word in counted:
                    counted.append(word)
                    if word in idfs:
                        idfs[word] += 1
                    else:
                        idfs[word] = 1

        # 计算每个词逆文档值
        for word in idfs:
            idfs[word] = round(math.log(d / float(idfs[word])), 2)

            res = sorted(idfs.items(), key=lambda x: x[1])
            if len(res) >= 50:
                res = res[:50]
        return res

    return idf(data)


# 对 readdata 的dic输出，建立生词表，编码
def processdata(comment_dic):
    word_dic = {}
    allwords = []
    vals = comment_dic.values()
    for i in vals:
        for item in i:
            allwords.append(item[0])
    allwords = list(set(allwords))
    for i in range(len(allwords)):
        word_dic[allwords[i]] = i

    with open('commentdata.pkl', 'wb') as f:
        pickle.dump((comment_dic, word_dic), f)
    return word_dic


# 处理要输入到sklearn的数据
def input_data():
    try:  # 检测pkl是否存在

        f = open("commentdata.pkl", "rb")
        print("检测pkl是否存在,是")

        comment_dic, word_dic = pickle.load(f)
    except:
        print("检测pkl是否存在,否")
        comment_dic = readdata(path)
        print("finish comment_dic")
        word_dic = processdata(comment_dic)
    res = []
    # print(word_dic)
    keys = comment_dic.keys()  # 按照key值取出对应words
    for key in keys:
        val = comment_dic[key]
        temp = [0] * len(word_dic)
        for item in val:  # val格式为[('!', 2.48), ('song', 2.93)...]
            loc = word_dic[item[0]]

            temp[loc] = 1
        res.append(temp)
    # print(res)
    return np.array(res)


# kmeans
# 输入全部txt文档的tfidf处理结果，
#   格式为[[],[],...]
#   输出聚类结果kmeans.labels_


'''

from sklearn.cluster import KMeans
num_clusters=  3#聚为四类，可根据需要修改
cluster = KMeans(n_clusters=num_clusters,random_state=0).fit(X)
'''

# cluster keywords


'''
制作用于绘图的数据，
    以文档名称为单位，对应a值v值，以及cluster=2,4,8时的聚类结果，总计6列数据
'''

if __name__ == "__main__":
    path = './data/soundcloud'
    # readdata(path)
    # readdata(path)

    # file = os.walk(path)
    lrc_data = []
    file_name = "1.txt"
    # print('flag1')
    file = open(path + "/" + file_name, "r", encoding="utf-8")
    # print(path +"/"+ file_name)
    comment_list = file.readlines()
    for i in comment_list:
        lrc_word = i.strip().split(" ")
        # print(lrc_word)
        if str(lrc_word[-1]) != "":
            # 对于筛选出的句子，剔除掉全部非英文字符
            rstr = r"[\=\(\)\,\/\\\:\*\?\"\<\>\|\' ']"
            text = re.sub(rstr, " ", lrc_word[-1])
            token = nltk.word_tokenize(text)
            filtered = [w for w in token if w not in stopwords.words('english')]
            # print(token,filtered)

            lrc_data.append(filtered)
    # print(lrc_data)
    res = tfidf(lrc_data)
    # print(res)

    '''


    '''
    # comment_dic=readdata(path)
    # print(comment_dic)
    X = input_data()
    # print(len(X),len(X[0]))

    cluster = 8  # 设置聚类个数
    kmeans = KMeans(n_clusters=cluster, random_state=0).fit(X)
    print(kmeans.labels_)
    # 根据labels找出对应的文档，输出全部的tfidf值
    file = os.walk(path)
    for path, dir_list, file_list in file:
        file_name_list = file_list
    print(len(file_name_list), len(kmeans.labels_))  # 514

    cluster_res = [[], [], [], [], [], [], [], []]
    cluster_comment = [[], [], [], [], [], [], [], []]
    f = open("commentdata.pkl", "rb")
    comment_dic, word_dic = pickle.load(f)
    for i, item in enumerate(kmeans.labels_):  # 建立一个聚类结果的二重listh
        # print(i,item,cluster_res[item],"i,item++++++++++++++++")
        file_name = file_name_list[i]
        # print(file_name)
        cluster_res[item].append(file_name)
        # print("cluster_res",cluster_res)
        comment = []
        for content in comment_dic[file_name]:
            # print(content,"content")
            comment.append(content[0])
        # print("item",item,comment,cluster_comment[item])
        cluster_comment[item].append(comment)
        # print("cluster_comment",cluster_comment)
    print(cluster_res, "这个是聚类的文档名结果")

    res_for_plot = []
    for item in cluster_comment:
        # print(item)
        res_for_plot.append(tfidf(item))
    # print("res_for_plot",res_for_plot)
    # 根据comment_dic提取内容
    data = res_for_plot
    # print(len(word_dic))

    '''
    kmeans绘图操作
    1. 获取文档名，和聚类结果
    2. 根据文档名，获取a v
    文档名为file_name_list，聚类编号
    '''
    res = [file_name_list]
    annotation_dict = {}
    with open("./data/static_annotations.csv", "r") as f:
        annotation = f.readlines()
        for i in annotation[1:]:
            annotation_word = i.strip().split(",")
            key = annotation_word[0]
            val = [float(annotation_word[1]), float(annotation_word[2])]
            annotation_dict[key] = val
    # 根据file_name_list获取av值
    val_res = []
    for i in file_name_list:
        num = i[:-4]
        try:
            val = annotation_dict[num]
        except:
            val = [0, 0]
        val_res.append(val)
    labels = list(map(list, zip(*val_res)))
    print("检测labels", labels)
    res = res + labels

    clslist = [2, 8, 12]
    for cls in clslist:
        kmeans = KMeans(n_clusters=cls, random_state=0).fit(X)
        res.append(kmeans.labels_)

    # print("检测res",res)
    res_t = list(map(list, zip(*res)))
    print("检测res", res)
    import matplotlib.pyplot as plt

    x = res[1]
    y = res[2]
    color_set = ['#000000', '#008000', '#FFFF00', '#00FF00', '#000080', '#800000', '#FF0000', '#0000FF', '#00FFFF',
                 '#808080', '#FF00FF', '#FF1493']
    for j, color in enumerate([3, 4, 5]):
        label = [color_set[i] for i in res[color]]

        scatter = plt.scatter(x, y, c=label)
        plt.title('cluster reslut k = ' + str(clslist[j]))
        plt.ylabel('Arousal')
        plt.xlabel('Valence')
        plt.legend()
        plt.show()


    def plotsingle(cluster_num):
        x_f = []
        y_f = []
        c_f = []
        for i, j in enumerate(res[4]):
            # print(i,j)
            if j == cluster_num:
                x_f.append(res[1][i])
                y_f.append(res[2][i])
                c_f.append(res[4][i])
        label = [color_set[i] for i in c_f]

        plt.scatter(x_f, y_f, c=label)
        plt.title('cluster reslut k = 8. cluster = ' + str(cluster_num))
        plt.ylabel('Arousal')
        plt.xlabel('Valence')
        plt.legend()
        plt.show()


    for i in range(8):
        plotsingle(i)

    x_f = []
    y_f = []
    c_f = []
    for i, j in enumerate(res[4]):
        # print(i,j)
        if j == 5 or j == 7:
            x_f.append(res[1][i])
            y_f.append(res[2][i])
            c_f.append(res[4][i])
    label = [color_set[i] for i in c_f]

    plt.scatter(x_f, y_f, c=label)
    plt.title('cluster reslut k = 8. cluster = 5 7')
    plt.ylabel('Arousal')
    plt.xlabel('Valence')
    plt.legend()
    plt.show()
    '''
    import wordcloud
    import multidict as multidict
    import numpy as np
    w = wordcloud.WordCloud(background_color="white")
    fullTermsDict = multidict.MultiDict()
    data=[[('love', 0.57),('good', 1.2), ('like', 1.2), ('nice', 1.29), ('best', 1.61), ('cool', 1.7), ('great', 1.75), ('amazing', 1.8), ('shit', 1.83), ('much', 1.88), ('one', 1.91), ('music', 2.13), ('awesome', 2.17), ('dope', 2.21), ('fuck', 2.21), ('get', 2.21), ('lit', 2.24), ('wow', 2.24), ('ever', 2.24),('really', 2.29), ('cover', 2.33), ('beautiful', 2.37), ('remix', 2.37), ('beat', 2.37), ('song', 2.37), ('real', 2.37), ('omg', 2.42), ('man', 2.42), ('bad', 2.42), ('voice', 2.42), ('lol', 2.42), ('listen', 2.42), ('thank', 2.47), ('oh', 2.47), ('better', 2.47), ('fire', 2.47), ('got', 2.52), ('thanks', 2.57), ('please', 2.57), ('original', 2.57), ('new', 2.57), ('go', 2.63), ('yes', 2.63), ('bro', 2.69)],
      [('love', 0.0), ('life', 0.0), ('stop', 0.0), ('read', 0.0), ('bad', 0.0), ('kissed', 0.0), ('started', 0.01), ('friday', 0.01), ('nearest', 0.01), ('best', 0.03), ('freaky', 0.03), ('you', 0.04), ('name', 0.05), ('possible', 0.05), ('reading', 0.05), ('luck', 0.07), ('done', 0.07), ('day', 0.08), ('now', 0.1), ('press', 0.11), ('this', 0.14), ('crushes', 0.14), ('don', 0.15), ('appear', 0.2), ('big', 0.23), ('like', 0.28), ('good', 0.28), ('screen', 0.35), ('actually', 0.35), ('when', 0.39), ('letters', 0.43), ('nice', 0.79), ('awesome', 0.82), ('great', 0.85), ('cool', 0.98), ('lit', 1.05), ('put', 1.05), ('15', 1.09), ('is', 1.09), ('tomorrow', 1.12), ('much', 1.12), ('amazing', 1.12), ('so', 1.12), ('lol', 1.17), ('but', 1.17), ('ignore', 1.17), ('song', 1.17), ('144', 1.25)], 
      [('remix', 0.0), ('ass', 0.0), ('bad', 0.0), ('nice', 0.0), ('stupid', 0.0), ('shit', 0.0), ('man', 0.0), ('guess', 0.0), ('lit', 0.0), ('haters', 0.0), ('bruh', 0.0), ('fgyu', 0.0), ('lmao', 0.0), ('expecting', 0.0), ('user-666801694', 0.0), ('suck', 0.0), ('sylva_808', 0.0), ('aint', 0.0), ('dump', 0.0), ('mamma', 0.0), ('seems', 0.0), ('like', 0.0), ('one', 0.0), ('..sorry', 0.0), ('..stupid', 0.0), ('trash', 0.0), ('omg', 0.0), ('best', 0.0), ('ever', 0.0), ('af', 0.0), ('red', 0.0), ('hell', 0.0), ('ya', 0.0), ('fucked', 0.0), ('oii', 0.0), ('bomb', 0.0), ('mean', 0.0), ('bro', 0.0), ('sucks', 0.0), ('hey', 0.0), ('dumb', 0.0), ('this', 0.0), ('is', 0.0), ('sick', 0.0), ('user-105509710', 0.0), ('dude', 0.0), ('stop', 0.0), ('love', 0.0), ('guys', 0.0), ('make', 0.0)],
      [('love', 0.0), ('this', 0.01), ('good', 0.01), ('like', 0.05), ('best', 0.1), ('amazing', 0.16), ('nice', 0.2), ('great', 0.26), ('awesome', 0.4), ('it', 0.46), ('one', 0.51), ('ever', 0.54), ('you', 0.58), ('much', 0.58), ('life', 0.6), ('music', 0.62), ('stop', 0.65), ('really', 0.69), ('omg', 0.69), ('im', 0.72), ('wow', 0.72), ('cool', 0.74), ('the', 0.8), ('song', 0.8), ('bad', 0.82), ('listen', 0.82), ('voice', 0.85), ('lol', 0.91), ('my', 0.91), ('day', 0.91), ('so', 0.94), ('hi', 0.97), ('shit', 1.07), ('get', 1.15), ('go', 1.15), ('oh', 1.19),('favorite', 1.27), ('beautiful', 1.32), ('better', 1.32), ('fuck', 1.36), ('say', 1.36), ('check', 1.41), ('time', 1.41), ('got', 1.46), ('know', 1.46), ('makes', 1.52), ('track', 1.52), ('name', 1.52)], 
      [('goo.gl', 0.0), ('new', 0.0), ('check', 0.0), ('music', 0.0), ('love', 0.0), ('soundcloud.com', 0.0), ('latest', 0.0), ('news', 0.0), ('video', 0.0), ('back', 0.0), ('got', 0.0), ('the', 0.0), ('lil', 0.0), ('listen', 0.0), ('link', 0.0), ('ft', 0.0), ('shit', 0.0), ('my', 0.0), ('ft.', 0.0), ('like', 0.18), ('you', 0.18), ('go', 0.18), ('one', 0.18), ('drops', 0.18), ('track', 0.18), ('watch', 0.18), ('releases', 0.18), ('for', 0.18), ('drake', 0.18), ('wayne', 0.18), ('up', 0.18), ('best', 0.41), ('this', 0.41), ('follow', 0.41), ('work', 0.41), ('remix', 0.41),('dj', 0.41), ('justin', 0.41), ('bieber', 0.41), ('lit', 0.69), ('months', 0.69), ('life', 0.69), ('ago', 0.69), ('big', 1.1), ('great', 1.1), ('started', 1.1), ('good', 1.1), ('no', 1.1)],
      [('love', 0.0), ('good', 0.0), ('nice', 0.02), ('amazing', 0.04), ('awesome', 0.06), ('great', 0.06), ('like', 0.1), ('best', 0.1), ('cool', 0.1), ('wow', 0.17), ('remix', 0.24), ('music', 0.29), ('one', 0.31), ('dope', 0.31), ('track', 0.37), ('sick', 0.4), ('soundcloud.com', 0.45), ('the', 0.55), ('you', 0.55), ('it', 0.58), ('drop', 0.62), ('omg', 0.62), ('yeah', 0.62), ('beat', 0.62), ('lol', 0.66), ('yes', 0.66), ('shit', 0.66), ('really', 0.73), ('check', 0.82), ('fuck', 0.82), ('lit', 0.82), ('ever', 0.82), ('much', 0.82), ('so', 0.82), ('fucking', 0.86), ('damn', 0.91), ('better', 1.06), ('new', 1.06), ('song', 1.06), ('oh', 1.12), ('my', 1.12), ('is', 1.12), ('bad', 1.12), ('fire', 1.18), ('hi', 1.24), ('work', 1.24), ('original', 1.31), ('please', 1.31), ('stop', 1.31)], 
      [('love', 0.0), ('this', 0.0), ('best', 0.0), ('don', 0.0), ('read', 0.0), ('you', 0.0), ('will', 0.0), ('be', 0.0), ('kissed', 0.0), ('the', 0.0), ('by', 0.0), ('of', 0.0), ('your', 0.0), ('life', 0.0), ('now', 0.0), ('ve', 0.0), ('started', 0.0), ('is', 0.0), ('on', 0.07), ('nearest', 0.07), ('possible', 0.07), ('friday', 0.07), ('reading', 0.07), ('stop', 0.07), ('so', 0.07), ('freaky', 0.07), ('day', 0.15), ('life.tomorrow', 0.24), ('1.', 0.24), ('say', 0.24), ('please', 0.34), ('name', 0.34), ('10', 0.44), ('good', 0.56), ('bad', 0.56), ('crushes', 0.69), ('it', 0.85), ('times', 0.85), ('2.', 0.85), ('mom', 0.85), ('3.', 0.85), ('4.', 1.03), ('like', 1.03), ('know', 1.25), ('paste', 1.25), ('crush', 1.25), ('luck', 1.25), ('great', 1.25), ('nice', 1.25), ('kiss', 1.54)], 
      [('love', 0.0), ('shit', 0.0), ('check', 0.07), ('like', 0.09), ('fire', 0.12), ('best', 0.12), ('you', 0.12), ('lit', 0.14), ('follow', 0.17), ('the', 0.25), ('dope', 0.27), ('music', 0.3), ('fuck', 0.3), ('good', 0.33), ('new', 0.33), ('go', 0.39), ('got', 0.46), ('listen', 0.46), ('im', 0.5), ('get', 0.53), ('track', 0.53), ('my', 0.65), ('beat', 0.65), ('bad', 0.65), ('stop', 0.65), ('day', 0.65), ('nice', 0.69), ('life', 0.74), ('lol', 0.83), ('read', 0.94), ('page', 1.0), ('it', 1.0), ('big', 1.06), ('work', 1.06), ('please', 1.12), ('make', 1.12), ('fucking', 1.19), ('know', 1.19), ('started', 1.19), ('actually', 1.19), ('beats', 1.26), ('back', 1.34), ('done', 1.34), ('nigga', 1.34), ('real', 1.34), ('one', 1.34), ('hard', 1.43), ('bitch', 1.43)]]


    for i,item in enumerate(data):
        fullTermsDict = multidict.MultiDict()
        print(i)
        for i,tup in enumerate(item):
            #print(tup)
            fullTermsDict.add(tup[0],float(tup[1])+10/(i+0.1))
        #print(fullTermsDict)
        w.generate_from_frequencies(fullTermsDict)
        w.to_file('output'+str(i)+'.jpg')

    #单独print负向的簇

    fullTermsDict = multidict.MultiDict()
    for i,tup in enumerate(data[2]):
            fullTermsDict.add(tup[0],float(tup[1])+10/(i+0.1))
    w.generate_from_frequencies(fullTermsDict)
    w.to_file('output_p.jpg')
    '''
