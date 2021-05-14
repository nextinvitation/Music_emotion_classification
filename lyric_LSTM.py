#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.layers import LSTM
tf.__version__


# In[8]:


import os 
import re
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
from gensim.models.word2vec import Word2Vec
import numpy as np

import pickle


# In[9]:


"""
step 1 输入lyric.lyric数据预处理逻辑，读入路径下全部文件名，载入数据，进行停词分词
"""
# readdata输入一个path，输出一个list储存textdata
# processdata，输入textdata，输出停词分词后的data
def lrc2data(path,file_name):
    # 打开文件
    file = open(path +"/"+ file_name, "r", encoding="utf-8")
    # 读取文件全部内容
    lrc_list = file.readlines()
    lrc_data = []
    for i in lrc_list:
        lrc_word = i.strip().split("]")
        #print(i,"str(lrc_word[-1])",str(lrc_word[-1]),"str(lrc_word[-1])")
        if str(lrc_word[-1])!="":
            #对于筛选出的句子，剔除掉全部非英文字符
            rstr = r"[\=\(\)\,\/\\\:\*\?\"\<\>\|\' ']"
            text = re.sub(rstr, " ", lrc_word[-1])
            #print(text)
            # LookupError(resource_not_found)
            # nltk.download('punkt')
            token = nltk.word_tokenize(text)
            filtered = [w for w in token if w not in stopwords.words('english')]
        
            lrc_data+=filtered # 得到单首歌曲 全部歌词的一个list
    return lrc_data


# In[10]:


def readdata(path,annotation_dict):
    file = os.walk(path)
    lrc=[]
    labels=[]
    for path,dir_list,file_list in file:
        for file_name in file_list:
            lrc_data=lrc2data(path,file_name)
            # label里没有歌曲101
            if file_name[:-4] in annotation_dict.keys():
                labels.append(annotation_dict[file_name[:-4]])
                lrc.append(lrc_data)
    with open('pkldata.pkl', 'wb') as f:
        pickle.dump((lrc,labels), f)
        
    return lrc,labels


# In[13]:


# 制作一个annotation的dic，key=歌曲编号，val=标签 [a, v]
annotation_dict={}
with open("./data/static_annotations.csv","r") as f:
    annotation=f.readlines()
    for i in annotation:
        annotation_word = i.strip().split(",")
        key = annotation_word[0]
        val = [annotation_word[1], annotation_word[2]]
        annotation_dict[key]=val
# print(annotation_dict["10"])            
path = "./data/lyrics"
try:#检测pkl是否存在
    f=open("pkldata.pkl","rb")
    
    lrc,labels = pickle.load(f)
except:
    lrc,labels = readdata(path,annotation_dict)    
print(min([len(i) for i in lrc]))


# In[14]:


"""
step 2 数据预处理：对应标签，拆分训练及测试集,
需要单独制作一个labels 对应数据集每一个数据。当前labels为一个值arouse
"""
print("test labels",labels[10])
label=[[i[1]] for i in labels]
#print("label",label[:10],lrc[:1])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(lrc, label, 
                                                    test_size=0.2# 占比
                                                    )
X_train_w2v=Word2Vec(sentences=X_train, size=32, min_count=2, workers=4)
X_test_w2v=Word2Vec(sentences=X_test, size=32, min_count=2, workers=4)
print(y_train[:10],X_train[:5])
print("before--X_train",len(X_train),len(X_train[1]),len(X_train[0]))


# In[15]:


# 以np形式读取出词向量
max_time = 100         # rnn 文本最大长度
INPUT_SIZE = 32         # rnn 词向量长度

vector_dim=INPUT_SIZE # 目前限定单个lyric 9个词
all_embedding=Word2Vec(sentences=lrc, size=32, min_count=1, workers=4)

def generate_embedding(data=X_train):
    embedding_matrix = np.zeros((len(data), max_time, vector_dim))
    for item, lyr in enumerate(data):
        #print(lyr)
        embedding_temp=[]
        for i in range(len(lyr)):
            word= lyr[i]
            try:
                embedding_vector = all_embedding[word]
            except:
                embedding_vector =None
            #print(embedding_vector,word)
            if embedding_vector is not None:
                embedding_temp.append(embedding_vector)
            if len(embedding_temp)==max_time:
                break
        # 对于不足max_time的，进行补齐
        if len(lyr)<max_time:
            for num in range(max_time-len(lyr)):
                embedding_vector=[0]*vector_dim
                embedding_temp.append(embedding_vector)
                
        #print(embedding_temp)
        embedding_matrix[item] = embedding_temp
    return embedding_matrix
embedding_matrix=generate_embedding()
print("embedding_matrix",embedding_matrix[0])


print(len(embedding_matrix),len(embedding_matrix[0]),len(embedding_matrix[1]),"embedding_matrix")
print(len(y_train))


# In[ ]:





# In[38]:


"""
step 3 建立深度神经网络结构
"""
import tensorflow.keras.backend as K
def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f

model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(loss='mse', optimizer='rmsprop', metrics=['mae',r2])
'''

history = model.fit(train_dataset, epochs=10,
                    validation_data=test_dataset, 
                    validation_steps=30)
'''
train_x=embedding_matrix
train_y=[[float(i[0])] for i in y_train]

dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
dataset = dataset.batch(32)
dataset = dataset.repeat()

model.fit(dataset, epochs=1000, steps_per_epoch=2,verbose=2)


# In[ ]:





# In[36]:


embedding_test=generate_embedding(X_test)
X_test1 = tf.convert_to_tensor([np.asarray(i) for i in embedding_test])
Y_test1=tf.convert_to_tensor(y_test)
X_test1 = embedding_test
Y_test1=[[float(i[0])] for i in y_test]
X_test1 = tf.convert_to_tensor([np.asarray(i) for i in embedding_test])
Y_test1=tf.convert_to_tensor(Y_test1)
print("\nTesting ...")
r2 = model.evaluate(
    X_test1, 
    Y_test1,
    batch_size=61, verbose=1
)


# In[ ]:




