#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import os
from PIL import Image
import pathlib
import csv

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

#Keras
import keras

import warnings
warnings.filterwarnings('ignore')


# In[2]:


import tensorflow as tf
from tensorflow.keras.layers import LSTM
tf.__version__
import os 
import re
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
#nltk.download('stopwords')
from gensim.models.word2vec import Word2Vec
import numpy as np
import pickle
import sklearn


# In[3]:


############
# 数据预处理
############
# audio part

# 读取audio数据，shape=(767, 58)
data = pd.read_csv('./data/Emotion_feature_mood.csv')
data.head()


# In[4]:


# Dropping unneccesary columns
data1 = data.drop(['Unnamed: 0'],axis=1)
data2 = data1.drop(['song_name'],axis=1)
Arousal = data2.iloc[:, -2]
valence = data2.iloc[:, -1]

Y = Arousal
scaler = StandardScaler()
X = scaler.fit_transform(np.array(data2.iloc[:, :-2], dtype = float))
data.iloc[2]["song_name"]
dic_audio={}
for i in range(len(data)):
    song_name=data.iloc[i]["song_name"]
    content=list(data.iloc[i])[2:-2]
    #if i<2:
        #print(X[i],song_name,list(Y)[i])
    #content = data2.iloc[i]
    content = list(X)[i]
    dic_audio[song_name[:-4]]=content
#print(dic_audio['224'])


# In[ ]:





# In[5]:


############
# lyric part
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
def readdata(path,annotation_dict):
    file = os.walk(path)
    lrc=[]
    labels=[]
    musicid=[]
    for path,dir_list,file_list in file:
        for file_name in file_list:
            lrc_data=lrc2data(path,file_name)
            # label里没有歌曲101
            if file_name[:-4] in annotation_dict.keys():
                labels.append(annotation_dict[file_name[:-4]])
                lrc.append(lrc_data)
                musicid.append(file_name[:-4])
    with open('pkldata_with_musicID.pkl', 'wb') as f:
        pickle.dump((lrc,labels,musicid), f)
        
    return lrc,labels,musicid


# In[6]:


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
    f=open('pkldata_with_musicID.pkl',"rb")
    
    lrc,labels,musicid = pickle.load(f)
except:
    lrc,labels,musicid = readdata(path,annotation_dict)    
print(min([len(i) for i in lrc]))
print(musicid[:10])
dic_music_label={}
dic_lyric = {}
for i in range(len(musicid)):
    dic_music_label[musicid[i]]=labels[i]
    dic_lyric[musicid[i]] = lrc[i]


# In[7]:


'''
数据切分思路：
    储存两个字典，key=音乐编号，val分别为audio和lyric，
    train_test_split切key，根据key找数据
    dic_audio 
    dic_lyric
    

'''
songname_au = [i[:-4] for i in list(data["song_name"])]
songname_ly = musicid
#取交集作为下一步使用的数据集
songnames = list(set(songname_au)&set(songname_ly))
print(len(songnames),len(songname_au),len(songname_ly))
#由于要使用train_test_split，还要制作一个songnames对应的labels

label=[]
for name in songnames:
    #print(dic_music_label[name])
    label.append(dic_music_label[name][0])#这里选取了第一个值
#print(label)


#这里要把musicid与lrc以及audio编号匹配，内容变量分别为lrc，X

X_train_id, X_test_id, y_train, y_test = train_test_split(songnames, label, 
                                                   test_size=0.2)# 占比                                                  )
# 重新制作audio与lyric的数据
#X_train_au
#Y_train_au
#以及lyric

X_train_ly_raw = []
X_test_ly_raw = []
X_train_au_raw=[]
X_test_au_raw=[]
for item in X_train_id:
    X_train_ly_raw.append(dic_lyric[item])
    X_train_au_raw.append(dic_audio[item])
for item in X_test_id:
    X_test_ly_raw.append(dic_lyric[item])
    X_test_au_raw.append(dic_audio[item])
    #print("music id:", item, " data: ",dic_audio[item])
#print(len(X_train_au_raw))
X_train_au_raw=np.array(X_train_au_raw)
X_test_au_raw=np.array(X_test_au_raw)
#print(y_test)
#print(X_train_ly_raw)


# In[8]:


# 以np形式读取出词向量
max_time = 100         # rnn 文本最大长度
INPUT_SIZE = 32         # rnn 词向量长度

vector_dim=INPUT_SIZE # 目前限定单个lyric 9个词
#all_embedding=Word2Vec(sentences=lrc, vector_size =32, min_count=1, workers=4) #新版本用法
all_embedding=Word2Vec(sentences=lrc, size =32, min_count=1, workers=4)
print(all_embedding.wv["Uhh"])

def generate_embedding(data=X_train_ly_raw):
    embedding_matrix = np.zeros((len(data), max_time, vector_dim))
    for item, lyr in enumerate(data):
        #print(lyr)
        embedding_temp=[]
        for i in range(len(lyr)):
            word= lyr[i]
            try:
                embedding_vector = all_embedding.wv[word]
            except:
                embedding_vector =None
            #print(embedding_vector,word)
            if embedding_vector is not None:
                embedding_temp.append(embedding_vector)
            if len(embedding_temp)==max_time:
                break
        #print("embedding_temp",len(embedding_temp))
        # 对于不足max_time的，进行补齐
        if len(lyr)<max_time:
            for num in range(max_time-len(lyr)):
                #print(num,len(lyr))
                embedding_vector=[0]*vector_dim
                embedding_temp.append(embedding_vector)
                
        #print(len(embedding_temp))
        embedding_matrix[item] = embedding_temp
    return embedding_matrix
embedding_matrix=generate_embedding()
print("embedding_matrix",embedding_matrix[0])


print(len(embedding_matrix),len(embedding_matrix[0]),len(embedding_matrix[1]),"embedding_matrix")
print(len(y_train))


# In[9]:


####
##数据整理汇总
####
#lyric数据
train_x=embedding_matrix
test_x = generate_embedding(X_test_ly_raw)
train_y=tf.convert_to_tensor([[float(i)] for i in y_train])
test_y=tf.convert_to_tensor([[float(i)] for i in y_test])
# audio数据
X_train_au = np.reshape(X_train_au_raw, (X_train_au_raw.shape[0], 1, X_train_au_raw.shape[1]))
X_test_au = np.reshape(X_test_au_raw, (X_test_au_raw.shape[0], 1, X_test_au_raw.shape[1]))
print(X_train_au.shape)


# In[ ]:





# In[10]:


"""
step 3 建立深度神经网络结构
"""
from keras.layers import Input, Embedding, LSTM, Dense,Activation,Multiply
from keras.models import Model
import numpy as np
import keras
import tensorflow.keras.backend as K
def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f


def Att(att_dim,inputs,name):
    V = inputs
    QK = Dense(att_dim)(inputs)
    QK = Activation("softmax",name=name)(QK)
    MV = Multiply()([V, QK])
    return(MV)



x_ly=tf.convert_to_tensor(train_x)
#X_train_au1=X_train_au[:484]
#print("lyric输入shape: ",x_ly.shape,"\n audio输入shape: ",
#      X_train_au.shape,"\n修改后的shape：",X_train_au1.shape)
train_x=embedding_matrix
train_y=tf.convert_to_tensor([[float(i)] for i in y_train])
#print(y_train)
ly_input = Input((100,32), name='ly_input' )
au_input = Input((1,54), name='au_input')
lstm_out_ly = LSTM(32)(ly_input)
lstm_out_au = LSTM(32)(au_input)

#x = keras.layers.concatenate([lstm_out, lstm_out1])
x = keras.layers.concatenate([lstm_out_ly, lstm_out_au])
print(x.shape)
x = Att(64,x,"attention_vec")

x = Dense(32, activation='relu')(x)
main_output = Dense(1, activation='sigmoid', name='main_output')(x)
 
model = Model(inputs=[ly_input, au_input], outputs=[main_output])
#model.compile(optimizer='rmsprop', loss='binary_crossentropy', loss_weights=[1., 0.2])
#model.compile(optimizer='rmsprop', 
#            loss={'main_output': 'binary_crossentropy'},
#            loss_weights={'main_output': 1.})
#model.compile(loss='mse', optimizer='rmsprop', metrics=['mae',r2])
model.compile(loss='mse', optimizer='Adam', metrics=['mae',r2])

#print(x_ly.shape, X_train_au1.shape,train_y.shape,X_train_au.shape)
Fusion=model.fit(x={'ly_input': x_ly, 'au_input': X_train_au},
            y={'main_output': train_y},
          steps_per_epoch=32, epochs=20,verbose=1)


# In[11]:


#test
# 测试数据预处理流程
# 
embedding_test=generate_embedding(X_test_ly_raw)
X_test1 = tf.convert_to_tensor([np.asarray(i) for i in embedding_test])
Y_test=tf.convert_to_tensor(test_y)


print(embedding_test.shape)
print(X_test_au.shape)
model.evaluate(
            x={'ly_input': X_test1, 'au_input': X_test_au},
            y={'main_output': Y_test},steps=10,verbose=1
)




