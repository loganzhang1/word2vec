#!/usr/bin/env python
# coding: utf-8

# 这里用pytorch实现word2vec中的CBOW词袋模型

# In[1]:


import torch
import torch.nn as nn
import numpy as np
import pandas as pd


# In[2]:


data=pd.read_csv('../data/test_a.csv')[:100]
data=data.values
data=[i[0] for i in data]


# In[3]:


class Vocab:
    def __init__(self):
        self.words=[]
        self.word2id=[]
        self.length=0
    def buildVoca(self,train_data):
        for i in train_data:
            splits=i.split()
            for j in splits:
                if j not in self.words:
                    self.words.append(j)
        changedToId=lambda x:dict(zip(x,range(len(x))))
        self.word2id=changedToId(self.words)
        self.length=len(self.words)


# In[4]:


vo=Vocab()
vo.buildVoca(data)


# In[5]:


#建立好词典之后开始构建训练集
#数据的格式是([0...1...],[[0..1..],[0...1...0]...])
def getDataSet():
    ret=[]
    for i in data:
        splits=i.split()
        for j in range(2,len(splits)-2):
            li=[]  #刚开始的时候li这里定义写到外面了，导致每一次产生一个训练数据的时候都累加到上一次的后面了
            target=vo.word2id[splits[j]]
            li.append(vo.word2id[splits[j-1]])
            li.append(vo.word2id[splits[j-2]])
            li.append(vo.word2id[splits[j+1]])
            li.append(vo.word2id[splits[j+2]])
            ret.append((li,target))
    return ret


# In[6]:


alldata=getDataSet()


# In[7]:


window_size=2 #对于每个target每次左右各取2个单词进行训练
embedding_dim=100
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.embedding=nn.Embedding(vo.length,embedding_dim)
        self.linear=nn.Linear(embedding_dim,vo.length)
        self.activation_function=nn.LogSoftmax(dim=-1)
    def forward(self,inputs):#x中包含一个单词周围单词的One-Hot编码
        #x:(batch_size,4,vo.length)
        #y:(batch_size,vo.length)
        embeddings=self.embedding(inputs)
        #print(embeddings.shape)
        
        out=sum(embeddings).view(1,-1)
        out=self.linear(out)
        #print(out.shape)
        out=self.activation_function(out)
        return out
model=Model()


# In[8]:


criterion=nn.NLLLoss()
parameters=torch.optim.SGD(model.parameters(),lr=0.01)


# In[14]:


epochs=20
def train():
    for i in range(20):
        losses=0
        for context,target in alldata:
            #temp=torch.zeros((4,vo.length))
            #print('context',context)
            #print('target',target)
            context=torch.tensor(context, dtype=torch.long)
            target=torch.tensor([target], dtype=torch.long)
            #print(context.shape)
            #print(target.shape)
            model.zero_grad()
            out=model(context)
            print('out:',out)
            print('target:',target)
            loss=criterion(out,target)
            losses=losses+loss
            loss.backward()
            parameters.step()
        print("[INFO] Epoch:",i,losses)


# In[ ]:


train()


# In[13]:


with torch.no_grad():
    context,target=alldata[0]
    context=torch.tensor(context,dtype=torch.long)
    target=torch.tensor([target],dtype=torch.long)
    out=model(context)
    print(out)
    print(np.argmax(out))


# In[ ]:




