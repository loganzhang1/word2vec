#!/usr/bin/env python
# coding: utf-8

# 这里用pytorch实现word2vec中的Skip-Gram词袋模型

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
    def getOneHot(self,target):
        ret=torch.zeros(vo.length)
        ret[target]=1
        return ret
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
        for j in range(1,len(splits)-1):
            target=vo.word2id[splits[j]]
            for k in [-1,1]:
                context=vo.word2id[splits[j+k]]
                ret.append((context,target))
    return ret


# In[6]:


alldata=getDataSet()


# In[7]:


window_size=1 #对于每个target每次左右各取2个单词进行训练
embedding_dim=100
class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.embedding=nn.Embedding(vo.length,embedding_dim)
        self.linear=nn.Linear(embedding_dim,vo.length)
        self.activation_function=nn.LogSoftmax(dim=-1)
    def forward(self,inputs):#x中包含一个单词周围单词的One-Hot编码
        out=self.embedding(inputs)
        out=self.linear(out)
        out=self.activation_function(out)
        return out
model=Model()


# In[8]:


batch_size=100
def getBatched():
    batch_num = int(np.ceil(len(alldata) / float(batch_size)))
    batched_data=[]
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(alldata) - batch_size * i
        docs = [alldata[i * batch_size + b] for b in range(cur_batch_size)]
        batched_data.append(docs)
    return batched_data


# In[9]:


batched_data=getBatched()


# In[10]:


criterion=nn.NLLLoss()
parameters=torch.optim.SGD(model.parameters(),lr=0.01)


# In[11]:


epochs=20
def train():
    for i in range(20):
        losses=0
        for batch in batched_data:
            context=torch.zeros(batch_size,1)
            target=torch.zeros(batch_size)
            for j in range(len(batch)):
                context[j]=batch[j][0]
                target[j]=batch[j][1]
            context=torch.tensor(context, dtype=torch.long)
            target=torch.tensor(target, dtype=torch.long)
            model.zero_grad()
            out=model(context)
            out=torch.squeeze(out,dim=1)
            loss=criterion(out,target)
            losses=losses+loss
            loss.backward()
            parameters.step()
        print("[INFO] Epoch:",i,losses)


# In[12]:


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




