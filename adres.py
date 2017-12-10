
# coding: utf-8

# In[1]:


import pandas as pd


# In[40]:


import nltk
import numpy as np
from nltk.corpus import brown
stop_words= nltk.corpus.stopwords.words('russian')
import pymystem3
mystem = pymystem3.Mystem()


# In[69]:


def lemm_row(text):
#     print(text.decode('utf-8'))
#     gr = " ".join([
#             row['group'].encode('utf-8'),
#             row['subgroup'].encode('utf-8'),
#             row['questions']
#         ])
    word=nltk.word_tokenize(text)# токинезация текста i-го документа      
    word_ws=[w.lower()  for w in   word if w.isalpha() ]#исключение слов и символов      
    word_w=[w for w in word_ws if w not in stop_words ]#нижний регистр  
    lem = mystem.lemmatize ((" ").join(word_w))# лемматизация i -го документа
    lema=[w for w in lem if w.isalpha() and len(w)>1]
    lem_str = " ".join(lema)
    return lem_str


# In[293]:


adr = pd.read_csv('adress.csv', names = ['data'], sep = '\n')


# In[128]:


import re


# In[294]:


adr['need'] = adr['data'].apply(lambda text: 1 if re.findall(r"(\d{6}),", text) != [] else 0)


# In[309]:


adr_str = adr[adr.need == 1]
adr_str['post_ind'] = adr_str['data'].apply(lambda txt: txt.split(',')[0])
adr_str['city'] = adr_str['data'].apply(lambda txt: txt.split(',')[1])
adr_str['city_low'] = adr_str['city'].apply(lambda x : x.strip().lower())
adr_str['rest'] = adr_str['data'].apply(lambda txt: ', '.join(txt.split(',')[2:]))


# In[311]:


counts = adr_str['city'].value_counts().reset_index()
counts.columns = ['city', 'count']


# In[312]:


adr_str = adr_str.merge(counts, on = 'city', how = 'left')


# In[313]:


counts = adr_str.groupby(['city','city_low', 'count'])['rest'].apply(lambda x : "; ".join(x)).reset_index()


# In[316]:


# city_list1 = counts.city.tolist()
city_list = adr_str['city_low'].tolist()


# In[317]:



#!/usr/bin/python3
# -*- coding: utf-8 -*-

def check_hardcode(text):
#     text = req['messages'][-1][2]
#     print(lemm_row(text))
#     print(city_list)
    if any(x.lower() in lemm_row(text) for x in [u'филиал', u'отделение']) & any(y.lower() in lemm_row(text) for y in [u'адрес']):
        for x in city_list:
            if x.lower() in lemm_row(text):
                return (u"В городе " + str(adr_str.loc[adr_str.city_low == x,'city'].values[0]).strip() + u' сейчас отделений '
                        + str(adr_str.loc[adr_str.city_low == x,'count'].values[0])
                        + ". " + u"Расположение в городе: " + str(adr_str.loc[adr_str.city_low == x,'rest'].values[0]).strip()
                       )
                        
        return u'в Вашем городе, к сожалению, филиалов нет. Пожалуйста, для уточнения информации позвоните по телефону нашего контактного центра 8 495 956−55−55'
        
#     if any(x.lower() in lemm_row(text) for x in city_list):
#         return u"добрый день, "
    return ""


# In[318]:


#OAtmp_adr = "адрес филиала в Барнаул"
#check_hardcode(tmp_adr)

