#!/usr/bin/python3
# -*- coding: utf-8 -*-

from keras.layers import (Dense, LSTM, Activation, Input, Reshape, RepeatVector, Add, Dot, merge, Lambda)
from keras import backend as K
from keras.models import Model
from keras.engine.topology import Layer
from keras.losses import categorical_crossentropy
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
import sklearn.metrics

from sklearn import preprocessing
import numpy as np
from collections import namedtuple

import re
from gensim.models import Word2Vec
import numpy as np
import pickle
from functools import partial

from keras.models import load_model


Test = namedtuple('Test', ['ptr', 'qtr', 'ctr'])


class WLayer(Layer):
    """
    Основной слой
    """

    def __init__(self, emb_dim=25, **kwargs):
        self.k = emb_dim
        super(WLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='W',
            shape=(self.k, self.k),
            initializer='glorot_uniform',
            regularizer=l2(0.01),
            trainable=True
        )
        super(WLayer, self).build(input_shape)

    def call(self, input_, mask=None):
        return K.dot(input_, self.W)

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 3:
            return input_shape[0], input_shape[1], input_shape[2]
        else:
            return input_shape[0], input_shape[1]


class AttentionLayer(Layer):
    """
    Вычисляет важность слов в контексте
    """

    def __init__(self, emb_dim=25, **kwargs):
        self.k = emb_dim
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name='W',
            shape=(self.k, 1),
            initializer='glorot_uniform',
            regularizer=l2(0.01),
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, input_, mask=None):
        return K.dot(input_, self.W)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], 1


class RLayer(Layer):
    """
    Вспомогательный слой, использует веса
    """

    def __init__(self, emb_dim=25, **kwargs):
        self.k = emb_dim
        super(RLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(RLayer, self).build(input_shape)

    def call(self, input_, mask=None):
        # Y * alpha
        return K.batch_dot(input_[0], input_[1], axes=[1, 1])

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][2]


class MyModel:
    def __init__(self, train, test=None):
        self.model = None
        self.attention_model = None
        self.train = train
        self.test = test
        self.pshape = (self.train.ptr.shape[1], self.train.ptr.shape[2])
        self.qshape = (self.train.qtr.shape[1], self.train.qtr.shape[2])
        self.acc = None
        self.prediction = None

    def compile(self, emb=25, lr=0.01, loss='mean_squared_error', metrics=['accuracy'], norm=False):
        """
        Данные:
        conc: полный комментарий + все цепочки для комментария,
        ptr: полный комментарий,
        qtr: цепоки для комментария
        """
        par_input = Input(shape=self.pshape)
        qus_input = Input(shape=self.qshape)

        # conc = Concatenate(axis=1)([par_input, qus_input])
        Ans = LSTM(units=emb, return_sequences=True, dropout=0.5)(par_input)
        Que = LSTM(units=emb, return_sequences=True, dropout=0.5)(qus_input)
        
        Ans_ = WLayer(emb_dim=emb)(Ans)
        Que_ = WLayer(emb_dim=emb)(Que)
        
        M_Ans = Activation('tanh')(Ans_)
        M_Que = Activation('tanh')(Que_)

        alpha_m_ans = AttentionLayer(emb_dim=emb, name='answer_att')(M_Ans)
        alpha_m_que = AttentionLayer(emb_dim=emb, name='quest_att')(M_Que)

        alpha_m_ans = Reshape((int(alpha_m_ans.shape[1]),))(alpha_m_ans)
        alpha_m_ans = Activation('softmax')(alpha_m_ans)
        
        alpha_m_que = Reshape((int(alpha_m_que.shape[1]),))(alpha_m_que)
        alpha_m_que = Activation('softmax')(alpha_m_que)

        r_ans = RLayer(emb_dim=emb, name='ans')([Ans_, alpha_m_ans])
        r_que = RLayer(emb_dim=emb, name='que')([Que_, alpha_m_que])
        
        
        r_ans_modified = WLayer(emb_dim=emb, name='answer')(r_ans)
        r_que_modified = WLayer(emb_dim=emb, name='quest')(r_que)  # Модифицируем вопрос так,
        
        
        dot = Dot(axes=1, normalize=norm)([r_que_modified, r_ans_modified])
        out = Activation('sigmoid')(dot)
        
        self.model = Model(inputs=[par_input, qus_input], outputs=out)
        self.model.compile(loss=loss, optimizer=Adam(lr=lr), metrics=metrics)

        # layer_name = self.model.layers[9].name
        # self.attention_model = Model(
        #    inputs=self.model.input,
        #    outputs=self.model.get_layer(layer_name).output
        # )

    def fit(self, n=None, epochs=10, batch_size=64, verbose=1):
        if n is None:
            n = len(self.train.ptr)
            
        label = self.train.tt[:n]
        self.model.fit([self.train.ptr[:n], self.train.qtr[:n]],
                       label, epochs=epochs, verbose=verbose, batch_size=batch_size)
        return self.model

    def predict(self, dataset=None):
        if dataset is None:
            test = self.test
        else:
            test = dataset
        prediction = self.model.predict([test.ptr, test.qtr])
        self.acc = sklearn.metrics.mean_squared_error(prediction, test.tt)
        self.prediction = prediction

        return prediction


class Word2VecWrapper(Word2Vec):
    def __init__(self, df, **kwargs):
        # TODO: catch exceptions if is not in the voacabulary
        self._df = df
        self.kw = kwargs
        self._reg_split = '\ |\,|\!|\.\—|\:|\;|\.|\?|\(|\)|\[|\]\-|\—'

        self._split_sentences = self._get_words_lists()

        super(Word2VecWrapper, self).__init__(self._split_sentences, **kwargs)
        self.vectors, self.word2vec, self.word2num, self.num2words = self._parse_wv()
        self.data_split = None

    def _get_words_lists(self):
        sentences = [i for i in self.df['CLEAN_TEXT']]
        _split_sentences = [list(filter(lambda x: x != '' and x != ' ',
                                        re.split(self._reg_split, str(s))))
                            for s in sentences]

        return _split_sentences

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, x):
        self._df = x
        self._get_words_lists()
        self._split_sentences = self._get_words_lists()
        super(Word2VecWrapper, self).__init__(self._split_sentences)

    def _parse_wv(self):
        vocab = self.wv.index2word
        vectors = np.array([self.wv.word_vec(w) for w in vocab])
        num_to_words = dict(enumerate(vocab))
        words_to_num = {v: k for k, v in num_to_words.items()}
        word_to_vectors = {w: v for w, v in zip(vocab, vectors)}
        return vectors, word_to_vectors, words_to_num, num_to_words    
    
    
class Echo:
    def __init__(self, model, answers, w2v_model, texts, pad_len=12):
        self.model = model
        self.answers = answers
        self.w2v_model = w2v_model
        self.no_vocab = np.array([0 for _ in range(100)])
        self.voc_keys = w2v_model.wv.vocab.keys() 
        self.pad_len = pad_len
        self.texts = texts

    def __call__(self, phrase):
        vec_phrase  = [np.array([self.w2v_model.wv.word_vec(w) 
                                 if w in self.voc_keys else self.no_vocab
                                 for w in re.split(
                                     '\ |\,|\!|\.\—|\:|\;|\.|\?|\(|\)|\[|\]\-|\—', 
                                     phrase.lower()
                                 )]
                               )
                      ]
        
        padded_vec_phrase = pad_sequences(
            vec_phrase, 
            maxlen=self.pad_len,
            padding='post',
            dtype='float32',
            value=self.no_vocab
        )
        
        que_inp = np.array([padded_vec_phrase[0,:,:] for _ in range(self.answers['vector'].shape[0])])        
        predictions = np.array([i[0] for i in self.model.predict([self.answers['vector'], que_inp])])
        
        arg_max = predictions.argmax()        
        
        return {
            'text': self.texts[arg_max],
            'code': self.answers['code'][arg_max]
        }


texts = pickle.loads(open('texts.pkl', 'rb').read())
answers = pickle.loads(open('answers.pkl', 'rb').read())
w2v_model = pickle.loads(open('w2v_model.pkl', 'rb').read())

train = pickle.loads(open('train.pkl', 'rb').read())
model = load_model('model.hd5')


echo = Echo(model, answers,  w2v_model, texts)


def check_hardcode_(echo, req):
    text = req['messages'][-1][1]
    return echo(text)


check_hardcode = partial(check_harcode_, echo)


def check_hard_calc(req):
    text = req['messages'][-1][1]
    if any(x.lower() in text.lower() for x in [u"каско",u"осаго",u"авто",u"машин",u"hundai",u"kia",u"иномарк"]):
        if any(x.lower() in text.lower() for x in [u"рассчет",u"стоит",u"почем",u"скольк",u"купит",u"взят",u"дай",u"посчит",u"офор"]):
            return True
    return False
