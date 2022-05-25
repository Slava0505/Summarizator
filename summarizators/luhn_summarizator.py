from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords



from summarizators.base_summarizator import BaseSummarizator

class LuhnSummarizator(BaseSummarizator):
    """
    Класс реализующий суммаризацию Луна
    https://courses.ischool.berkeley.edu/i256/f06/papers/luhn58.pdf
    """
    def __init__(self, significant_word_procentile = 80, sum_sent_procentile = 95, word_normaliztion_type='stemmer'):
        """
        Parameters
        ----------
        significant_word_procentile:
        процентиль по которому
        будет создаваться список сигнификант слов.
        Сигнификант словами будут те, что
        окажутся больше процентился
        Важно! Формат в процентах, например 90

        sum_sent_procentile:
        процентиль по которому
        будут выбираться предложения для суммаризации
        Все предложения с сигнификант фактором
        больше этого процентиля попадут в
        суммаризованный текст
        Важно! Формат в процентах, например 90

        word_processing_type:
        Тип нормализации
        'stemmer' or None
        By default 'stemmer'
        """
        self.significant_word_procentile = significant_word_procentile
        self.sum_sent_procentile = sum_sent_procentile
        self.stop_words = set(stopwords.words('russian'))
        self.vectorizer = CountVectorizer()
        self.sent_tokenize_function = sent_tokenize
        if word_normaliztion_type=='stemmer':
            self.word_transformer = SnowballStemmer("russian")
            self.word_transforming_function = self.word_transformer.stem
        else:
            self.word_transforming_function = lambda x: x

    def normalize_sent(self, sent):
        """
        Метод принимает на вход предложение и возвращает
        его, фильтруя каждое слово по списку стоп слов
        и обработавает стеммером
        """
        tokens = [self.word_transforming_function(word) for word in \
                  word_tokenize(sent.lower()) if word not in self.stop_words]
        tokens = ' '.join(tokens)
        return tokens

    def sum_text(self, text):
        """
        Метод выполняющий суммаризацию

        Parameters
        ----------
        text: Текст для суммаризации

        Return
        ----------
        Суммаризированных текст
        """
        # Создание таблицы предложений на основе входящего текста
        self.sent_table = pd.DataFrame(self.sent_tokenize_function(text), columns=['sent'])
        # Нормализация каждого предложения
        self.sent_table['stemmed_sent'] = self.sent_table['sent'].apply(lambda sent: self.normalize_sent(sent))
        # Рассчет списка сигнификант слов
        self.calculate_significant_word_list()
        # Рассчет сигнификант фактора для каждого предложения
        self.sent_table['significant_factor'] = self.sent_table['stemmed_sent'].apply(lambda x: self.calculate_significant_factor(x))
        # Рассчет
        sum_sent_quantile = self.sum_sent_procentile / 100
        sf_treshhold = self.sent_table['significant_factor'].quantile(sum_sent_quantile)
        out_df = self.sent_table[self.sent_table['significant_factor'] > sf_treshhold]
        out = '\n'.join(out_df['sent'])
        return out

    def calculate_significant_word_list(self):
        """
        Метод рассчитывает список сигнификант слов, то
        есть слов наиболее значимых в рамках текста
        """
        significant_word_quantile = self.significant_word_procentile/100
        self.vectorizer.fit(self.sent_table['stemmed_sent'])
        word_features = self.vectorizer.transform(self.sent_table['stemmed_sent'])
        voc = self.vectorizer.vocabulary_
        voc_id = {word_id: word for (word, word_id) in voc.items()}
        count = word_features.sum(axis=0)
        counts = pd.Series(np.array(count)[0], index=pd.Series(voc_id).sort_values())
        count_treshhold = counts.quantile(significant_word_quantile)
        self.all_sign_words = list(counts[counts > count_treshhold].index)

    def calculate_significant_factor(self, sent):
        """
        Метод рассчитывающий сигнификант фактор
        для отдельного предложения

        Parameters
        ----------
        sent: Предложение подающееся
        на вход

        Return
        ----------
        Сигнификант фактор предложения
        """
        sent_sign_words_number = 0
        sent = sent.split()
        for word in sent:
            if word in self.all_sign_words:
                sent_sign_words_number += 1
        return sent_sign_words_number ** 2 / len(sent)

