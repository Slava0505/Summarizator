import math
import numpy as np
from numpy.linalg import svd

import razdel
import spacy

# Список частей речи, которые мы не хотим считать значимыми.
# Подбирался на глаз.
BAD_POS = ("PREP", "NPRO", "CONJ", "PRCL", "NUMR", "PRED", "INTJ", "PUNCT", "CCONJ", "ADP", "DET", "ADV")

# Загрузка модели для частеречной разметки.
spacy_model = spacy.load("ru_core_news_md")


# Метод для разбиения текста на предложения.
def sentenize(text):
    return [s.text for s in razdel.sentenize(text)]


# Метод для токенизации предложения.
def tokenize_sentence(sentence):
    sentence = sentence.strip().replace("\xa0", "")
    tokens = [token.lemma_ for token in spacy_model(sentence) if token.pos_ not in BAD_POS]
    tokens = [token for token in tokens if len(token) > 2]
    return tokens


# Метод для токенизации всего текста.
def tokenize_text(text):
    all_tokens = []
    for sentence in sentenize(text):
        all_tokens.extend(tokenize_sentence(sentence))
    return all_tokens



class LsaSummarizer:
    """
    Латентно-семантический анализ для реферирования.
    Оригинальная статья: https://www.cs.bham.ac.uk/~pxt/IDA/text_summary.pdf
    """

    def __init__(
            self,
            verbose
    ):
        self.verbose = verbose

    def __call__(self, text, target_sentences_count):
        original_sentences = [s for s in sentenize(text)]
        tokenized_sentences = [tokenize_sentence(s) for s in original_sentences]

        # Словарь для последующего построения матрицы
        vocabulary = {token for sentence in tokenized_sentences for token in sentence}
        vocabulary = {word: idx for idx, word in enumerate(vocabulary)}
        if not vocabulary:
            return ""

        # Собственно построение матрицы
        matrix = self._create_matrix(tokenized_sentences, vocabulary)
        matrix = self._norm_matrix(matrix)

        # Сингулярное разложение
        _, sigma, v_matrix = svd(matrix, full_matrices=False)

        # Оставляем только важные тематики
        min_dimensions = max(3, target_sentences_count)
        topics_weights = [s ** 2 for s in sigma[:min_dimensions]]
        print("Веса важных тематик:", topics_weights)

        # Смотрим, как предложения в представлены в этих важных тематиках
        ranks = []
        for sentence_column, s in zip(v_matrix.T, original_sentences):
            sentence_column = sentence_column[:min_dimensions]
            print("ПРЕДЛОЖЕНИЕ: веса тематик: {}, текст: {}".format(sentence_column, s))
            rank = sum(s * v ** 2 for s, v in zip(topics_weights, sentence_column))
            ranks.append(math.sqrt(rank))

        indices = list(range(len(sentences)))
        indices = [idx for _, idx in sorted(zip(ranks, indices), reverse=True)]
        indices = indices[:target_sentences_count]
        indices.sort()
        return " ".join([original_sentences[idx] for idx in indices])

    def _create_matrix(self, sentences, vocabulary):
        """
        Создание матрицы инцидентности
        """
        words_count = len(vocabulary)
        sentences_count = len(sentences)

        matrix = np.zeros((words_count, sentences_count))
        for col, sentence in enumerate(sentences):
            for word in sentence:
                row = vocabulary[word]
                matrix[row, col] += 1
        return matrix

    def _norm_matrix(self, matrix):
        """
        Нормировка матрицы инцидентности
        """
        max_word_frequencies = np.max(matrix, axis=0)
        rows, cols = matrix.shape
        for row in range(rows):
            for col in range(cols):
                max_word_frequency = max_word_frequencies[col]
                if max_word_frequency != 0:
                    matrix[row, col] /= max_word_frequency
        return matrix