import math
import numpy as np
import networkx as nx
from scipy.linalg import eig
from matplotlib import pyplot as plt

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




def text_rank_preprocessing(sentence):
    return tokenize_sentence(sentence)


def text_rank_similarity(tokens1, tokens2):
    intersection_size = sum(tokens2.count(w) for w in tokens1)
    if intersection_size == 0:
        return 0.0

    if len(tokens1) <= 1 and len(tokens2) <= 1:
        return intersection_size

    assert len(tokens1) > 0 and len(tokens2) > 0
    norm = math.log(len(tokens1)) + math.log(len(tokens2))
    return intersection_size / norm


class TextRankSummarizer:
    """
    TextRank.
    """

    def __init__(
            self,
            damping=0.85,
            epsilon=1e-4,
            niter=100,
            preprocessing_function=text_rank_preprocessing,
            similarity_function=text_rank_similarity,
            verbose=False
    ):
        self.damping = damping
        self.epsilon = epsilon
        self.niter = niter
        self.preprocessing_function = preprocessing_function
        self.similarity_function = similarity_function
        self.threshold = None
        self.verbose = True

    def __call__(self, text, target_sentences_count):
        original_sentences = sentenize(text)
        sentences = [self.preprocessing_function(s) for s in original_sentences]

        graph = self._create_graph(sentences)
        graph = self._apply_threshold(graph)
        if self.verbose:
            plt.figure(figsize=(15, 10))
            sns.heatmap(graph, annot=True, fmt=".2f").set_title("Матрица схожести предложений")
        norm_graph = self._norm_graph(graph)
        ranks = self._iterate(norm_graph)

        if self.verbose:
            print("Значимости: {}".format(ranks))

            # Можно считать PageRank библиотечными методами.
            # При запуске на оригинальном графе должно быть то же самое.
            nx_graph = nx.from_numpy_array(graph)
            indices = list(range(len(sentences)))
            nx_ranks = nx.pagerank(nx_graph)
            nx_ranks = [ranks[i] for i in indices]
            assert np.all(np.isclose(nx_ranks, ranks))
            print("Проверка через NetworkX в порядке!")

            # Можно считать через честный метод степенных итераций над
            # модифицированной матрицей. Должно быть то же самое.
            random_transitions = np.full(graph.shape, 1.0 / len(graph))
            full_matrix = (1.0 - self.damping) * random_transitions + self.damping * norm_graph
            pm_ranks = self._power_method(full_matrix)
            assert np.all(np.isclose(pm_ranks, ranks))
            assert np.all(np.isclose(np.dot(full_matrix.T, pm_ranks), pm_ranks, atol=self.epsilon))
            print("Проверка через метод степенных итераций в порядке!")

            # А ещё можно через собственные вектора.
            # Только они могут отличаться на константный множитель из-за нормировки.
            vals, vecs = eig(full_matrix.T, left=False, right=True)
            eig_ranks = vecs[:, vals.argmax()]
            assert np.all(np.isclose(np.dot(full_matrix.T, eig_ranks), eig_ranks, atol=self.epsilon))
            multiplier = ranks[0] / eig_ranks[0]
            eig_ranks *= multiplier
            assert np.all(np.isclose(eig_ranks, ranks, atol=self.epsilon * 100))
            print("Проверка через собственные вектора в порядке!")

        indices = list(range(len(sentences)))
        indices = [idx for _, idx in sorted(zip(ranks, indices), reverse=True)]
        indices = indices[:target_sentences_count]
        indices.sort()
        return " ".join([original_sentences[idx] for idx in indices])

    def set_sim_function(self, func):
        self.similarity_function = func

    def set_preprocessing_function(self, func):
        self.preprocessing_function = func

    def set_threshold(self, threshold):
        self.threshold = threshold

    def _create_graph(self, sentences):
        """ Сборка изначального графа схожостей """
        sentences_count = len(sentences)
        graph = np.zeros((sentences_count, sentences_count))
        for sentence_num1, sentence1 in enumerate(sentences):
            for sentence_num2 in range(sentence_num1, sentences_count):
                sentence2 = sentences[sentence_num2]
                sim = self.similarity_function(sentence1, sentence2)
                graph[sentence_num1, sentence_num2] = sim
                graph[sentence_num2, sentence_num1] = sim
        return graph

    def _apply_threshold(self, graph):
        """ Обрезка графа по порогу, понадобится в LexRank """
        if self.threshold is None:
            return graph
        graph[graph < self.threshold] = 0.0
        return graph

    def _norm_graph(self, graph):
        """
        Нормировка по строкам, потому что ниже p_vector - вектор, а не столбец.
        Если бы p_vector был столбцом, надо было бы нормировать по столбцам.
        """
        norm = graph.sum(axis=1)[:, np.newaxis]
        norm_graph = graph / (norm + 1e-7)
        assert np.isclose(np.sum(norm_graph[0, :]), 1.0)
        assert np.all(np.isclose(norm_graph.sum(axis=1), np.ones((graph.shape[0],))))
        return norm_graph

    def _iterate(self, matrix):
        sentences_count = len(matrix)
        iter = 0
        lambda_val = 0.1
        p_vector = np.full((sentences_count,), 1.0 / sentences_count)
        random_transitions = np.full((sentences_count,), 1.0 / sentences_count)

        transposed_matrix = matrix.T
        while iter < self.niter and lambda_val > self.epsilon:
            next_p = (1.0 - self.damping) * random_transitions + self.damping * np.dot(transposed_matrix, p_vector)
            lambda_val = np.linalg.norm(np.subtract(next_p, p_vector))
            p_vector = next_p
            iter += 1
        return p_vector

    def _power_method(self, matrix):
        sentences_count = len(matrix)
        iter = 0
        lambda_val = 0.1
        p_vector = np.full((sentences_count,), 1.0 / sentences_count)

        transposed_matrix = matrix.T
        while iter < self.niter and lambda_val > self.epsilon:
            next_p = np.dot(transposed_matrix, p_vector)
            lambda_val = np.linalg.norm(np.subtract(next_p, p_vector))
            p_vector = next_p
            iter += 1
        return p_vector