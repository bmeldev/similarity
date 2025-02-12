import gensim
import numpy
import nltk
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity


def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text, language='spanish')
    tokens = [word for word in tokens if word.isalnum()]
    stop_words = set(stopwords.words('spanish'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens


def vectorize(text_tokens, model):
    vectors = [model[word] for word in text_tokens if word in model]
    if not vectors:
        return numpy.zeros(model.vector_size)
    return numpy.mean(vectors, axis=0)


def compare(text1, text2, model):
    tokens1 = preprocess(text1)
    tokens2 = preprocess(text2)
    
    vec1 = vectorize(tokens1, model)
    vec2 = vectorize(tokens2, model)
    
    similarity = cosine_similarity([vec1], [vec2])[0][0]
    return similarity


path = "path"
# model = KeyedVectors.load_word2vec_format(path, binary=True)
model = KeyedVectors.load_word2vec_format(path, binary=False)

text1 = "La Guía Michelin de 2025 incluye muchos restaurantes españoles."
text2 = "Me gusta comer pizza en un restaurante italiano."

simscore = compare(text1, text2, model)
print(f"Similarity: {simscore}")
