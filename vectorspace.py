"""

Test file for vectorspace model


"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compare_docs(d1, d2):

    tfidf_vect = TfidfVectorizer()

    tfidf_matrix = tfidf_vect.fit_transform([d1, d2])

    sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

    return sim[0][0]


if __name__ == "__main__":

    doc1 = "I am eating a pizza"
    doc2 = "I am eating in an Italian restaurant"

    similarity_score = compare_docs(doc1, doc2)
    print(f"Similarity score: {similarity_score:.2f}")

