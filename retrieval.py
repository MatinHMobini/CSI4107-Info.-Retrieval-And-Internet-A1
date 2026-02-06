
import math
from collections import Counter, defaultdict


# Function to do ranking
# Params: query: str -> query, index: inverted index from step 2, idf: dict -> inverted document frequency, preprocess: function -> function to preprocess text input, doc_lengths: dict -> precomputed document lengths, top_k: int -> number of top documents to return
# Return the top_k ranked documents as a list of tuples (doc_id, score)
def rank_documents(query, index, idf, preprocess, doc_lengths, top_k=100):

    # we preprocess the query to remove stopwords and apply stemming and tokenization
    query_t = preprocess(query)

    # Return empty list if query is empty
    if not query_t:
        return []
    
    # Compute term frequency for each term in query
    query_tf = Counter(query_t)

    # Compute weights for terms using TF-IDF
    query_weights = {}
    for term, tf in query_tf.items():
        if term in idf:
            query_weights[term] =  tf * idf[term]

    if not query_weights:
        return []
    
    # Compute the norm/magnitude of the query vector
    query_length = math.sqrt(sum(weight ** 2 for weight in query_weights.values()))
    if query_length == 0:
        return []
    
    # Computing the similarity scores for docs
    scores = defaultdict(float)

    for term, query_w in query_weights.items():
        if term not in index:
            continue
       
        for doc_id, doc_tf in index[term].items():
            doc_weight = doc_tf * idf[term]

            scores[doc_id] += query_w * doc_weight

    # Computing cosine similarity
    for doc_id in list(scores.keys()):
        denominator = doc_lengths.get(doc_id,0)
        if denominator > 0:
            scores[doc_id] /= denominator * query_length

        else:
            del scores[doc_id]

    # Sorting documents based on scores
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return ranked_docs[:top_k]

    # {(docid : score);(docid : score);(docid : score);(docid : score);(docid : score);(docid : score);(docid : score);(docid : score)}
        