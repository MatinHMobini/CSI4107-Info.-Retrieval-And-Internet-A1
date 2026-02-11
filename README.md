# CSI4107-Info.-Retrieval-And-Internet-A1

Assignment 1

Student Names and Student Numbers

1- Michael Massaad (300293612).
2- Gabriel Zohrob (300309391).
3- Matin Hassanzadeh Mobini ().
4- Eric Zhou (300286231).

Tasks division:

- Michael Massaad: Implemented the retrieval.py for Step 3 - Retrieval and Ranking
- Gabriel Zohrob: Implemented the retrieval.py for Step 3 - Retrieval and Ranking
- Matin Hassanzadeh Mobini:
- Eric Zhou: 

- retrieval.py

    The file retrieval.py implements the retrieval and ranking component of the Information Retrieval system. It is responsible for scoring and ranking documents for a given query using the vector space model with TF-IDF weighting and cosine similarity.

    The main function provided in this file is rank_documents(), which takes a query and the inverted index produced in Step 2 and returns the top-ranked documents.

    Function: rank_documents(query, index, idf, preprocess, doc_lengths, top_k=100)

    This function performs the following operations:

        1. Query preprocessing  
        The input query string is preprocessed using the same preprocessing function applied to documents during indexing. This includes tokenization, stopword removal, and optional stemming, ensuring consistency between query terms and indexed terms.

        2. Query term frequency computation  
        After preprocessing, term frequencies for the query are computed using a Counter structure. Query terms that do not exist in the vocabulary (i.e., are not present in the IDF dictionary) are ignored.

        3. Query TF-IDF vector construction  
        TF-IDF weights for the query are computed using the formula:

        w(t,q) = tf(t,q) * idf(t)

        Only terms that appear in the inverted index are retained. The Euclidean norm of the query TF-IDF vector is then computed to support cosine similarity normalization.

        4. Candidate document selection and scoring  
        For each query term, the inverted index is used to retrieve all documents containing that term. These documents form the candidate set. For each candidate document, a partial similarity score is accumulated using the dot product between the query and document TF-IDF weights:

        w(t,d) = tf(t,d) * idf(t)

        The dot product is computed by summing w(t,q) * w(t,d) over all shared terms between the query and the document.

        5. Cosine similarity normalization  
        The accumulated dot product score for each document is normalized by the product of the query vector norm and the corresponding document vector norm (provided through the doc_lengths parameter), resulting in a cosine similarity score. Documents with zero or missing norms are discarded.

        6. Ranking and output  
        All candidate documents are sorted in decreasing order of their cosine similarity scores. The function returns the top k documents (default k = 100) as a list of (doc_id, score) tuples, which are later used to generate the final Results file in TREC format.


