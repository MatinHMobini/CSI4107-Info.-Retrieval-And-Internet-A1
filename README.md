# CSI4107-Info.-Retrieval-And-Internet-A1

## Assignment 1

### Student Names and Student Numbers

1. Michael Massaad (300293612)
2. Gabriel Zohrob (300309391)
3. Matin Hassanzadeh Mobini (300283854)
4. Eric Zhou (300286231)

### Tasks division:

- Michael Massaad: Implemented the retrieval.py for Step 3 - Retrieval and Ranking
- Gabriel Zohrob: Implemented the retrieval.py for Step 3 - Retrieval and Ranking
- Matin Hassanzadeh Mobini: Implemented Step 2 - Indexing (indexing.py), including building the inverted index, computing IDF values, and precomputing document vector lengths. Also created the runner script (run_ir.py) to build the index and generate the Results file in TREC format for the test queries.
- Eric Zhou: Implemented the preprocessing.py for Step 1 - Preprocessing

### Code Explanation
- preprocessing.py

    This file implements the preprocessing component of thee Information Retrieval system. It is responsible for tokenization and stopword removal on a given document of text. It filters out markup, punctuation, numbers, and stopwords, and a step of Porter Stemming is included.

  There are two functions:
  - The first is preprocess_corpus which is used to process the text of each document in order as well as combining the title and text of each document together.
  - The seecond is preprocess_text which is the main method used to filter and tokenize each word.

- indexing.py

This file implements Step 2 - Indexing. It builds an inverted index over the SciFact corpus using the tokens produced by preprocessing.py. The index maps each term to the list of documents that contain the term along with its term frequency. The file also computes IDF values for each term and precomputes each document’s TF-IDF vector length (L2 norm) to support cosine similarity normalization during retrieval.

Outputs produced by this module:
- index: term -> {doc_id: tf}
- idf: term -> idf(term)
- doc_lengths: doc_id -> ||doc||

The generated structures are saved to disk (index.pkl) so retrieval can be run efficiently without rebuilding the index every time.

- run_ir.py

This script runs the full pipeline on the test queries. It can rebuild/load the index and produces a Results file in the required TREC format (query_id Q0 doc_id rank score tag) for the top-100 documents per query. It supports running using query titles only or titles + full text.


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

### How to run the code: 

1. Open a terminal in the project directory (where all the project files are located).

2. Create a virtual environment by running: python -m venv .venv

3. Activate the virtual environment:
- On Windows:.venv\Scripts\activate
- On macOS/Linux: source .venv/bin/activate

4. Install the required library: pip install nltk

5. Run the system using query titles only: python run_ir.py --rebuild --mode title --run_name title_run --results_path Results_title

6. Run the system using query titles and full text: python run_ir.py --mode title+text --run_name title_text_run --results_path Results_title_text

7. Evaluate the results using trec_eval:
- trec_eval qrels.txt Results_title
- trec_eval qrels.txt Results_title_text


### Experiments and Results (MAP)

We ran two configurations:

Run A (doc_title): indexed document titles only (--doc_mode title)
MAP = 0.3650

Run B (doc_title_text): indexed document titles + abstracts (--doc_mode title+text)
MAP = 0.5368

Best run: doc_title_text (MAP 0.5368). The final submission file "Results" corresponds to this run.


### Discussion

Indexing titles + abstracts (Run B) performs substantially better than indexing titles only (Run A), improving MAP from 0.3650 to 0.5368. This is expected because abstracts contain many domain terms (methods, outcomes, entities) that do not appear in titles, increasing term overlap between queries (claims) and relevant documents and improving recall without severely harming precision.



### Vocabulary size and sample tokens

Run A (doc_title) vocabulary size: 8729 tokens
Run B (doc_title_text) vocabulary size: 32152 tokens

Sample of 100 vocabulary tokens (Run B):
Vocab sample (first 100): ['aa', 'aaa', 'aaaatpas', 'aaafamili', 'aab', 'aabenhu', 'aacr', 'aacrthi', 'aad', 'aadinduc', 'aadtreat', 'aag', 'aah', 'aai', 'aakampk', 'aalpha', 'aam', 'aanatsnat', 'aarhu', 'aaronquinlangmailcom', 'aasv', 'aatf', 'aauaaa', 'aav', 'ab', 'abad', 'abandon', 'abas', 'abb', 'abber', 'abbott', 'abbrevi', 'abc', 'abca', 'abcamedi', 'abcb', 'abcc', 'abcg', 'abciximab', 'abciximabrel', 'abctarget', 'abd', 'abda', 'abdb', 'abdomen', 'abdomin', 'abdominala', 'abduct', 'aberr', 'aberrantincomplet', 'aberrantli', 'abeta', 'abf', 'abfreb', 'abfrebsit', 'abi', 'abibas', 'abil', 'ability–', 'abiot', 'abirateron', 'abl', 'ablat', 'abm', 'abmd', 'abnorm', 'abocompat', 'abolish', 'abort', 'abound', 'abovefacil', 'abovement', 'abp', 'abpa', 'abpi', 'abrb', 'abroad', 'abrog', 'abrupt', 'abruptli', 'abscess', 'abscis', 'absciss', 'absenc', 'absent', 'absolut', 'absoluteconcentr', 'absorb', 'absorbancecytophotometr', 'absorpt', 'absorptiometri', 'abstain', 'abstent', 'abstin', 'abstract', 'abstracta', 'abstractmicrorna', 'abt', 'abuja', 'abulia']



### Top-10 retrieved documents for the first two queries (Best run)

We show the first 10 ranked documents (TREC format lines) for the first two query IDs in the Results file (Run B / best run):

Query 9 top 10:

9 Q0 44265107 1 0.587922 doc_title_text
9 Q0 24700152 2 0.301629 doc_title_text
9 Q0 32787042 3 0.198282 doc_title_text
9 Q0 14647747 4 0.190943 doc_title_text
9 Q0 27188320 5 0.184525 doc_title_text
9 Q0 6477536 6 0.180812 doc_title_text
9 Q0 9056874 7 0.171567 doc_title_text
9 Q0 19511011 8 0.168167 doc_title_text
9 Q0 26105746 9 0.152509 doc_title_text
9 Q0 25182647 10 0.152358 doc_title_text

Query 11 top 10:

11 Q0 25510546 1 0.259486 doc_title_text
11 Q0 1887056 2 0.217684 doc_title_text
11 Q0 5107861 3 0.196525 doc_title_text
11 Q0 16712164 4 0.175488 doc_title_text
11 Q0 4399311 5 0.172709 doc_title_text
11 Q0 13780287 6 0.172630 doc_title_text
11 Q0 2251426 7 0.166276 doc_title_text
11 Q0 29459383 8 0.161431 doc_title_text
11 Q0 32587939 9 0.151116 doc_title_text
11 Q0 3173489 10 0.147288 doc_title_text



### Algorithms, Data Structures, Optimizations explanation:

  Step 1: Preprocessing (tokenization, stopwords, stemming)

    Algorithm

      - Lowercase the text.
      - Remove HTML/markup tags using a regex (<.*?>).
      - Remove punctuation via str.translate(...).
      - Remove numbers via regex (\d+).
      - Tokenize by splitting on whitespace.
      - Remove stopwords loaded from the provided stopwords.html file (parsed from the <pre>...</pre> section).
      - Apply Porter stemming to every remaining token. 

    Data structures
    
    - Stopwords stored as a Python list (then used for filtering).
    - Tokens stored as Python lists. 
  
    Optimization(s)
    
    - The preprocessing pipeline is lightweight (regex + string ops + list filtering) and is reused consistently for documents and queries.

Step 2: Indexing (inverted index + TF-IDF statistics)

  Algorithm

    - The corpus is read from corpus.jsonl, and for each document the indexed text is title + text concatenated.
    - Index building is done in two passes:

      1- Pass 1: compute document frequency df(t) and count total indexed docs N.
      2- Pass 2: build the inverted index term -> {doc_id: tf} and compute each document’s TF-IDF vector magnitude (used later for cosine similarity).

  Weighting

    - IDF uses a smooth formula:
      idf(t) = log ((N + 1) / (df(t) + 1)) + 1
    - TF uses either:
      raw tf (default), or log-tf if enabled: tf_w = 1 + log (tf) via the use_log_tf flag.
  
  Data Structures

    - df: Counter() for documents frequencies.
    - Inverted index: defaultdict(dict) storing postings as index[term][doc_id] = tf_w.
    - doc_lengths: dict mapping doc_id -> ||doc|| (TF-IDF vector magnitude, not token count).
    - Index persistence: saved/loaded with pickle (save_index, load_index).

  Optimizations

    - Precomputes document vector magnitudes (doc_lengths) during indexing, so cosine similarity is fast at query time.
    - Optional log-tf can be enabled with --use_log_tf to reduce the impact of very frequent terms.

Step 3: Retrieval and Ranking (TF-IDF cosine similarity)

  Algorithm

  For each query:
    - Preprocess the query using the same preprocessing function as documents.
    - Compute query term frequencies with Counter.
    - Compute query TF-IDF weights w(t,q) = tf(t,q) * idf(t).
    - Use the inverted index to score only candidate docs that contain at least one query term (via postings lookup).
    - Accumulate dot product: sum_t w(t,q) * w(t,d) where w(t,d) = tf(t,d) * idf(t).
    - Normalize by cosine similarity:
      score(d,q) = dot(d,q) / (||d|| * ||q||)
      ||d|| is taken from doc_lengths computed in indexing.
    - Sort scores descending and return top-100.

  Data Structures: 

    - Counter for query tf.
    - defaultdict(float) for accumulating scores.
    - Postings accessed from index[term] which is a dict of doc_id → tf_w.
  
  Optimizations:

    - Candidate restriction: only documents appearing in postings for at least one query term get scored (avoids scoring the whole corpus).
    - Uses precomputed doc_lengths (TF-IDF norms) to avoid recomputing document norms for every query.
