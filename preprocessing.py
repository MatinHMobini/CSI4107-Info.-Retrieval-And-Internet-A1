import json
import re
import string
from nltk.stem import PorterStemmer

# get stopwords from provided html file
with open("stopwords.html", "r", encoding="utf-8") as f:
    html_content = f.read()

# extract content inside <pre>...</pre>
match = re.search(r"<pre>(.*?)</pre>", html_content, re.DOTALL | re.IGNORECASE)
if match:
    pre_content = match.group(1)
else:
    pre_content = ""

# Split into words, strip whitespace, ignore empty lines
stopwords = [word.strip() for word in pre_content.splitlines() if word.strip()]

stemmer = PorterStemmer()

def preprocess_text(text):

    # lowercase
    text = text.lower()

    # remove markup
    text = re.sub(r'<.*?>', '', text)

    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # remove numbers
    text = re.sub(r'\d+', '', text)

    # tokenize
    tokens = text.split()

    # remove stopwords
    tokens = [token for token in tokens if token not in stopwords]

    # stemming
    tokens = [stemmer.stem(word) for word in tokens]

    return tokens


def preprocess_corpus(corpus):

    documents_tokens = {}

    with open(corpus, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)

            doc_id = doc["_id"]

            # combine title + abstract text
            text = doc.get("title", "") + " " + doc.get("text", "")

            tokens = preprocess_text(text)

            documents_tokens[doc_id] = tokens

    return documents_tokens