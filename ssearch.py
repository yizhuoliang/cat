from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import nmslib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

def __create_index(embeddings):
    # Initialize a new index
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(embeddings)
    index.createIndex({'post': 2})
    return index

def __semantic_search(query: str, index, sentence_list, top_k: int = 5):
    # Convert the query to embeddings
    query_embedding = model.encode([query])
    ids, distances = index.knnQuery(query_embedding, k=top_k)
    return [sentence_list[i] for i in ids], distances

def __extract_keywords(sentence: str):
    stop_words = set(stopwords.words('english')) 
    tokenized = word_tokenize(sentence)
    tagged = pos_tag(tokenized)
    return [word for word, pos in tagged if (pos.startswith('N') or pos.startswith('V')) and word.lower() not in stop_words]

def search_and_add(new_line: str) -> List[str]:
    # Extract keywords from new line
    keywords = __extract_keywords(new_line)
    print(keywords)

    # Perform semantic search with keywords
    results, distances = __semantic_search(' '.join(keywords), search_index, conversational_history)
    
    # Now add the new line to the conversational history
    conversational_history.append(new_line)

    # Add the new line's embedding to the search index
    new_line_embedding = model.encode([new_line])
    search_index.addDataPoint(len(conversational_history)-1, new_line_embedding[0])

    # Return the search results
    return results, distances

def init():
    # Loading the transformer model
    global model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    global stop_words
    stop_words = set(stopwords.words('english'))

    global conversational_history
    conversational_history = {}

    sentence_embeddings = model.encode(conversational_history)
    sentence_embeddings_np = np.array(sentence_embeddings)
    global search_index
    search_index = __create_index(sentence_embeddings_np)





