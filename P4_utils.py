from nltk.corpus import stopwords
import nltk
import pickle
import re
import numpy as np
import gensim

nltk.download('stopwords')

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'word_embeddings.tsv',
    'CORNEL_MOVIES_DIALOGS_DATASET': 'movies_stripped_lines.txt'
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""

    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """

    # Hint: you have already implemented a similar routine in the 3rd assignment.
    # Note that here you also need to know the dimension of the loaded embeddings.
    # When you load the embeddings, use numpy.float32 type as dtype

    ########################
    #### YOUR CODE HERE ####
    ########################
    entities = []
    vecs = []

    for line in open(embeddings_path):
        try:
            word, vec = line.split('\t', 1)
            entities.append(word)
            vecs.append([float(i) for i in vec.split('\t')])
        except:
            print(line)

    embeddings = gensim.models.keyedvectors.BaseKeyedVectors(100)
    embeddings.add(entities, vecs)
    return embeddings, len(vecs[-1])


def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""
    # Hint: you have already implemented exactly this function in the 3rd assignment.
    words = question.split()
    res = np.zeros(shape=dim)
    w_count = 0
    for w in words:
        if w in embeddings.vocab.keys():
            res = np.add(res, embeddings[w], dtype=np.float32)
            w_count += 1

    if w_count > 0:
        return np.true_divide(res, w_count)
    else:
        return res
    ########################
    #### YOUR CODE HERE ####
    ########################


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
