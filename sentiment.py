import re
import nltk
import numpy as np
from sklearn.naive_bayes import GaussianNB


re_begin = re.compile(r"(’)(\w+ )")
re_end = re.compile(r"( \w+)(’)")
re_both = re.compile("(’)(\w+)(’)")

negation_ending_tokens = set([r"but", r"nevertheless", r"however", r".", r"?", r"!", r";"])

vocabulary = {}


def load_corpus(corpus_path):
    """
    Loading a .txt file as corpus
    :param corpus_path: path to txt file
    :return: list of tuples of the form (snippet, label) in file
    """
    file = open(corpus_path)

    lines = file.readlines()

    data = [tuple([line.split("\t")[0], int(line.split("\t")[1])]) for line in lines]

    return data


def tokenize(snippet):
    """
    Tokenizing a snippet of text with whitespaces between apostrophe and words
    :param snippet: string of text to be tokenized
    :return: processed tokenized snippet
    """
    out = re.sub(re_begin, r"\1 \2", snippet)
    out = re.sub(re_end, r"\1 \2", out)
    out = re.sub(re_both, r"\1 \2 \3", out)

    return out.split()


def tag_edits(tokenized_snippet):
    """
    Add edit tags to words added by editor
    :param tokenized_snippet: list of words split on whitespaces
    :return: tokenized snippet with EDIT meta tags
    """
    tag_edited_snippet = []
    edit_mode = False
    for word in tokenized_snippet:
        if word.find("[") != -1:
            edit_mode = True
            word = word.replace("[", "")
        if edit_mode:
            word = "EDIT_" + word
        if word.find("]") != -1:
            edit_mode = False
            word = word.replace("]", "")
        tag_edited_snippet.append(word)
    return tag_edited_snippet


def tag_negation(tokenized_snippet):
    """
    Add negation tags to words followed by a negation word
    :param tokenized_snippet: processed snippet split on whitespaces with EDIT meta tag
    :return: processed tokenized snippet with EDIT and NOT meta tags
    """
    negation_words = set([r"not", r"no", r"cannot", r"never"])
    tokens = tokenized_snippet.copy()
    for i in range(len(tokens)):
        if tokens[i][:5] == "EDIT_":
            tokens[i] = tokens[i][5:]

    tokens = [token for token in tokens if token]
    tokenized_snippet = [token for token in tokenized_snippet if token]
    tagged_tokens = nltk.pos_tag(tokens)
    negate = False
    for idx, token in enumerate(tagged_tokens):
        if token[0] in negation_words or token[0][(len(token)-3):] == r"n't":
            negate = True
            if (idx < len(tokens) - 1) and tagged_tokens[idx+1][0] == r"only":
                negate = False
        if negate:
            tokenized_snippet[idx] = "NOT_" + tokenized_snippet[idx]
        if token[0] in negation_ending_tokens or token[1] == "JJR" or token[1] == "RBR":
            negate = False
    return [tuple([word, pos[1]]) for word, pos in zip(tokenized_snippet, tagged_tokens)]


def get_features(preprocessed_snippet):
    """
    Get 1D feature vector for the preprocessed snippet
    :param preprocessed_snippet: Preprocessed tokenized snippet with EDIT and NOT meta tags
    :return: 1D Feature Vector
    """
    feature_vector = np.zeros(len(vocabulary))
    for token, pos in preprocessed_snippet:
        if token.find("EDIT_") == -1:
            if vocabulary.get(token, None) is not None:
                feature_vector[vocabulary[token]] += 1
    return feature_vector


def normalize(X):
    """
    Min-max normalize array to get values in 0-1 range
    :param X: 2D feature matrix
    :return: Normalized 2D feature matrix
    """
    n = X.shape[1]
    for i in range(n):
        col = X[:, i]
        X[:, i] = (col - col.min()) / (col.max() - col.min())

    return X


def evaluate_predictions(y_pred, y_true):
    """
    Find precision, recall and f-measure of model on test set
    :param y_pred: Model predictions on test set
    :param y_true: True labels
    :return: Tuple of (precision, recall, fmeasure)
    """
    tp = 0
    fp = 0
    fn = 0
    for y_p, y_t in zip(y_pred, y_true):
        if y_p == y_t == 1:
            tp += 1
        if y_t == 0 and y_p == 1:
            fp += 1
        if y_t == 1 and y_p == 0:
            fn += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fmeasure = (2 * (precision * recall)) / (precision + recall)
    return tuple([precision, recall, fmeasure])


if __name__ == '__main__':

    corpus_path = "train.txt"
    corpus = load_corpus(corpus_path)
    preprocessed_corpus = []
    for line in corpus:
        preprocessed_corpus.append(tuple([tokenize(line[0]), line[1]]))

    for idx, line in enumerate(preprocessed_corpus):
        edited_corpus = tag_edits(line[0])
        negated_edited_corpus = tag_negation(edited_corpus)
        preprocessed_corpus[idx] = tuple([negated_edited_corpus, line[1]])

    idx = 0
    for line, label in preprocessed_corpus:
        for token, pos in line:
            if token.find("EDIT_") == -1:
                if vocabulary.get(token, None) is None:
                    vocabulary[token] = idx
                    idx += 1

    X_train = np.empty([len(preprocessed_corpus), len(vocabulary)])
    y_train = np.empty([len(preprocessed_corpus), ])
    for idx, sample in enumerate(preprocessed_corpus):
        X = get_features(sample[0])
        y = sample[1]
        X_train[idx] = X
        y_train[idx] = y

    X_train = normalize(X_train)

    model = GaussianNB()
    model.fit(X_train, y_train)

    test_corpus_path = "test.txt"
    test_corpus = load_corpus(test_corpus_path)

    preprocessed_test_corpus = []
    for line in test_corpus:
        preprocessed_test_corpus.append(tuple([tokenize(line[0]), line[1]]))

    for idx, line in enumerate(preprocessed_test_corpus):
        edited_corpus = tag_edits(line[0])
        negated_edited_corpus = tag_negation(edited_corpus)
        preprocessed_test_corpus[idx] = tuple([negated_edited_corpus, line[1]])

    X_test = np.empty([len(preprocessed_test_corpus), len(vocabulary)])
    y_test = np.empty([len(preprocessed_test_corpus), ])
    for idx, sample in enumerate(preprocessed_test_corpus):
        X = get_features(sample[0])
        y = sample[1]
        X_test[idx] = X
        y_test[idx] = y

    y_pred = model.predict(X_test)

    print(evaluate_predictions(y_pred, y_test))
