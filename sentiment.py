import re
import nltk
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


re_begin = re.compile(r"( ')(\w+ )")
re_end = re.compile(r"( \w+)(' )")
re_both = re.compile("( ')(\w+)(' )")

re_exp = re.compile(r"(' [0-9]+s\b)|(' em\b)|(' tis\b)")

negation_tokens = re.compile(r"(\bno\b)|(\bnever\b)|(\bnot\b)|(\bcannot\b)|(\w+n't)")
negation_ending_tokens = set([r"but", r"nevertheless", r"however", r".", r"?", r"!", r";"])

vocabulary = {}
reverse_vocabulary = {}


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
    found = re.findall(re_exp, out)
    if len(found) != 0:
        for elem in found[0]:
            if elem != '':
                out = out.replace(elem, elem.replace(" ", ""))
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
    tokens = tokenized_snippet.copy()
    for i in range(len(tokens)):
        if tokens[i][:5] == "EDIT_":
            tokens[i] = tokens[i][5:]

    tokens = [token for token in tokens if token]
    tokenized_snippet = [token for token in tokenized_snippet if token]
    tagged_tokens = nltk.pos_tag(tokens)
    negate = False
    for idx, token in enumerate(tagged_tokens):
        if negate:
            tokenized_snippet[idx] = "NOT_" + tokenized_snippet[idx]
        if negation_tokens.match(token[0]):
            negate = True
            if (idx < len(tokens) - 1) and tagged_tokens[idx+1][0] == r"only":
                negate = False
        if token[0] in negation_ending_tokens or token[1] == "JJR" or token[1] == "RBR":
            negate = False
    return [tuple([word, pos[1]]) for word, pos in zip(tokenized_snippet, tagged_tokens)]


def get_features(preprocessed_snippet):
    """
    Get 1D feature vector for the preprocessed snippet
    :param preprocessed_snippet: Preprocessed tokenized snippet with EDIT and NOT meta tags
    :return: 1D Feature Vector
    """
    mod_V = len(vocabulary)
    feature_vector = np.zeros(mod_V+3)
    vals = score_snippet(preprocessed_snippet, dal)
    for token, pos in preprocessed_snippet:
        if token.find("EDIT_") == -1:
            if vocabulary.get(token, None) is not None:
                feature_vector[vocabulary[token]] += 1
    feature_vector[mod_V] = vals[0]
    feature_vector[mod_V+1] = vals[1]
    feature_vector[mod_V+2] = vals[2]
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


def top_features(logreg_model, k):
    """
    Get top k features from the Logistic Regression Model coefficients
    :param logreg_model: sklearn.linear_model.LogisticRegression model
    :param k: Number of features we want
    :return: top_k features from the coefficients
    """
    weights = logreg_model.coef_
    weights = weights.tolist()[0]
    indexed_weights = [tuple([i, abs(weight)]) for i, weight in enumerate(weights[:(len(weights)-3)])]
    indexed_weights.sort(key=lambda x: x[1], reverse=True)
    req_weights = indexed_weights[:k]
    top_k = [tuple([reverse_vocabulary[i], weight]) for i, weight in req_weights]
    return top_k


def load_dal(dal_path):
    """

    :param dal_path:
    :return:
    """
    file = open(dal_path)
    lines = file.readlines()
    dal = {}
    for record in lines[1:]:
        word, activation, evaluation, imagery = record.split("\t")
        dal[word] = tuple([float(activation), float(evaluation), float(imagery)])
    return dal


def score_snippet(preprocessed_snippet, dal):
    """

    :param preprocessed_snippet:
    :param dal:
    :return:
    """
    activation = 0
    evaluation = 0
    imagery = 0
    count = 0
    for token, pos in preprocessed_snippet:
        if token.find("EDIT_") == -1:
            if dal.get(token, None) is not None:
                vals = dal[token]
                activation += vals[0]
                evaluation += vals[1]
                imagery += vals[2]
                count += 1

    if count != 0:
        return tuple([float(activation)/count, float(evaluation)/count, float(imagery)/count])
    else:
        return tuple([0., 0., 0.])


dal = load_dal("dict_of_affect.txt")

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
                    reverse_vocabulary[idx] = token
                    idx += 1

    X_train = np.empty([len(preprocessed_corpus), len(vocabulary)+3])
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

    X_test = np.empty([len(preprocessed_test_corpus), len(vocabulary)+3])
    y_test = np.empty([len(preprocessed_test_corpus), ])
    for idx, sample in enumerate(preprocessed_test_corpus):
        X = get_features(sample[0])
        y = sample[1]
        X_test[idx] = X
        y_test[idx] = y

    y_pred = model.predict(X_test)

    print("Precision, Recall and F-measure for Naive Bayes Model:", evaluate_predictions(y_pred, y_test))

    modelLR = LogisticRegression()
    modelLR.fit(X_train, y_train)

    y_pred_lr = modelLR.predict(X_test)

    print("Precision, Recall and F-measure for Logistic Regression Model:", evaluate_predictions(y_pred_lr, y_test))

    print(top_features(modelLR, 10))
