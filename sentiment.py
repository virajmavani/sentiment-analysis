import re
import nltk
import numpy as np


re_begin = re.compile(r"(’)(\w+ )")
re_end = re.compile(r"( \w+)(’)")
re_both = re.compile("(’)(\w+)(’)")

negation_ending_tokens = set([r"but", r"nevertheless", r"however", r".", r"?", r"!", r";"])

vocabulary = {}


def load_corpus(corpus_path):
    """

    :param corpus_path:
    :return:
    """
    file = open(corpus_path)

    lines = file.readlines()

    data = [tuple([line.split("\t")[0], int(line.split("\t")[1])]) for line in lines]

    return data


def tokenize(snippet):
    """

    :param snippet:
    :return:
    """
    out = re.sub(re_begin, r"\1 \2", snippet)
    out = re.sub(re_end, r"\1 \2", out)
    out = re.sub(re_both, r"\1 \2 \3", out)

    return out.split()


def tag_edits(tokenized_snippets):
    """

    :param tokenized_snippets:
    :return:
    """
    tag_edited_snippet = []
    edit_mode = False
    for word in tokenized_snippets:
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

    :param tokenized_snippet:
    :return:
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

    :param preprocessed_snippet:
    :return:
    """
    feature_vector = np.zeros(len(vocabulary))
    for token, pos in preprocessed_snippet:
        if token.find("EDIT_") == -1:
            feature_vector[vocabulary[token]] += 1
    return feature_vector


if __name__ == '__main__':
    corpus_path = "train.txt"
    corpus = load_corpus(corpus_path)
    tokenized_corpus = []
    for line in corpus:
        tokenized_corpus.append(tuple([tokenize(line[0]), line[1]]))

    for idx, line in enumerate(tokenized_corpus):
        edited_corpus = tag_edits(line[0])
        negated_edited_corpus = tag_negation(edited_corpus)
        tokenized_corpus[idx] = tuple([negated_edited_corpus, line[1]])

    idx = 0
    for line, label in tokenized_corpus:
        for token, pos in line:
            if token.find("EDIT_") == -1:
                if vocabulary.get(token, None) is None:
                    vocabulary[token] = idx
                    idx += 1

    X_train = np.empty([len(tokenized_corpus), len(vocabulary)])
    y_train = np.empty([len(tokenized_corpus), 1])
    for sample, label in tokenized_corpus:
        X = get_features(sample)
        y = label


