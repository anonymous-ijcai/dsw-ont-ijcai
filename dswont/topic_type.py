from collections import OrderedDict, defaultdict, Counter
from sklearn import cross_validation
from sklearn.dummy import DummyClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics.metrics import classification_report
from sklearn.svm.classes import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from dswont.dbpedia import title_to_uri, uri_to_title, to_category_uri
from dswont import util
from dswont.util import pos_tag, head_word_pos
import functools
import pandas as pd
import nltk
import logging
import re
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


""" A module for classifying the type of Wikipedia categories.

"""

# NodeType describes the type of the node in the category network.
# A node can be an INDIVIDUAL or a CLASS
#
# NodeType := INDIVIDUAL | CLASS
class NodeType(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __eq__(self, other):
        return self.value == other.value

    def __repr__(self):
        return 'NodeType({})'.format(self.name)

    def __hash__(self):
        return self.value

    @classmethod
    def by_name(cls, name):
        uppercase_name = name.upper()
        if hasattr(cls, uppercase_name):
            return getattr(cls, uppercase_name)
        else:
            raise ValueError("Unknown node type '{}'".format(name))


NodeType.INDIVIDUAL = NodeType('individual', 1)
NodeType.CLASS = NodeType('class', 2)


def read_node_type_labels(filename):
    """Read the 'ground truth' topic labeling from the file.

    Every line in the file consists of the topic title. The titles are delimited
    with underscore ('_'), and the label is one of {i, c, t}, like here:
        Machine_learning t
        Software c
        Linux i

    Here, i stands for INSTANCE, c stands for CLASS, and t stands for TOPIC:

    """

    translate_label = defaultdict(lambda: 'xxx')
    translate_label.update({'i': 'individual', 't': 'individual', 'c': 'class'})

    def parse_line(line):
        title, label = line.strip().split()
        topic = title_to_uri(title.replace('_', ' '), category=True)
        type_ = NodeType.by_name(translate_label[label])
        return topic, type_

    with open(filename, encoding='UTF-8', mode='r') as file:
        data = []
        for line in file.readlines():
            try:
                data.append(parse_line(line))
            except ValueError:
                logging.info('!! Could not parse: ' + line)
        return list(zip(*data))


def label_to_class(label: NodeType):
    return NodeType.CLASS == label

def read_ground_truth(filename):
    topic_uris, labels = read_node_type_labels(filename)
    classes = np.array([label_to_class(label) for label in labels], dtype=bool)
    return np.array(topic_uris), classes


# ## Features

# ### Converting the topics into features

def one_to_features(features, x):
    return OrderedDict((name, feature(x)) for name, feature in features.items())

def to_features(features, X):
    return pd.DataFrame([one_to_features(features, x) for x in X], 
                             columns=list(features.keys()))


# ### Implementation of the various features

def ends_with_ftr(letter):
    def ftr(uri):
        return uri_to_title(uri).lower().endswith(letter)
    return ftr
    
def generate_all_pos(topic_uris):
    return [pos for uri in topic_uris
            for token, pos in pos_tag(uri_to_title(uri))]

def generate_common_pos(all_pos):
    return [pos for pos, freq in nltk.FreqDist(all_pos).most_common()]
    
def last_word_pos_ftr(pos):
    def ftr(uri):
        return pos == pos_tag(uri_to_title(uri))[-1][1]
    return ftr

def first_word_pos_ftr(pos):
    def ftr(uri):
        return pos == pos_tag(uri_to_title(uri))[0][1]
    return ftr
        
def generate_all_suffices(topic_uris):
    return [word[-i:] for i in [1, 2, 3]
            for uri in topic_uris
            for word in nltk.word_tokenize(uri_to_title(uri))]

def generate_common_suffixes(all_suffixes):
    return [suffix for suffix, freq in nltk.FreqDist(all_suffixes).most_common()]

def last_word_suffix_ftr(suffix):
    def ftr(uri):
        return uri_to_title(uri).endswith(suffix)
    return ftr
    
def contains_pos_ftr(pos):
    def ftr(uri):
        tagging = pos_tag(uri_to_title(uri))
        pos_set = set(pos for token, pos in tagging)
        return pos in pos_set
    return ftr

def n_subcats_ftr(uri):
    return len(wiki_index.get_subcats(uri))

def n_supercats_ftr(uri):
    return len(wiki_index.get_subcats(uri))
    
def head_word_pos_ftr(pos):
    def ftr(uri):
        title = uri_to_title(uri)
        head_word, head_pos = head_word_pos(title)
        return head_pos == pos
    return ftr

def head_word_suffix_ftr(suffix):
    def ftr(uri):
        title = uri_to_title(uri)
        head_word, head_pos = head_word_pos(title)
        return head_word.endswith(suffix)
    return ftr


# ### Generating default features

def generate_default_features(topic_uris, **kwargs):
    features = OrderedDict()
    n_common_suffixes = kwargs.get('n_common_suffixes', 20)
    all_suffixes = generate_all_suffices(topic_uris)
    common_suffixes = generate_common_suffixes(all_suffixes)
    for suffix in common_suffixes[:n_common_suffixes]:
        features["head_word_suffix({})".format(suffix)] = head_word_suffix_ftr(suffix)
    return features


# ### Cross-validation procedure

_PARAM_GRID = {
    'C': [0.01, 0.002, 0.005, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
}

def train_cv_clf(topics_train, classes_train, features, n_folds=10, 
                 param_grid=_PARAM_GRID, tuned_clf=SVC(C=1, kernel='linear'),
                 scoring=util.weighted_f1, random_state=0):
    """Trains the topic type classifier, given the various parameters.
    
    """
    kf = cross_validation.KFold(len(topics_train), n_folds=n_folds, random_state=random_state)
    cv_clf = GridSearchCV(estimator=tuned_clf, param_grid=param_grid, cv=kf, scoring=scoring)
    topic_vectors_train = to_features(features, topics_train)
    cv_clf.fit(topic_vectors_train, classes_train)
    return cv_clf


def generate_topic_classifier():
    topic_uris, labels = read_node_type_labels(util.resource('labeled-topic-types-1000-dm.txt'))
    classes = [label_to_class(label) for label in labels]
    features = generate_default_features(topic_uris)
    
    cv_clf = train_cv_clf(topic_uris, classes, features)
    best_clf = cv_clf.best_estimator_
    
    best_clf.fit(to_features(features, topic_uris), classes)
    
    ground_truth_class = dict(zip(topic_uris, classes))
    
    def is_class(uri):
        if uri in ground_truth_class:
            return ground_truth_class[uri]
        else:
            return best_clf.predict(to_features(features, [uri]))[0]
        
    return is_class


class TopicTypePrediction(object):
    
    def __init__(self, topic_uris, classes, remember_gt=False, **kwargs):
        self._features = generate_default_features(topic_uris)
        self._cv_clf = train_cv_clf(topic_uris, classes, self._features, **kwargs)
        self._predict = self._cv_clf.best_estimator_.predict
        self._remember_gt = remember_gt
        if self._remember_gt:
            self._ground_truth = dict(zip(topic_uris, classes))
        
    def is_class(self, node):
        node = to_category_uri(node)
        if self._remember_gt and node in self._ground_truth:
            return self._ground_truth[node]
        else:
            return self._predict(to_features(self._features, [node]))[0]
    
def topic_type_prediction(topic_uris=None, classes=None, 
                          ground_truth_file=util.resource('labeled-topic-types-1000-dm.txt'),
                          n_folds=10, param_grid=_PARAM_GRID,
                          tuned_clf=LinearSVC(loss='l1'), scoring='f1',
                          random_state=0, remember_gt=False):
    if ground_truth_file and not topic_uris:
        topic_uris, classes = read_ground_truth(ground_truth_file)
    return TopicTypePrediction(topic_uris, classes,
                               n_folds=n_folds, param_grid=param_grid,
                               tuned_clf=tuned_clf, scoring=scoring,
                               random_state=random_state,
                               remember_gt=remember_gt)


def print_common_suffixes():
    gt_file = util.resource('labeled-topic-types-1000-dm.txt')
    topic_uris, _ = read_ground_truth(gt_file)
    all_suffixes = generate_all_suffices(topic_uris)
    print("Most common suffixes:", generate_common_suffixes(all_suffixes)[:20])


# print_common_suffixes()

