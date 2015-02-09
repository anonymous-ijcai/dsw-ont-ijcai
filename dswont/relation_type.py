from collections import OrderedDict, defaultdict, Counter
from sklearn import cross_validation
from sklearn.dummy import DummyClassifier
from sklearn.base import BaseEstimator, clone
from sklearn.grid_search import GridSearchCV, BaseSearchCV
from sklearn.metrics.metrics import classification_report
from sklearn import metrics
from sklearn.svm.classes import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.base import LinearModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from dswont.dbpedia import title_to_uri, uri_to_title, to_title
from dswont import util
from dswont.util import pos_tag, head_word_pos, weighted_f1
from dswont import topics
from dswont import topic_type
from dswont import wikiapi
import collections
import functools
import pandas as pd
import nltk
import re
from collections import Set
from nltk.corpus import wordnet as wn
from matplotlib import pyplot
import numpy as np
import random

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import logging


""" A module for classifying relations between the Wikipedia categories.

"""


# Reading in data

# ### Data-reading functions

# RelationType describes the type of the relation between the two nodes in the category network.
# A relation can be an SUBCLASS, PART_OF, INSTANCE_OF, and RELATED
#
# RELATION_TYPE := SUBCLASS | PART_OF | INSTANCE_OF | RELATED
@functools.total_ordering
class RelationType(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __eq__(self, other):
        return self.value == other.value
    
    def __lt__(self, other):
        return self.value < other.value

    def __repr__(self):
        return 'RelationType({})'.format(self.name)

    def __hash__(self):
        return self.value

    @classmethod
    def by_name(cls, name):
        uppercase_name = name.upper()
        if hasattr(cls, uppercase_name):
            return getattr(cls, uppercase_name)
        else:
            raise ValueError("Unknown relation type '{}'".format(name))

RelationType.SUBCLASS = RelationType('subclass', 1)
RelationType.PART = RelationType('part', 2)
RelationType.INSTANCE = RelationType('instance', 3)
RelationType.RELATED = RelationType('related', 4)


def read_relation_type_labels(filename):
    """Read the 'ground truth' between-topic relation labeling from the file.

    Every line in the file consists of a pair of topic titles and the label.
    The label is one of {s, p, r, i}, like here:
        Ontology_(information_science) -> Ontology_learning_(computer_science) r
        Microcomputers -> Portable_computers s
        Problem_solving -> Game_theory p
        Computing_platforms -> Inferno_(operating_system) i
    Here, stands for SUBCLASS, p stands for PART, i stands for INSTANCE, and r stands for RELATED.

    """

    translate_label = defaultdict(lambda: 'xxx')
    translate_label.update({'s': 'subclass', 'sc': 'subclass', 
                            'p': 'part', 'i': 'instance', 'r': 'related'})

    def parse_line(line):
        title_parent, title_child_label = line.split(' -> ')
        title_child, label = title_child_label.strip().split()
        topic_parent = title_parent.replace('_', ' ')
        topic_child = title_child.replace('_', ' ')
        type_ = RelationType.by_name(translate_label[label]).name.capitalize()
        return topic_parent, topic_child, type_

    with open(filename, encoding='UTF-8', mode='r') as file:
        data = []
        for line in file.readlines():
            try:
                data.append(parse_line(line))
            except ValueError:
                logging.info('!! Could not parse: ' + line)
        return pd.DataFrame(data, columns=['parent', 'child', 'relation_type'])


# ### Reading in the data, predicting the node types, fixing the prediction errors

def read_ground_truth_data(filename=util.resource('labeled-relations-new-1000-dm.txt')):
    data = read_relation_type_labels(filename)

    is_class = topic_type.topic_type_prediction(remember_gt=True).is_class
    
    def node_to_type_char(title):
        uri = title_to_uri(title, category=True)
        return 'Class' if is_class(uri) else 'Individual'

    data['parent_type'] = data['parent'].apply(node_to_type_char)
    data['child_type'] = data['child'].apply(node_to_type_char)

    # fixing the incorrectly classified nodes
    data.ix[data['parent'] == 'Computational linguistics', 'parent_type'] = 'Individual'
    data.ix[data['child'] == 'Twitter', 'child_type'] = 'Individual'
    data.ix[data['child'] == 'Populous', 'child_type'] = 'Individual'
    data.ix[data['child'] == 'Canary Islands', 'child_type'] = 'Individual'
    data.ix[data['parent'] == 'Algorithms and data structures', 'parent_type'] = 'Individual'
    data.ix[data['child'] == 'Computational statistics', 'child_type'] = 'Individual'
    data.ix[data['child'] == 'Digital electronics', 'child_type'] = 'Individual'
    data.ix[data['parent'] == 'Computer storage media', 'parent_type'] = 'Class'
    data.ix[data['parent'] == 'IEC 61131', 'parent_type'] = 'Class'
    data.ix[data['child'] == 'IEC 61131', 'child_type'] = 'Class'
    data.ix[data['parent'] == 'Digital cameras', 'parent_type'] = 'Class'
    data.ix[data['child'] == 'Sony cameras', 'child_type'] = 'Class'
    data.ix[data['parent'] == 'Computer companies', 'parent_type'] = 'Class'
    data.ix[data['parent'] == 'Computer companies', 'parent_type'] = 'Class'
    data.ix[data['parent'] == 'Mathematics of computing', 'parent_type'] = 'Individual'
    data.ix[data['parent'] == '1970s in computer science', 'parent_type'] = 'Individual'
    data.ix[data['parent'] == '1980s in computer science', 'parent_type'] = 'Individual'
    data.ix[data['parent'] == 'Health informatics', 'parent_type'] = 'Individual'
    data.ix[data['parent'] == '1990s in video gaming', 'parent_type'] = 'Individual'
    data.ix[data['child'] == 'Anime based on video games', 'child_type'] = 'Class'
    data.ix[data['child'] == 'Android cameras with optical zoom', 'child_type'] = 'Class'
    data.ix[data['parent'] == 'Algorithms', 'parent_type'] = 'Class'
    data.ix[data['child'] == 'Algorithms', 'child_type'] = 'Class'
    
    return data


# Defining features

stemmer = nltk.stem.snowball.EnglishStemmer()

def stem_word(word: str):
    return stemmer.stem(word)

def start_start_pattern(parent_words, child_words):
    matchFound = False
    for parent_word, child_word in zip(parent_words, child_words):
        if stem_word(parent_word) == stem_word(child_word):
            matchFound = True
        else:
            return matchFound
    else:
        return False
    
assert start_start_pattern('Data processing'.split(), 'Data management'.split())
assert not start_start_pattern('Internet protocols'.split(), 'Internet Protocol'.split())
assert not start_start_pattern('Operating systems'.split(),	'Operating system technology'.split())
assert not start_start_pattern('Artificial intelligence'.split(), 'Machine learning'.split())

def end_end_pattern(parent_words, child_words):
    return start_start_pattern(list(reversed(parent_words)), list(reversed(child_words)))

assert end_end_pattern('Mac OS user interface'.split(), 'OS X user interface'.split())
assert end_end_pattern('Virtual reality'.split(), 'Mixed reality'.split())
assert not end_end_pattern('User interfaces'.split(), 'OS X user interface'.split())
assert not end_end_pattern('Operating systems'.split(), 'Operating system technology'.split())

def all_all_pattern(parent_words, child_words):
    if len(parent_words) != len(child_words):
        return False
    else:
        for parent_word, child_word in zip(parent_words, child_words):
            if stem_word(parent_word) != stem_word(child_word):
                return False
        else:
            return True
        
assert all_all_pattern('Internet protocols'.split(), 'Internet Protocol'.split())
assert not all_all_pattern('Data'.split(), 'Data management'.split())
        
def start_all_pattern(parent_words, child_words):
    if len(parent_words) <= len(child_words):
        return False
    for parent_word, child_word in zip(parent_words, child_words):
        if stem_word(parent_word) != stem_word(child_word):
            return False
    else:
        return True

assert start_all_pattern('Learning in computer vision'.split(), 'Learning'.split())
assert not start_all_pattern('Learning in computer vision'.split(), 'Computer vision'.split())
assert not start_all_pattern('Learning in computer vision'.split(), 'Computer vision'.split())
assert not start_all_pattern('Data'.split(), 'Data management'.split())
assert not start_all_pattern('Software engineering'.split(), 'Software quality'.split())
            
def all_start_pattern(parent_words, child_words):
    return start_all_pattern(child_words, parent_words)

def all_end_pattern(parent_words, child_words):
    return all_start_pattern(list(reversed(parent_words)), list(reversed(child_words)))

def end_all_pattern(parent_words, child_words):
    return start_all_pattern(list(reversed(parent_words)), list(reversed(child_words)))

assert end_all_pattern('Learning in computer vision'.split(), 'Computer vision'.split())

def start_end_pattern(parent_words, child_words):
    for i in range(len(child_words)):
        if start_all_pattern(parent_words, child_words[i:]):
            return True
    else:
        return False

assert start_end_pattern('Image segmentation algorithms'.split(), 'Medical image segmentation'.split())
assert not start_end_pattern('Data'.split(), 'Data management'.split())
assert not start_end_pattern('Data'.split(), 'Data'.split())
assert not start_end_pattern('Mac OS user interface'.split(), 'OS X user interface'.split())

def end_start_pattern(parent_words, child_words):
    return start_end_pattern(child_words, parent_words)


# ### Some more features

def word_list(string: str):
    return [token
            for token in nltk.tokenize.word_tokenize(string)]

def head_word_same(row):
    return row['parent_head_pos'][0].lower() == row['child_head_pos'][0].lower()

def first_word_same(row):
    return row['parent_pos'][0][0].lower() == row['child_pos'][0][0].lower()

def head_stem_same(row):
    return stem_word(row['parent_pos'][0][0]) == stem_word(row['child_pos'][0][0])

def max_common_suffix(word1: str, word2: str):
    if word1 and word2 and word1[-1] == word2[-1]:
        return max_common_suffix(word1[:-1], word2[:-1]) + word1[-1]
    else:
        return ""
    
def suffix_score(word1: str, word2: str):
    stem1 = stem_word(word1)
    stem2 = stem_word(word2)
    return len(max_common_suffix(stem1, stem2)) / min(len(stem1), len(stem2))

def head_suffix_score(row):
    return suffix_score(row['parent_head_pos'][0], row['child_head_pos'][0])

def jaccard_sim(set1: Set, set2: Set):
    return len(set1.intersection(set2)) / len(set1.union(set2))

def stem_jaccard(row):
    stems1 = set(stem_word(word) for word in row['parent_words'])
    stems2 = set(stem_word(word) for word in row['child_words'])
    return jaccard_sim(stems1, stems2)

def wn_simialrity_fn(similarity):
    def similarity_fn(word1, word2):
        scores = [0];
        for sense1 in wn.synsets(word1):
            pos = sense1.pos()
            if pos == wn.ADJ_SAT:
                pos = wn.ADJ
            for sense2 in wn.synsets(word2, pos):
                try:
                    score = similarity(sense1, sense2)
                    if score:
                        scores.append(score)
                except Exception:
                    continue
        return max(scores)
    return similarity_fn

def head_wn_similarity_fn(similarity_fn):
    similarity = wn_simialrity_fn(similarity_fn)
    def similarity_fn(row):
        parent_head = row['parent_head_pos'][0]
        child_head = row['child_head_pos'][0]
        if stem_word(parent_head) == stem_word(child_head):
            return 0
        return similarity(parent_head, child_head)
    return similarity_fn

def avg_pairwise_wn_similarity_fn(similarity_fn):
    similarity = wn_simialrity_fn(similarity_fn)
    def similarity_fn(row):
        scores = []
        for parent_word in row['parent_words']:
            for child_word in row['child_words']:
                scores.append(similarity(parent_word, child_word))
        return np.mean(scores) 
    return similarity_fn

def parent_child_pattern_fn(pattern_fn):
    def pattern_feature_fn(row):
        return pattern_fn(row['parent_words'], row['child_words'])
    return pattern_feature_fn

def possible_hypernym_synsets(lemma):
    result = set()
    for synset in wn.synsets(lemma):
        result.update(hypernym_sense for hypernym_sense, distance in synset.hypernym_distances())
    return result

def could_be_hypernym(word, candidate_hypernym):
    if (stem_word(word) == stem_word(candidate_hypernym)):
        return True
    candidate_hypernym_synsets = set(wn.synsets(candidate_hypernym))
    possible_hypernyms = possible_hypernym_synsets(word)
    return candidate_hypernym_synsets.intersection(possible_hypernyms)

def head_word_hypernym(row):
    parent_head = row['parent_head_pos'][0]
    child_head = row['child_head_pos'][0]
    return bool(could_be_hypernym(child_head, parent_head))

def head_word_hyponym(row):
    parent_head = row['parent_head_pos'][0]
    child_head = row['child_head_pos'][0]
    return bool(could_be_hypernym(parent_head, child_head))

def hypernym_feature(row):
    def result(hypernym_found, hypernym_is_parent, hypernym_parent_head, hypernym_child_head):
        return pd.Series(OrderedDict(hypernym_found=hypernym_found, hypernym_is_parent=hypernym_is_parent,
                         hypernym_parent_head=hypernym_parent_head, hypernym_child_head=hypernym_child_head))
    child_head = row['child_head_pos']
    parent_head = row['parent_head_pos']
    child_pos = [child_head]; child_pos.extend(row['child_pos'])
    parent_pos = [parent_head]; parent_pos.extend(row['parent_pos'])
    for pword, _ in parent_pos:
        for cword, _ in child_pos:
            if (could_be_hypernym(cword, pword)):
                return result(True, True, pword == parent_head[0], cword == child_head[0])
            if (could_be_hypernym(pword, cword)):
                return result(True, False, pword == parent_head[0], cword == child_head[0])
    return result(False, False, False, False)


# ### The method for generating the features

def to_features(data: pd.DataFrame):
    result = data.copy()
    
    result['parent_words'] = result['parent'].apply(word_list)
    result['child_words'] = result['child'].apply(word_list)

    result['parent_pos'] = result['parent_words'].apply(pos_tag)
    result['child_pos'] = result['child_words'].apply(pos_tag)
    
    result['parent_head_pos'] = result['parent_pos'].apply(head_word_pos)
    result['child_head_pos'] = result['child_pos'].apply(head_word_pos)
    
    result['head_word_same'] = result.apply(head_word_same, axis=1)
    result['first_word_same'] = result.apply(first_word_same, axis=1)
    result['head_stem_same'] = result.apply(head_stem_same, axis=1)
    result['head_suffix_score'] = result.apply(head_suffix_score, axis=1)
    result['stem_jaccard'] = result.apply(stem_jaccard, axis=1)
    result['head_wn_path_sim'] = result.apply(head_wn_similarity_fn(wn.path_similarity), axis=1)
    result['head_wn_jcn_sim'] = result.apply(head_wn_similarity_fn(wn.jcn_similarity), axis=1)
    result['head_wn_lch_sim'] = result.apply(head_wn_similarity_fn(wn.lch_similarity), axis=1)
    result['head_wn_lin_sim'] = result.apply(head_wn_similarity_fn(wn.lin_similarity), axis=1)
    result['head_wn_res_sim'] = result.apply(head_wn_similarity_fn(wn.res_similarity), axis=1)    
    result['head_wn_wup_sim'] = result.apply(head_wn_similarity_fn(wn.wup_similarity), axis=1)    
    
    result['avg_wn_path_sim'] = result.apply(avg_pairwise_wn_similarity_fn(wn.path_similarity), axis=1)
    result['avg_wn_lch_sim'] = result.apply(avg_pairwise_wn_similarity_fn(wn.lch_similarity), axis=1)
    result['avg_wn_wup_sim'] = result.apply(avg_pairwise_wn_similarity_fn(wn.wup_similarity), axis=1)
    
    result['start_start_ptrn'] = result.apply(parent_child_pattern_fn(start_start_pattern), axis=1)
    result['start_end_ptrn'] = result.apply(parent_child_pattern_fn(start_end_pattern), axis=1)
    result['start_all_ptrn'] = result.apply(parent_child_pattern_fn(start_all_pattern), axis=1)
    result['end_start_ptrn'] = result.apply(parent_child_pattern_fn(end_start_pattern), axis=1)
    result['end_end_ptrn'] = result.apply(parent_child_pattern_fn(end_end_pattern), axis=1)
    result['end_all_ptrn'] = result.apply(parent_child_pattern_fn(end_all_pattern), axis=1)
    result['all_start_ptrn'] = result.apply(parent_child_pattern_fn(all_start_pattern), axis=1)
    result['all_end_ptrn'] = result.apply(parent_child_pattern_fn(all_end_pattern), axis=1)
    result['all_all_ptrn'] = result.apply(parent_child_pattern_fn(all_all_pattern), axis=1)
    
    result['parent_is_hypernym'] = result.apply(head_word_hypernym, axis=1)
    result['parent_is_hyponym'] = result.apply(head_word_hyponym, axis=1)
    
    hfeatures = result.apply(hypernym_feature, axis=1)
    hypernym_ftr_columns = ['hypernym_is_parent', 'hypernym_child_head', 'hypernym_parent_head', 'hypernym_found']
    for col in hypernym_ftr_columns:
        result[col] = hfeatures[col]
        
    result['hypernym_head_head'] = result['hypernym_child_head'] & result['hypernym_parent_head']
    result['hypernym_head_nonhead'] = result['hypernym_child_head'] & ~result['hypernym_parent_head']
    result['hypernym_nonhead_head'] = ~result['hypernym_child_head'] & result['hypernym_parent_head']
    result['hypernym_nonhead_nonhead'] = ~result['hypernym_child_head'] & ~result['hypernym_parent_head']

    return result


# Running the cross-validation procedure

# ### Computing the features on the whole dataset

relevant_feature_names = [
            'head_word_same', 
            'first_word_same', 
            'head_stem_same',
            'head_suffix_score',
            'stem_jaccard', 'head_wn_path_sim', 'head_wn_jcn_sim', 
            'head_wn_lch_sim',
            
            'head_wn_lin_sim', 'head_wn_res_sim', 

            'head_wn_wup_sim',
            'avg_wn_path_sim', 'avg_wn_lch_sim', 'avg_wn_wup_sim',
            'start_start_ptrn', 'start_end_ptrn', 'start_all_ptrn',
            'end_start_ptrn', 'end_end_ptrn', 'end_all_ptrn',
            'all_start_ptrn', 'all_end_ptrn', 'all_all_ptrn',
            'hypernym_found', 'hypernym_is_parent', 'hypernym_parent_head', 'hypernym_child_head',
            'hypernym_head_head',
            'hypernym_head_nonhead', 'hypernym_nonhead_head',
            'hypernym_nonhead_nonhead',
            ]


# ## Running the cross-validation procedure

class Task(object):
    SUBCLASS = 'Class', 'Class', 'Subclass'
    INSTANCE = 'Class', 'Individual', 'Instance'
    CLASS_PART = 'Individual', 'Class', 'Part'
    IND_PART = 'Individual', 'Individual', 'Part'

    
class Clf(object):
    SVC_RBF = SVC(kernel='rbf', class_weight=None, random_state=0)
    SVC_RBF_CW = SVC(kernel='rbf', class_weight='auto', random_state=0)
    LINEAR_L1 = LinearSVC(loss='l1', random_state=0, class_weight=None)
    LINEAR_L1_CW = LinearSVC(loss='l1', random_state=0, class_weight='auto')
    LINEAR_SVC = SVC(kernel='linear', random_state=0, class_weight='auto')
    TREE = DecisionTreeClassifier(random_state=0)
    RF = RandomForestClassifier(random_state=0)
    MAJORITY = DummyClassifier(strategy='most_frequent')
    RANDOM = DummyClassifier(strategy='stratified')
    ADABOOST = AdaBoostClassifier(random_state=0)
    LR = LogisticRegression()

    
def limit_to_task_data(data, task: Task):
    parent_type, child_type, relation_type = task
    task_data = data[data['parent_type'] == parent_type][data['child_type'] == child_type].copy()

    def encode_relation_type(rel):
        return int(rel == relation_type)

    task_data['relation_type'] = task_data['relation_type'].apply(encode_relation_type)
    return task_data

    
def get_task_data(data, task: Task, feature_names=relevant_feature_names):
    task_data = limit_to_task_data(data, task)
    predictors = task_data[feature_names]
    response = task_data['relation_type']
    return predictors, response

    
def default_param_grid(tuned_clf):

    param_grid = {}

    if hasattr(tuned_clf, 'C'):
        param_grid['C'] = [0.0001, 0.001, 0.002, 0.005, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

    if isinstance(tuned_clf, SVC) and tuned_clf.kernel == 'rbf':
        param_grid['gamma'] = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

    if isinstance(tuned_clf, DecisionTreeClassifier):
        param_grid['max_depth'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    if isinstance(tuned_clf, RandomForestClassifier):
        param_grid['n_estimators'] = [10, 20, 50, 100, 500]
        param_grid['max_depth'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    if isinstance(tuned_clf, AdaBoostClassifier):
        param_grid['n_estimators'] = [10, 20, 50, 100, 200, 500]
        param_grid['learning_rate'] = [0.1, 0.2, 0.5, 1]
            
    return param_grid
        
    
def train_classifier(predictors, response, feature_names=relevant_feature_names, tuned_clf=Clf.LINEAR_SVC,
                     param_grid=None, test_size=0.5, scoring=weighted_f1, random_state=0):
    param_grid = param_grid or default_param_grid(tuned_clf)
    kf_cv = cross_validation.StratifiedKFold(response, n_folds=10, shuffle=True, random_state=random_state)
    cv_clf = GridSearchCV(estimator=tuned_clf, param_grid=param_grid, cv=kf_cv, scoring=scoring)
    cv_clf.fit(predictors, response)
    
    return cv_clf


class RelationTypePrediction(object):
    
    def __init__(self, topic_pairs, rel_types, feature_names=relevant_feature_names, 
                 tuned_clf=Clf.LINEAR_SVC, random_state=0):
        parents, children = zip(*topic_pairs)
        raw_data = pd.DataFrame({'parent': parents, 'child': children, 'relation_type': rel_types})
        data = to_features(raw_data)
        predictors = data[feature_names]
        response = data['relation_type']
        self.feature_names = feature_names
        self._cv_clf = train_classifier(predictors, response, feature_names, tuned_clf=tuned_clf,
                                        param_grid=None, test_size=0.5, scoring=weighted_f1, random_state=random_state)
        self._predict = self._cv_clf.best_estimator_.predict
    
    def predict(self, parent, child):
        parent = to_title(parent)
        child = to_title(child)
        raw_data = pd.DataFrame({'parent': [parent], 'child': [child]})
        return self._predict(to_features(raw_data)[self.feature_names])[0]


# all_data = read_ground_truth_data()
# data = limit_to_task_data(all_data, Task.SUBCLASS)
# parent = data['parent']; child=data['child']; relation = data['relation_type']
# classifier = RelationTypePrediction(zip(parent.values, child.values), relation.values)

# classifier.predict('Machine learning', 'Statistical methods')

