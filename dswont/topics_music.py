# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

# <codecell>

import os
# This hack makes ipython notebook import the modules correctly.
if (os.path.basename(os.getcwd()) == 'dswont'):
    os.chdir(os.path.dirname(os.getcwd()))

# <codecell>

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

import logging
logging.basicConfig(level=logging.WARN)
# Silence the verbose urllib logger.
logging.getLogger('requests.packages.urllib3.connectionpool').setLevel(logging.WARN)

from dswont import topics
from dswont import util
from dswont import dbpedia

# <codecell>

ROOT_CATEGORY_MUSIC = 'http://dbpedia.org/resource/Category:Music'
DEFAULT_SELECTION_DEPTH = 9
DEFAULT_RELATION_CACHE = topics.CategoryRelationCache(
    subcat_index_file=util.resource('wikipedia/uri-to-subcats-music'),
    supercat_index_file=util.resource('wikipedia/uri-to-supercats-music'))

def music_category_selection(**params):
    updated_params = {
        'root' : ROOT_CATEGORY_MUSIC,
        'relation_cache' : DEFAULT_RELATION_CACHE}
    updated_params.update(params)
    selection = topics.CategorySelection(**updated_params)
    selection.run()
    return selection

def precompute_full_selection(precomputed_data={}):
    if not 'full_selection' in precomputed_data:
        precomputed_data['full_selection'] = music_category_selection(max_depth=DEFAULT_SELECTION_DEPTH)
    return precomputed_data['full_selection']

# <codecell>

def make_topic_data_frame(selection):
    topic_df = pd.DataFrame({'topic':list(selection)})
    topic_df['depth'] = topic_df['topic'].apply(selection.get_depth)
    topic_df['title'] = topic_df['topic'].apply(dbpedia.to_title)
    topic_df = topic_df.reindex(columns=['topic', 'title', 'depth'])
    return topic_df

def precompute_unlabeled_topic_data_frame(precomputed_data={}):
    if not 'unlabeled_topic_df' in precomputed_data:
        precomputed_data['unlabeled_topic_df'] = make_topic_data_frame(precompute_full_selection())
    return precomputed_data['unlabeled_topic_df']

# <codecell>

def report_level_distribution(topic_df):
    return topic_df.groupby('depth').count()['title']

# <codecell>

# report_level_distribution(precompute_unlabeled_topic_data_frame())

# <codecell>

def sample_from_df(df, nrows = 10, seed=0):
    np.random.seed(seed)
    rows = np.random.choice(df.index.values, nrows, replace=False)
    return df.ix[rows]

# <codecell>

def sample_from_level(df, level, nrows=10, seed=0):
    return sample_from_df(df[df['depth']==level], nrows, seed)

# <codecell>

# sample_from_level(precompute_unlabeled_topic_data_frame(), 9, 100)['title'].values
# # Number of relevant topics : 1
# # 'Trauma Records albums'
# # 95% conf. interval: [0.000, 0.054]
# # Conclusion: could probably discard level 9

# sample_from_level(precompute_unlabeled_topic_data_frame(), 8, 100)['title'].values
# # Number of relevant topics : 21
# # 'Raised by Swans albums'
# # 'Low-importance Madonna articles'
# # 'Cub Country albums'
# # 'The Revolution Smile albums'
# # 'Island Records albums'
# # 'Polar Music albums'
# # 'Hannah Georgas albums'
# # 'Operas set in Turkey'
# # 'Category-Class Madonna articles'
# # 'Free multimedia codecs, containers, and splitters'
# # 'J Storm albums'
# # 'Portal-Class Madonna articles'
# # 'The Folk Implosion albums'
# # 'Hawksley Workman albums'
# # 'Skipping Girl Vinegar albums'
# # 'Loveless albums'
# # 'The Hours albums'
# # 'Nadine songs'
# # 'Two Hours Traffic albums'
# # 'Alternative rock groups from Maryland'
# # 'Cusco (band) albums'
# # 95% conf. interval: (0.135, 0.303)

None

# <codecell>

def clopper_pearson(k, n, alpha):
    """
    http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    alpha confidence intervals for a binomial distribution of k expected successes on n trials
    Clopper Pearson intervals are a conservative estimate.
    """
    lo = stats.beta.ppf(alpha/2, k, n-k+1)
    hi = stats.beta.ppf(1 - alpha/2, k+1, n-k)
    return lo, hi

# <codecell>

def generate_and_save_topics_for_labeling(filename):
    topic_data_sampler = topics.TrainingDataSelection(precompute_full_selection())
    topic_sample = topic_data_sampler.sample_paths_through_from_anywhere(1000)
    data_processing = topics.TrainingDataProcessing()
    topics_for_labeling = list(topic_sample)
    data_processing.save_topic_labels(topics_for_labeling, 
                                      [None] * len(topics_for_labeling), 
                                      topic_data_file, 
                                      topic_data_sampler)
    
# generate_and_save_topics_for_labeling()

def read_ground_truth_topic_labels():
    data_processing = topics.TrainingDataProcessing()
    return data_processing.read_topic_labels(util.resource('labeled-topics-music-1000-dm.txt'))

def make_labeled_topic_data_frame(selection, ground_truth_data):
    selection_df = pd.DataFrame({'topic':list(selection)})
    selection_df['depth'] = selection_df['topic'].apply(selection.get_depth)
    ground_truth_topic_relevance_topics, ground_truth_topic_relevance_relevance =\
        zip(*ground_truth_data.items())
    ground_truth_topic_relevance_df = pd.DataFrame(
        {'topic':ground_truth_topic_relevance_topics,
         'is_relevant':ground_truth_topic_relevance_relevance})
    selection_df = selection_df.merge(ground_truth_topic_relevance_df, how='outer')
    selection_df['title'] = selection_df['topic'].apply(dbpedia.to_title)
    selection_df =\
        selection_df.reindex(columns=['topic', 'title', 'depth', 'is_relevant'])
    return selection_df

def precompute_labeled_topic_data_frame(precomputed_data={}):
    if not 'labeled_topic_df' in precomputed_data:
        selection = precompute_full_selection()
        ground_truth_data = read_ground_truth_topic_labels()
        selection_df = make_labeled_topic_data_frame(selection, ground_truth_data)
        precomputed_data['labeled_topic_df'] = selection_df
    return precomputed_data['labeled_topic_df']

# <codecell>

# precompute_labeled_topic_data_frame().groupby('depth')['topic'].count()

# <codecell>

def apply_to_new_domain(selection_classifier:topics.CategorySelectionClassifier,
                        new_domain_full_selection:topics.CategorySelection):
        result = selection_classifier.copy()
        old_topic_classifier = result.selection._classifier
        result.full_selection = new_domain_full_selection
        result.max_depth = new_domain_full_selection._max_depth
        result.selection = topics.CategorySelection(
            new_domain_full_selection._root,
            old_topic_classifier,
            new_domain_full_selection._max_depth,
            new_domain_full_selection._relations)
        result.selection.run()
        return result

# <codecell>

# cs_clf = topics.default_trained_topic_selection_classifier()
# music_clf = apply_to_new_domain(cs_clf, precompute_full_selection())

# <codecell>

# categories, classes = zip(*read_ground_truth_topic_labels().items())
# print(topics.evaluate_classifier(music_clf, categories, classes, util.accuracy_score))
# print(topics.evaluate_classifier(music_clf, categories, classes, util.f1_pos_class))
# print(topics.evaluate_classifier(music_clf, categories, classes, util.f1_neg_class))
# print(topics.evaluate_classifier(music_clf, categories, classes, util.weighted_f1))

# <codecell>

def depth_based_selection(full_selection, depth):
    selection = topics.CategorySelection(
        full_selection._root,
        None,
        depth,
        full_selection._relations)
    selection.run()
    return selection

# for depth in range(1, 10):
#     clf = depth_based_selection(precompute_full_selection(), depth)
#     print("Depth:", depth)
#     print("Accuracy:", topics.evaluate_classifier(clf, categories, classes, util.accuracy_score))
#     print("Positive F1:", topics.evaluate_classifier(clf, categories, classes, util.f1_pos_class))
#     print("Negative F1:", topics.evaluate_classifier(clf, categories, classes, util.f1_neg_class))
#     print("Weighted F1:", topics.evaluate_classifier(clf, categories, classes, util.weighted_f1))

# <codecell>

from sklearn import cross_validation

def default_classifier_evaluation_params():
    categories_and_classes = list(read_ground_truth_topic_labels().items())
    np.random.shuffle(categories_and_classes)
    categories, classes = list(zip(*categories_and_classes))
    classes = np.array(classes, dtype=bool)
    categories = np.array(categories)
    inner_cross_validation = None  # No inner cross-validation.
    outer_cross_validation = topics.default_cross_validation
    def model_selection_measure(*args, **params):
        return util.weighted_f1(*args, **params)
    evaluation_measures = [util.accuracy_score, util.f1_pos_class, util.f1_neg_class, util.weighted_f1]
    return categories, classes, inner_cross_validation,\
           outer_cross_validation, model_selection_measure,\
           evaluation_measures

def evaluate_learning_based_classifier_cross_validated(training_size=None):
    np.random.seed(0)
    categories, classes, inner_cross_validation,\
        outer_cross_validation, model_selection_measure,\
        evaluation_measures = default_classifier_evaluation_params()
    def smaller_cross_validation(outputs):
        return cross_validation.StratifiedKFold(outputs, n_folds=2)
    inner_cross_validation = lambda outputs: cross_validation.StratifiedKFold(outputs, n_folds=3)
    param_grid = topics.new_training_params_cv()['param_grid']
    param_grid[0]['C'] = [0.25, 0.5, 1, 3, 7, 15]
    full_selection = precompute_full_selection()
    features = topics.default_features.copy()
    classifier_params = topics.default_classifier_params.copy()
    classifier_params['C'] = 0.25
    tuned_clf = topics.CategorySelectionClassifier(
        full_selection=full_selection,
        features=features,
        classifier_fn=topics.default_classifier,
        max_depth=full_selection._max_depth,
        instance_weight=lambda x: 1,
        **classifier_params)
    print(classes.dtype)
    return topics.train_evaluate_topic_classifier_cv(
        tuned_clf, categories, classes,
        inner_cross_validation,
        outer_cross_validation,
#         smaller_cross_validation,                                              
        model_selection_measure,
        evaluation_measures,
        param_grid=param_grid,
        learning=True,
        training_size=training_size)

# <codecell>

# metrics = evaluate_learning_based_classifier_cross_validated()

# <codecell>

# metric_names = ['accuracy', 'f1_pos', 'f1_neg', 'weighted_f1']
# for metric_name, metric in zip(metric_names, metrics):
#     print("{:<11s} : {:.3f} +- {:.3f}".format(metric_name, metric.mean(), metric.std()))

# <codecell>

def evaluate_depth_based_classifier_cross_validated(depth):
    np.random.seed(0)
    categories, classes, inner_cross_validation,\
        outer_cross_validation, model_selection_measure,\
        evaluation_measures = default_classifier_evaluation_params()
    def smaller_cross_validation(outputs):
        return cross_validation.StratifiedKFold(outputs, n_folds=2)
    tuned_clf = depth_based_selection(precompute_full_selection(), 6)
    return topics.train_evaluate_topic_classifier_cv(
        tuned_clf, categories, classes,
        inner_cross_validation,
#       smaller_cross_validation,
        outer_cross_validation,
        model_selection_measure,
        evaluation_measures,
        param_grid=None,
        learning=False)

# metrics = evaluate_depth_based_classifier_cross_validated(6)
# metric_names = ['accuracy', 'f1_pos', 'f1_neg', 'weighted_f1']
# for metric_name, metric in zip(metric_names, metrics):
#     print("{:<11s} : {:.3f} +- {:.3f}".format(metric_name, metric.mean(), metric.std()))

# <codecell>

def default_trained_topic_selection_classifier(precomputed_data={}):
    if 'music_clf' not in precomputed_data:
        full_selection = precompute_full_selection()
        training_data = read_ground_truth_topic_labels()
        training_params = topics.new_training_params()
        training_params['classifier_params']['C'] = 0.5
        training_params['instance_weight_fn'] = lambda x : 1
        clf = topics.train_topic_classifier(
                training_data.keys(), training_data.values(),
                full_selection,
                **training_params)
        precomputed_data['music_clf'] = clf
    return precomputed_data['music_clf']

