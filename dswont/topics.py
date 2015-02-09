import collections
from contextlib import closing
import copy
import itertools
import logging
from matplotlib import pyplot as plt
import nltk
import numpy as np
import pandas as pd
import random
import semidbm

from sklearn.svm import SVC
from sklearn.base import BaseEstimator
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model.base import LinearModel
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import cross_validation

from dswont import dbpedia
from dswont.dbpedia import uri_to_title, title_to_uri, to_title, to_category_uri
from dswont import util
from dswont import wikiapi


class WikipediaGraphIndex(object):
    """Provides navigation over the Wikipedia category graph.

    Returns the subcategories and the super-categories of a given category.
    The data is retrieved from Wikipedia and cached into a database.
    The categories are represented by their DBpedia URIs, such as, for example,
    http://dbpedia.org/resource/Category:Mac_OS_X_backup_software .

    You should open() WikipediaGraphIndex object before using and close() after,
    or otherwise use it with a 'with' clause, as below:

    Use:
        category = 'http://dbpedia.org/resource/Category:Computer_science'
        with WikipediaGraphIndex() as wiki:
            print(wiki.get_subcats(category))

    """

    def __init__(self,
                 subcat_index_file=util.resource('wikipedia/uri-to-subcats'),
                 supercat_index_file=util.resource('wikipedia/uri-to-supercats')):
        self._subcat_index_file = subcat_index_file
        self._supercat_index_file = supercat_index_file
        
    def _get_related_topics(self, topic, relation, cache, api_get):
        """Gets the topics related to the given topic, e.g. subcats or supercats."""
        if topic.encode('utf-8') in cache:
            related = cache[topic].decode('utf-8').split()
        else:
            related = api_get(topic)
            if related is None:
                logging.warning('Page not in Wikipedia, perhaps deleted: {}'
                                .format(uri_to_title(topic)))
                related = []
            cache[topic] = ' '.join(related)
        return related
        
    def get_subcats(self, topic):
        return self._get_related_topics(topic, 'subcat', self._subcat_index, wikiapi.subcats)

    def get_supercats(self, topic):
        return self._get_related_topics(topic, 'supercat', self._supercat_index, wikiapi.supercats)

    def open(self):
        self._subcat_index = semidbm.open(self._subcat_index_file, 'c')
        self._supercat_index = semidbm.open(self._supercat_index_file, 'c')
        return self

    def close(self):
        self._subcat_index.close()
        self._supercat_index.close()
        
    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
        return False
    
    
class CategoryRelationCache(object):
    """ The in-memory cache for the parent-child category relations.
    
    You should open() CategoryRelationCache before using, and close()
    after, or use it with a 'with' clause.
    
    """
    
    def __init__(self,
                 subcat_index_file=util.resource('wikipedia/uri-to-subcats'),
                 supercat_index_file=util.resource('wikipedia/uri-to-supercats')):
        self._wiki_graph = WikipediaGraphIndex(
            subcat_index_file=subcat_index_file,
            supercat_index_file=supercat_index_file)
        self._children = collections.OrderedDict()
        self._parents = collections.OrderedDict()

    def _compute_parents(self, node):
        supercat_uris = self._wiki_graph.\
            get_supercats(dbpedia.to_category_uri(node))
        return sorted([dbpedia.to_title(uri)
                       for uri in supercat_uris])
    
    def _compute_children(self, node):
        subcat_uris = self._wiki_graph.\
            get_subcats(dbpedia.to_category_uri(node))
        return sorted([dbpedia.to_title(uri)
                       for uri in subcat_uris])
    
    def parents(self, node):
        if node in self._parents:
            return self._parents[node]
        else:
            result = self._compute_parents(node)
            self._parents[node] = result
            return result
    
    def children(self, node):
        if node in self._children:
            return self._children[node]
        else:
            result = self._compute_children(node)
            self._children[node] = result
            return result
        
    def open(self):
        self._wiki_graph.open()
        return self
        
    def close(self):
        self._wiki_graph.close()
        
    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
        return False


class TopicFeatures(object):
    """A container of topic features.
    
    A topic feature is a function that maps a topic to a real number.
    TopicFeatures.compute calculated the values of the contained features
    on the given topics. For every topic, the computed values are placed
    in a dictionary, where the keys correspond to the feature names.
    
    The class is used in the machine learning–based topic selection algorithm.
    As most features depend on the state of the selection algorithm,
    the feature functions must take the algorithm as their first parameters:
        some_feature_fn(selection: CategorySelection, topic:str).
        
    Usage example:
    
        >>> def n_words_ftr(selection, node):
        >>>    return len(uri_to_title(node).split())
        >>>    
        >>> def is_capitalized_ftr(selection, node):
        >>>     return all([word[0].isupper() for word in uri_to_title(node).split()])
        >>> 
        >>> features = TopicFeatures({'n_words': n_words, 'is_capitalized': is_capitalized_ftr})
        >>> 
        >>> nodes = ['http://dbpedia.org/resource/Category:Mac_OS_X_backup_software',
        >>>          'http://dbpedia.org/resource/Category:Mac_OS']
        >>> 
        >>> features.compute(selection, nodes)
        [OrderedDict([('is_capitalized', False), ('n_words', 5)]),
         OrderedDict([('is_capitalized', True), ('n_words', 2)])]
        
    """

    def __init__(self, feature_dict):
        self._features = collections.OrderedDict(sorted(feature_dict.items(),
                                                 key=lambda pair: pair[0]))
        
    def feature_names(self):
        return list(self._features.keys())

    def get_feature(self, name):
        return self._features[name]

    def remove_feature(self, name):
        return self._features.pop(name, None)
    
    def add_feature(self, name, feature_fn):
        self._features[name] = feature_fn

    def _compute_one(self, selection, topic):
        return collections.OrderedDict((name, value(selection, topic))
                                       for name, value in self._features.items())

    def compute(self, selection, topics):
        return [self._compute_one(selection, topic) for topic in topics]
    
    def copy(self):
        result = TopicFeatures({})
        result._features = self._features.copy()
        return result


class TopicClassifier(object):
    """A classifier for predicting if a topic is relevant. 
    
    Provides the API in terms of topics (URIs) by wrapping the features
    and the actual sklearn classifier.
    
    """

    def __init__(self, features: TopicFeatures, classifier: LinearModel):
        self._features = features
        self._classifier = classifier

    def train(self, selection, data, **params):
        feature_values = self._features.compute(selection, data.keys())
        xs = pd.DataFrame(feature_values)
        ys = list(data.values())
        allowed_param_names = self._classifier._get_param_names()
        allowed_params = dict((name, params[name])
                               for name in params.keys()
                               if name in allowed_param_names)
        self._classifier.fit(xs, ys, **allowed_params)

    def predict(self, selection, topics):
        feature_values = self._features.compute(selection, topics)
        xs = pd.DataFrame(feature_values)
        return self._classifier.predict(xs)


def _always(node):
    return True
    
class CategorySelection(object):
    """Selects the subgraph of the Wikipedia category hierarchy.

    """

    def __init__(self, root, classifier: TopicClassifier=None, max_depth=None,
                 relation_cache=None):
        if not relation_cache:
            raise ValueError("Cannot create CategorySelection without relation_cache.")
        if (not classifier) and max_depth is None:
            raise ValueError("Either the training data or the max_depth "
                             "should be specified.")
        self._max_depth = max_depth
        self._root = root
        self._relations = relation_cache
        self._visited = set()
        self._schedule = collections.deque([root])
        self._relevant = {root}
        self._parents_in_graph = collections.defaultdict(set)
        self._parents = {}
        self._children = {}
        self._children_in_graph = collections.defaultdict(set)
        self._depth = {root: 0}
        self._classifier = classifier
        self._loops = []

    def _stop_criteria(self):
        return len(self._schedule) == 0

    def _node_in_hierarchy(self, node):
        return node in self._depth

    # TODO: decouple the side effects
    def _should_schedule_child(self, node, child):
        if not self._classifier and child in self._visited:
            return False  # not re-scheduling the visited nodes
                          # when the selection is unconditional (without classifier)
        if self.get_depth(node) >= self._max_depth:
            return False
        self._update_caches(node, child)
        if self._is_ancestor(node, child):
            # TODO: report the whole loop rather than only the endpoints.
            self._loops.append((to_title(node), to_title(child)))
            logging.warning("Loop '{}'<->'{}'".format(
                uri_to_title(node), uri_to_title(child)))
            return False
        child_was_irrelevant = not self.is_relevant(child)
        self._compute_and_update_relevance_status(child)
        child_is_relevant = self.is_relevant(child)
        # Re-scheduling only new nodes or nodes that were previously irrelevant;
        # this is an approximation.
        return child_was_irrelevant and child_is_relevant

    def _schedule_node(self, node):
        self._schedule.append(node)

    def _next_node(self):
        return self._schedule.popleft()

    def _compute_children(self, node):
        """Returns and caches the children (subcategories) for the given node.
    
        The children are fetched from Wikipedia or from the local cache.

        """
        children = self._children.get(node)
        if children is None:
#             children = set(self._wiki_graph.get_subcats(node) or [])
            children = set(to_category_uri(child)
                           for child in self._relations.children(node))
            self._children[node] = children
        return children

    def get_children(self, node):
        """Returns the children of the node in the graph explored so far.

        This returns the previously computed set of children.
        This set is updated in self._update_caches.

        """
        return self._children_in_graph[node]

    def _compute_parents(self, node):
        """Returns and caches the parents (super-categories) for the given node.

        The parents are fetched from Wikipedia or from the local cache.
        Unlike self.get_parents, returns also the parents that are not in the
        'subtree' of self._root.

        """
        parents = self._parents.get(node)
        if not parents:
#             parents = self._wiki_graph.get_supercats(node)
            self._parents[node] = set(to_category_uri(parent)
                                      for parent in self._relations.parents(node))

    def get_parents(self, node):
        """Returns the parents of the node in the graph explored so far.

        This returns the previously computed set of parents. The returned set
        only includes parents that are in the 'subtree' of the self._root.
        This set of parents is updated in self._update_caches.

        """
        return self._parents_in_graph[node]

    def _compute_depth(self, node):
        """Returns the depth of the node = the min depth of its parents + 1.

        The depth is with respect to the self._root, so only the parents
        that are in the current graph are considered. The computation assumes
        that the depths of the parents have already been computed and stored
        (which is true because of the BFS nature of the algorithm).

        """
        if node == self._root:
            return 0
        else:
            return 1 + min(map(self.get_depth, self.get_parents(node)))

    def get_depth(self, node):
        """Returns the previously computed depth of the node."""
        if node not in self._depth:
            raise ValueError("Cannot compute the depth of '{}': "
                             "the node has not been processed."
                            .format(uri_to_title(node)))
        return self._depth[node]

    def _classify_node(self, node):
        """Classifies the node, returns True if relevant, False otherwise.

        """
        prefix = "re" if node in self._visited else ""
#         logging.debug(prefix + "classifying the node '{}'".format(uri_to_title(node)))
        is_relevant = self._classifier.predict(self, [node])[0]
        if is_relevant:
            logging.debug("Node {}:'{}' is relevant".format(
                self.get_depth(node), uri_to_title(node)))
        else:
            logging.debug("Node {}:'{}' is irrelevant".format(
                self.get_depth(node), uri_to_title(node)))
        return is_relevant

    def _compute_and_update_relevance_status(self, node):
        if not self.is_relevant(node):
        # re-classify only new or previously irrelevant nodes;
        # this is an approximation
            is_relevant = True
            if self._classifier:
                is_relevant = self._classify_node(node)
            if is_relevant:
                self._relevant.add(node)
            elif node in self._relevant:
                self._relevant.remove(node)
        self._visited.add(node)

    def _update_caches(self, parent, child):
        """Updates the stored information about the node.

        Updates the child-to-parents map and the node's depth."""
        self._parents_in_graph[child].add(parent)
        self._compute_parents(child)

        # After self._compute_parents(child), self._parents[child] should
        # contain the parent already, but this is not always the case.
        # It happens when parent is a hidden category, and thus is not included
        # by self._compute_parents. Sometimes it happens when the parent is
        # not a hidden category, I don't know why. Example: Wiki->WikiLeaks.
        # The following line is a patch that fixes this problem in some way:
        self._parents[child].add(parent)

        self._children_in_graph[parent].add(child)
        self._depth[child] = self._compute_depth(child)

    def _is_ancestor(self, node, candidate_ancestor):

        def get_all_parents(anode):
            return self._parents[anode] if anode in self._parents else []

        # Attention: uses a dangerous feature:
        # visited_nodes is persisted between all the calls to is_ancestor
        # that happen withing a single invocation of _is_ancestor
        def is_ancestor(new_nodes, candidate_ancestor, visited_nodes={node}):
            if candidate_ancestor in new_nodes:
                return True
            elif len(new_nodes) == 0:
                return False
            else:
                new_parents = {parent for new_node in new_nodes
                               for parent in get_all_parents(new_node)
                               if parent not in visited_nodes}
                visited_nodes.update(new_parents)
                return is_ancestor(new_parents, candidate_ancestor)

        return is_ancestor({node}, candidate_ancestor)

    # # The old implementation of the _is_ancestor procedure.
    # # The new implementation is more efficient, but this old one was used
    # # to generate the training data.
    # #
    # def _is_ancestor(self, node, candidate_ancestor):
    #     if node == candidate_ancestor:
    #         return True
    #     elif any(self._is_ancestor(parent, candidate_ancestor)
    #              for parent in self.get_parents(node)):
    #         return True
    #     else:
    #         return False

    def is_relevant(self, node):
        return to_category_uri(node) in self._relevant
    
    def predict(self, X):
        return [self.is_relevant(x) for x in X]

    def _step(self):
        """Performs a single step of the currently BFS-like selection algorithm.

        Examines the next node in the schedule, computes the scores for its
        children, and schedules them. Returns the node.

        """
        node = self._next_node()

        logging.info("Processing the node {}:'{}'"
                      .format(self.get_depth(node), uri_to_title(node)))

        for child in sorted(self._compute_children(node)):
            logging.info("Processing the child '{}'->'{}'"
                          .format(uri_to_title(node), uri_to_title(child)))
            if self._should_schedule_child(node, child):
                self._schedule_node(child)

        logging.debug("Finished processing the children of '{}'"
                      .format(node))
        return node

    def run(self, **kwargs):
        """Runs the top-down selection procedure."""
        with self._relations:
            while not self._stop_criteria():
                self._step()

    def _bfs(self, should_schedule=_always, should_report=_always):
        """Performs the breadth-first traversal of the built category graph.

        Returns the generator for the set of unique nodes in the traversal.

        """
        visited = set([self._root])
        queue = collections.deque([(self._root, None)])
        while queue:
            node, parent = queue.popleft()
            if should_schedule(node):
                if should_report(node):
                    yield node, parent
                for child in sorted(self.get_children(node)):
                    if child not in visited:
                        queue.append((child, node))
                        visited.add(child)
                        
    def relevant_nodes(self):
        return [child for child, parent in self._bfs(should_report=self.is_relevant)]

    def __iter__(self):
        return (child for child, parent in self._bfs())


class CategorySelectionClassifier(BaseEstimator):
    """A scikit-learn classifier for predicting if a topic is relevant.
    
     The classifier wraps the category selction algorithm:
     The topic is predicted as relevant if it has been marked as such
     by the selection algorithm. Otherwise (if a topic was marked as irrelevant,
     or has not been discovered) the classifier predicts that it is irrelevant.
     
     The classifier can be fed to scikit-learn cross-validation procedures,
     for parameter tuning and evaluation.
     
     """

    def create_classifier(self, **params):
        classifier = self.classifier_fn
        return classifier(**params)

    def __init__(self,
                 full_selection: CategorySelection,
                 features: TopicFeatures,
                 classifier_fn,
                 max_depth,
                 instance_weight=lambda x: 1,
                 **classifier_params):
        self.full_selection = full_selection
        self.features = features
        self.max_depth = max_depth
        self.instance_weight = instance_weight
        self.classifier_fn = classifier_fn
        self.classifier_params = classifier_params
        for name, value in classifier_params.items():
            setattr(self, name, value)

    def copy(self):
        result = CategorySelectionClassifier(
            self.full_selection,
            self.features,
            self.classifier_fn,
            self.max_depth,
            self.instance_weight,
            **self.classifier_params)
        if hasattr(self, 'selection'):
            result.selection = self.selection
        return result
            
    def _get_param_names(self):
        param_names = ['full_selection', 'features', 'max_depth',
                       'instance_weight', 'classifier_params', 'classifier_fn']
        param_names.extend(self.classifier_params.keys())
        return param_names

    def fit(self, X, y):
        classifier_params = {name: getattr(self, name)
                             for name in self.classifier_params}
        classifier = TopicClassifier(
            self.features,
            classifier=self.create_classifier(**classifier_params)
        )
        training_data = dict(zip(X, y))
        sample_weight = [self.instance_weight(x) for x in X]
        classifier.train(self.full_selection, training_data,
                         sample_weight=sample_weight)
        self.selection = CategorySelection(self.full_selection._root,
                                           classifier, self.max_depth,
                                           self.full_selection._relations)
        self.selection.run()

    def predict(self, X):
        return self.selection.predict(X)


class TrainingDataSelection(object):
    
    """A bunch of methods for selecting samples (e.g. for labeling) topics.
    
    The methods were written ad hoc and are not well factored.
    
    """
    
    def __init__(self, selection: CategorySelection):
        self.selection = selection

    def sample_categories_uniformly(self, n_nodes, random_seed=29121985):
        random.seed(random_seed)
        return random.sample(list(self.selection), n_nodes)

    def _sample_path_down(self, node):
        path = []
        visited = set()
        children = [node]
        while children:
            child = random.choice(sorted(children))
            path.append(child)
            visited.add(child)
            children = [child for child in self.selection.get_children(child)
                              if child not in visited]
        return path

    def _shortest_path_from_root(self, anode):
        queue = collections.deque([(anode, 0)])

        def backtrack(target, visited_from):
            path = []
            node = target
            while node is not None:
                path.append(node)
                node = visited_from[node]
            # path.reverse()
            return path

        visited_from = {anode: None}

        while queue:
            node, level = queue.popleft()
            if node == self.selection._root:
                return backtrack(node, visited_from)
            else:
                for parent in self.selection.get_parents(node):
                    if parent not in visited_from:
                        queue.append((parent, level + 1))
                        visited_from[parent] = node
        else:
            return None

    def _sample_path_up(self, node):
        path = collections.deque([])
        parents = [node]
        visited = set()
        root = self.selection._root
        while parents:
            parent = random.choice(sorted(parents))          
            path.appendleft(parent)
            visited.add(parent)
            parents = [parent for parent in self.selection.get_parents(parent)
                              if parent not in visited and parent != root]
        return list(path)

    def _sample_path_through(self, node):
        path_up = self._sample_path_up(node)[1:]  # with node, without the root
        path_down = self._sample_path_down(node)[1:]  # without node
        return path_up + path_down

    def sample_paths_down_from_root(self, n_nodes, random_seed=29121985):
        random.seed(random_seed)
        result = collections.OrderedDict()
        while len(result) < n_nodes:
            start_node = self.selection._root
            path = self._sample_path_down(start_node)[1:]  # without root
            print(' -> '.join(uri_to_title(node) for node in path))
            for node in path:
                result[node] = None
        return result.keys()

    def sample_paths_down_from_anywhere(self, n_nodes, random_seed=29121985):
        random.seed(random_seed)
        result = collections.OrderedDict()
        all_nodes = list(self.selection)
        while len(result) < n_nodes:
            start_node = random.choice(all_nodes)
            path = self._sample_path_down(start_node)
            print(' -> '.join(uri_to_title(node) for node in path))
            for node in path:
                result[node] = None
        return result.keys()

    def sample_paths_up_from_leaves(self, n_nodes, random_seed=29121985):
        random.seed(random_seed)
        result = collections.OrderedDict()

        def is_leaf(node):
            return not self.selection.get_children(node)

        leaf_nodes = [node for node, parent
                      in self.selection._bfs(should_report=is_leaf)]
        while len(result) < n_nodes:
            start_node = random.choice(leaf_nodes)
            path = self._sample_path_up(start_node)[1:]  # without the root
            print(' -> '.join(uri_to_title(node) for node in path))
            for node in path:
                result[node] = None
        return result.keys()

    def sample_paths_through_from_anywhere(self, n_nodes, fixed_length=True, random_seed=29121985):
        random.seed(random_seed)
        result = collections.OrderedDict()
        all_nodes = list(self.selection)
        while len(result) < n_nodes:
            start_node = random.choice(all_nodes)
            path = self._sample_path_through(start_node)
            if fixed_length:
                path = path[:self.selection._max_depth]
            print("* " + uri_to_title(start_node) + " *")
            print(' -> '.join(uri_to_title(node) for node in path))
            for node in path:
                result[node] = None
        return result.keys()

    def sample_relations_with_children(self, n_nodes, max_children=None, with_paths=True,
                                       filter_parent=None, filter_child=None, random_seed=29121985):
        random.seed(random_seed)
        result = collections.OrderedDict()
        selection = self.selection
        all_nodes = [child for child, parent in selection._bfs(should_schedule=selection.is_relevant)]
        nodes_with_children = [node for node in all_nodes if selection.get_children(node)]
        random.shuffle(nodes_with_children)
        index = 0
        visited = set()
        
        get_path = (lambda node: self._shortest_path_from_root(node)
                    if with_paths
                    else lambda node: [node])
        
        filter_parents = (lambda parents: filter(filter_parent, parents)
                          if filter_parent
                          else parents)
        
        filter_children = (lambda children: filter(filter_child, children)
                           if filter_child
                           else children)
        
        def filter_visited_and_visit(nodes):
            result = []
            for node in nodes:
                if node not in visited:
                    result.append(node)
                    visited.add(node)
            return result
        
        def get_n_children(node, path):
            children = sorted(self.selection.get_children(node))
            children_on_path = [child for child in children if child in path]
            children_off_path = [child for child in children if child not in path]
            random.shuffle(children_off_path)
            children = children_on_path + children_off_path
            n_children = len(children) if max_children is None else max_children
            return list(itertools.islice(filter_children(children), 0, n_children))
        
        def report_parent_child(parent, child):
            parent_title = uri_to_title(node).replace(' ', '_')
            child_title = uri_to_title(child).replace(' ', '_')
            print("Sampled '{}' -> '{}'".format(parent_title, child_title))
            
        for sampled_node in nodes_with_children:
            if len(result) >= n_nodes:
                break
            if sampled_node in visited:
                continue
            path = filter_parents(filter_visited_and_visit(get_path(sampled_node)))
            for node in path:
                for child in get_n_children(node, path):
                    report_parent_child(node, child)
                    result[(node, child)] = None
            
        return result.keys()


class TrainingDataProcessing(object):
    """A class for reading and writing (labeled or not) ground truth data.
    
    The class is not well factored.
    
    """

    def save_topic_labels(self, topics, labels, filename, data_sampler: TrainingDataSelection = None):
        if (data_sampler):
            paths = (data_sampler._shortest_path_from_root(topic)[1:]
                     for topic in topics)
        else:
            paths = ([topic] for topic in topics)
        path_strings = (' -> '.join(uri_to_title(topic).replace(' ', '_')
                                    for topic in topics)
                        + ' ' + self._label_string(label)
                        for topics, label in zip(paths, labels))
        with open(filename, 'w', encoding='utf-8') as file:
            file.write('\n'.join(path_strings))
            
    def save_pairs(self, pairs, filename):
        pair_strings = (' -> '.join(uri_to_title(topic).replace(' ', '_')
                                    for topic in pair)
                        for pair in pairs)
        with open(filename, 'w', encoding='utf-8') as file:
            file.write('\n'.join(pair_strings))

    def _parse_label(self, label):
        if label == '+':
            return True
        elif label == '-':
            return False
        else:
            raise ValueError('Invalid label {}; must be one of {{+, -}}'
                             .format(label))
            
    def _label_string(self, label):
        if label is None:
            return ''
        else:
            return '+' if label else '-'

    def _parse_line(self, line):
        title_and_label = line.strip().split(' -> ')[-1]
        title_, label = title_and_label.strip().split()
        topic = title_to_uri(title_.strip().replace('_', ' '), category=True)
        label = self._parse_label(label.strip())
        return topic, label

    def read_topic_labels(self, filename):
        result = collections.OrderedDict()
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file.readlines():
                try:
                    topic, label = self._parse_line(line)
                    result[topic] = label
                except ValueError:
                    logging.info("Invalid label, ignoring the line: {}".format(line))
        return result


# ## Features of the category nodes

def normalized_depth_feature(selection: CategorySelection, node):
    return selection.get_depth(node) / selection._max_depth

def frac_parents_in_graph_feature(selection: CategorySelection, node):
    n_parents_total = len(selection._parents[node])
    n_parents_in_tree = len(selection.get_parents(node))
    return n_parents_in_tree / n_parents_total

def coparents_total(selection: CategorySelection, node):
    children = selection._children.get(node) or {}
    return [parent for child in children
                   for parent in selection._parents[child]
                   if parent != node]

def coparents_in_graph(selection: CategorySelection, node):
    children = selection._children.get(node) or {}
    return [parent for child in children
                   for parent in selection.get_parents(child)
                   if parent != node]

def frac_coparents_in_graph_feature(selection: CategorySelection, node):
    coparents_total_with_duplicates = coparents_total(selection, node)
    coparents_in_tree_with_duplicates = coparents_in_graph(selection, node)
    n_coparents_total = len(coparents_total_with_duplicates)
    n_coparents_in_tree = len(coparents_in_tree_with_duplicates)
    if n_coparents_total == 0:
        return 0
    else:
        return n_coparents_in_tree / n_coparents_total
    
def children_with_parents_in_graph(selection: CategorySelection, node):
    children_total = selection._children.get(node, [])
    def parents_in_graph_except_node(child):
        parents = set(selection.get_parents(child))
        parents.discard(node)
        return parents
    return [child for child in children_total
                  if parents_in_graph_except_node(child)]
    
def frac_children_with_parents_in_graph_feature(selection: CategorySelection, node):
    children_total = selection._children.get(node, [])
    children_with_parents = children_with_parents_in_graph(selection, node)
    if not children_total:
        return 0
    else:
        return len(children_with_parents) / len(children_total)

def parent_stat_feature(feature, statistics):
    def feature_fn(selection, node):
        parents_in_graph = selection.get_parents(node)
        parent_feature_values = [feature(selection, parent)
                                 for parent in parents_in_graph]
        return statistics(parent_feature_values)
    return feature_fn


def n_parents_in_graph_feature(selection: CategorySelection, node):
    return len(selection.get_parents(node))


def n_parents_total_feature(selection: CategorySelection, node):
    return len(selection._parents[node])


# ### Title-based features

stemmer = nltk.stem.snowball.EnglishStemmer()

def stem_word(word: str):
    return stemmer.stem(word)

def word_list(string: str):
    return [token
            for token in nltk.tokenize.word_tokenize(string)]

def jaccard_sim(set1: set, set2: set):
    return len(set1.intersection(set2)) / len(set1.union(set2))

def without_stopwords(words):
    return [word for word in words if word not in util.stopwords]

def stem_jaccard(str1, str2):
    stems1 = set(stem_word(word) for word in without_stopwords(word_list(str1)))
    stems2 = set(stem_word(word) for word in without_stopwords(word_list(str2)))
    return jaccard_sim(stems1, stems2)

assert stem_jaccard("Computer science related lists", "Lists and Computing") == 0.5

def parent_max_jaccard_similarity(selection: CategorySelection, node):
    return max(stem_jaccard(to_title(node), to_title(parent))
               for parent in selection.get_parents(node))

def get_grandparents(selection, node):
    result = set()
    for parent in selection.get_parents(node):
        result.update(selection.get_parents(parent))
    return result

def grandparent_max_jaccard_similarity(selection: CategorySelection, node):
    grandparents = get_grandparents(selection, node)
    if grandparents:
        return max(stem_jaccard(to_title(node), to_title(gp))
                   for gp in grandparents)
    else:
        return 1


# ## Training the classifier

def train_topic_classifier(topics, classes,
                           full_selection, max_depth, 
                           features, classifier_fn,
                           instance_weight_fn,
                           classifier_params) -> CategorySelectionClassifier:
    
    clf = CategorySelectionClassifier(full_selection=full_selection,
                                      features=features,
                                      classifier_fn=classifier_fn,
                                      max_depth=full_selection._max_depth,
                                      instance_weight=instance_weight_fn,
                                      **classifier_params)
    clf.fit(topics, classes)
    return clf
    

# ### Train the classifier with internal cross-validation

def train_topic_classifier_cv(topics, classes,
                              full_selection, max_depth, 
                              features, classifier_fn,
                              instance_weight_fn,
                              cross_validation,
                              evaluation_measure,
                              param_grid,
                              classifier_params) -> GridSearchCV:
    tuned_clf = CategorySelectionClassifier(full_selection=full_selection,
                                            features=features,
                                            classifier_fn=classifier_fn,
                                            max_depth=full_selection._max_depth,
                                            instance_weight=instance_weight_fn,
                                            **classifier_params)
    clf = GridSearchCV(estimator=tuned_clf, param_grid=param_grid, cv=cross_validation(classes),
                       scoring=evaluation_measure)
    clf.fit(topics, classes)
    return clf


# ### Default training parameters

def merge_ground_truth(data1, data2):
    """Merges the two ground truth datasets, e.g. from different annotators.
    
       Returns the intersection of the two dataset, which includes the labels
       on which the two datasets agree.
       
    """
    return collections.OrderedDict((node, label)
                                   for node, label in data1.items()
                                   if node in data2 and data2[node] == label)


default_ground_truth_data_file = 'labeled-topics-1000-dm-ls-merged.txt'


# Ad hoc procedure, used once to merge the two annotators' labelings.
def save_merged_ground_truth_data(full_selection, filename=util.resource(default_ground_truth_data_file)):
    data_processor = TrainingDataProcessing()
    data_ls = data_processor.read_topic_labels(
        util.resource('labeled-topics-1000-ls.txt')
    )
    data_dm = data_processor.read_topic_labels(
        util.resource('labeled-topics-1000-dm.txt')
    )
    data = merge_ground_truth(data_ls, data_dm)
    data_processor.save_topic_labels(data.keys(), data.values(), filename)

    
def read_ground_truth_data(filename=util.resource(default_ground_truth_data_file)):
    return TrainingDataProcessing().read_topic_labels(filename)


default_root="http://dbpedia.org/resource/Category:Computing"
default_max_depth=7


def depth_based_selection(root=default_root, max_depth=default_max_depth):
    relation_cache = CategoryRelationCache(
        subcat_index_file=util.resource('wikipedia/uri-to-subcats'),
        supercat_index_file=util.resource('wikipedia/uri-to-supercats'))
    full_selection = CategorySelection(root, max_depth=max_depth, relation_cache=relation_cache)
    full_selection.run()
    return full_selection


def default_classifier(**params):
    return SVC(**params)


default_features = TopicFeatures({
 'unity' : lambda selection, node: 1,
 'normalized_depth': normalized_depth_feature,
 'frac_parents_in_graph': frac_parents_in_graph_feature,
#  'frac_coparents_in_graph': frac_coparents_in_graph_feature,
#  'frac_children_with_parents_in_graph': frac_children_with_parents_in_graph_feature,
 'parent_max_jaccard_sim': parent_max_jaccard_similarity,
 'grandparent_max_jaccard_sim': grandparent_max_jaccard_similarity,
#  'parent_avg_jaccard_sim': parent_in_graph_avg_jaccard_similarity,
 'n_parents_in_graph': n_parents_in_graph_feature,
 'n_parents_total': n_parents_total_feature,
 'min_normalized_parent_depth': parent_stat_feature(
     normalized_depth_feature,
     np.min),
 'max_normalized_parent_depth': parent_stat_feature(
     normalized_depth_feature,
     np.max),
 'avg_normalized_parent_depth': parent_stat_feature(
     normalized_depth_feature,
     np.mean),
 'median_normalized_parent_depth': parent_stat_feature(
     normalized_depth_feature,
     np.median),
})


def default_data(selection=None):
    return read_ground_truth_data()


default_classifier_params = {'C': 3, 'kernel': 'linear', 'class_weight': {0: 1, 1: 1}}


default_param_grid = [{'C': [3], 'kernel': ['linear'], 'class_weight': [{0: 1, 1: 1}]}]


def precompute_full_selection(precomputed_data={}):
    if not 'full_selection' in precomputed_data:
        precomputed_data['full_selection'] = depth_based_selection()
    return precomputed_data['full_selection']


def default_instance_weight_fn(node):
    full_selection = precompute_full_selection()
    branching_factor = 1.5
    depth = full_selection.get_depth(node)
    return 1 / pow(branching_factor, depth)


default_training_params_ = {
    'max_depth' : default_max_depth,
    'features' : default_features,
    'classifier_fn' : default_classifier,
    'instance_weight_fn' : default_instance_weight_fn,
    'classifier_params' : default_classifier_params}


def default_cross_validation(outputs):
    return cross_validation.StratifiedKFold(outputs, n_folds=5)


default_training_params_cv_ = {
    'cross_validation' : default_cross_validation,
    'evaluation_measure' : util.weighted_f1,
    'param_grid' : default_param_grid
}


def new_training_params(**params):
    result = copy.deepcopy(default_training_params_)
    result.update(params)
    return result


def new_training_params_cv(**params):
    result = copy.deepcopy(default_training_params_)
    result.update(default_training_params_cv_)
    result.update(params)
    return result


# ## Evaluating the classifier

def evaluate_classifier(clf, topics, classes, eval_measure):
    return eval_measure(clf, list(topics), list(classes))

def train_evaluate_topic_classifier_cv(tuned_clf, topics, classes,
                                       inner_cross_validation,
                                       outer_cross_validation,
                                       model_selection_measure,
                                       evaluation_measures,
                                       param_grid=None,
                                       learning=True,
                                       training_size=None) -> GridSearchCV:
    results = []
    current_fold = 1
    for train_idx, test_idx in outer_cross_validation(classes):
        print("Fold {}".format(current_fold))
        sampled_train_idx = train_idx
        if training_size is not None:
            actual_training_size = min(len(train_idx), training_size)
            if training_size and actual_training_size < training_size:
                print("Warning: Training set size {} is smaller than requested: {}."
                      .format(actual_training_size, training_size))
            sampled_train_idx = np.random.choice(train_idx, training_size, replace=False)
        results.append([])
        inner_cross_validation_object = inner_cross_validation(classes[sampled_train_idx])\
                                        if inner_cross_validation else None
        clf = None
        if param_grid:
            clf = GridSearchCV(estimator=tuned_clf, param_grid=param_grid,
                   cv=inner_cross_validation_object,
                   scoring=model_selection_measure)
        else:
            clf = tuned_clf
        if learning:
            clf.fit(topics[sampled_train_idx], classes[sampled_train_idx])
        if hasattr(clf, 'best_params_'):
            print(clf.best_params_)
        for measure in evaluation_measures:
            score = evaluate_classifier(clf, topics[test_idx], classes[test_idx], measure)
            results[-1].append(score)
        current_fold += 1
    return np.array(results).T


### Baseline, dumb classifiers: majority rule, random, etc.

class MajorityClassClassifier(BaseEstimator):
    def _get_param_names(self):
        return []

    def fit(self, X, y):
        frequencies = collections.Counter(y).items()
        self._most_frequent_element = max(frequencies, key=lambda item: item[1])[0]
        
    def predict(self, X):
        return np.full(len(X), self._most_frequent_element)

    
class StratifiedRandomClassifier(BaseEstimator):

    def __init__(self, random_state):
        np.random.seed(random_state)
    
    def _get_param_names(self):
        return []

    def fit(self, X, y):
        self._y = y
        
    def predict(self, X):
        return np.random.choice(self._y, size=len(X), replace=True)
