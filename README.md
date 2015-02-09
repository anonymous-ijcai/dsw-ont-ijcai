# dsw-ont-ijcai

This project deals with bootstrapping domain-specific ontologies from the categories of Wikipedia.
The code, experiments, and data in this project are associated with a paper submitted to [IJCAI 2015](http://ijcai-15.org/) for double-blind review.

---

## About the files

### Code

The following python files contain the main code:
- [topics.py](./dswont/topics.py) -- extracting the categories relevant for a given domain,
- [topic_type.py](./dswont/topic_type.py) -- classifying categories into classes and individuals,
- [relation_type.py](./dswont/relation_type.py) -- classifying sub-category relations between categories.


### Experiments

The following IPython notebooks contain the main experiments:
- [topic_selection_eval.ipynb](./dswont/topic_selection_eval.ipynb) -- extracting the relevant categories,
- [topic_type_eval.ipynb](./dswont/topic_type_eval.ipynb) -- classifying categories,
- [relation_type_eval.ipynb](./dswont/relation_type_eval.ipynb) -- classifying relations.


### Data

#### The main "ground truth"/training/evaluation labeled data files:
- [labeled-topics-1000-dm-ls-merged.txt](./dswont/resources/labeled-topics-1000-dm-ls-merged.txt) -- relevant vs. irrelevant to `Computing`,
- [labeled-topics-music-1000-dm.txt](./dswont/resources/labeled-topics-music-1000-dm.txt) -- relevant vs. irrelevant to `Music`,
In these files `+` stands for `relevant`, while `-` stands for `irrelevant`.

- [labeled-topic-types-1000-dm.txt](./dswont/resources/labeled-topic-types-1000-dm.txt) -- `class` vs. `individual` for `Computing` .
- [labeled-topic-types-music-1000-dm.txt](./dswont/resources/labeled-topic-types-music-1000-dm.txt) -- `class` vs. `individual` for `Computing` .
In these files `c` stands for `class`, while `t` (topic) and `i` (individual) codify `individual`.

- [labeled-relations-150-subcl-in-wtx-with-3-children-dm.txt](./dswont/resources/labeled-relations-150-subcl-in-wtx-with-3-children-dm.txt) -- `subclas_of` vs. `related_to` for `Computing`.
- [labeled-relations-150-instance-in-wtx-with-3-children-dm.txt](./dswont/resources/labeled-relations-150-instance-in-wtx-with-3-children-dm.txt) -- `instance_of` vs. `related_to` for `Computing`.
- [labeled-relations-150-part-in-acm-with-all-children-ls.txt](./dswont/resources/labeled-relations-150-part-in-acm-with-all-children-ls.txt) -- `part_of` vs. `related_to` for `Computing`.
In these files `s` stands for `subclass_of`, `i` -- for `lsinstance_of`, `p` -- for `part_of`, `r` -- for `related_to`.


#### Nodes and relations extracted from `WikiTaxonomy`

The files related to `WikiTaxonomy` are in [dswont/resources/wikitaxonomy](./dswont/resources/wikitaxonomy).

- [wikipediaOntology.owl](./dswont/resources/wikitaxonomy/wikipediaOntology.owl) -- `WikiTaxonomy` itself,
- [node-types.txt](./dswont/resources/wikitaxonomy/node-types.txt) -- extracted node types,
- [relation-types.txt](./dswont/resources/wikitaxonomy/relation-types.txt) -- extracted relation types.


#### Nodes and relations extracted from ACM CCS

- [ACMCCS.xml](./dswont/resources/ACMCCS.xml) -- `ACM CCS` itself,
- [acm-concepts.txt](./dswont/resources/acm-concepts.txt) -- mapping from `ACM` ids to concept names,
- [acm-relations.txt](./dswont/resources/acm-relations.txt) -- pairs of parent-child concepts in `ACM`,
- [acm-wiki-mapping.txt](./dswont/resources/acm-wiki-mapping.txt) -- partial mapping between `ACMCCS` and `Wikipedia` categories.


#### Cache files

The project uses the [Web API](http://www.mediawiki.org/wiki/API:Main_page) of `Wikipedia` to extract categories and relations between them.
The relations are cached on disk to avoid unnecessary queries.
Directory [dswont/resources/wikipedia/](./dswont/resources/wikipedia/) contains the cached relations for the categories `Computing` and `Music` up to certain depth.
See `dswont.topics.CategoryRelationCache` and `dswont.topics.CategorySelection` for more details on using the cache.


#### The generated ontology skeleton

The following files contain the nodes and relations of the generated ontology skeleton for `Computing`:
- [2014-05-14-ontology-classes.csv](./dswont/resources/2014-05-14-ontology-classes.csv),
- [2014-05-14-ontology-individuals.csv](./dswont/resources/2014-05-14-ontology-individuals.csv),
- [2014-05-14-ontology-relations.csv](./dswont/resources/2014-05-14-ontology-relations.csv).


## Requirements

You will need Python to run the code and [IPython Notebooks](http://ipython.org/notebook.html) to run the experiments.
The code and experiments rely, among others, on [Pandas](http://pandas.pydata.org/) and [scikit-learn](http://scikit-learn.org/stable/).

We developed the project using Python 3.4.2 and IPython 2.3.0.

## License and disclaimer

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

