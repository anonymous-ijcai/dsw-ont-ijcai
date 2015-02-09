import urllib.request
import urllib.parse
import textwrap
import logging
import re


DBPEDIA_ENDPOINT = "http://dbpedia.org/sparql"
DBPEDIA_LIMIT = 50000
DBPEDIA_TIMEOUT = 30000
DBPEDIA_CATEGORY_PREFIX = "http://dbpedia.org/resource/Category:"
DBPEDIA_ARTICLE_PREFIX = "http://dbpedia.org/resource/"
TSV_FORMAT = "text/tab-separated-values"
JSON_FORMAT = "application/sparql-results+json"


def is_category_uri(uri):
    return uri.startswith(DBPEDIA_CATEGORY_PREFIX)


def is_article_uri(uri):
    return uri.startswith(DBPEDIA_ARTICLE_PREFIX)


def strip_dbpedia_prefix(uri):
    if uri.startswith(DBPEDIA_CATEGORY_PREFIX):
        return uri[len(DBPEDIA_CATEGORY_PREFIX):]
    elif uri.startswith(DBPEDIA_ARTICLE_PREFIX):
        return uri[len(DBPEDIA_ARTICLE_PREFIX):]
    else:
        message = "String doesn't start with a DBpedia uri prefix: " + uri
        raise Exception(message)


def uri_to_title(uri):
    """Extracts the title of the DBpedia article (or category) from its uri."""
    return strip_dbpedia_prefix(uri).replace('_', ' ')


def title_to_uri(title, category=False):
    """Constructs the uri of the DBpedia article or category from its title."""
    base = DBPEDIA_CATEGORY_PREFIX if category else DBPEDIA_ARTICLE_PREFIX
    return base + title.replace(' ', '_')


def to_category_uri(title_or_uri):
    if is_category_uri(title_or_uri):
        return title_or_uri
    else:
        return title_to_uri(title_or_uri, category=True)
    
def to_title(node):
    if is_category_uri(node):
        return uri_to_title(node)
    else:
        return node

def category_pairs_query(root, depth, limit=DBPEDIA_LIMIT, offset=0):
    """Creates a SPARQL query for retrieving parent-child category pairs from DBpedia."""
    return textwrap.dedent("""\
            prefix cat: <http://dbpedia.org/resource/Category:>
            prefix dbp: <http://dbpedia.org/ontology/>
            select distinct ?parent ?child
            where
            {{
            ?child skos:broader ?parent .
            ?parent skos:broader{{0,{}}} cat:{} .
            }}
            limit {}
            offset {}""").format(depth, root, limit, offset)


def category_articles_query(root, depth, limit=DBPEDIA_LIMIT, offset=0):
    """Creates a SPARQL query for retrieving category-article pairs from DBpedia."""
    return textwrap.dedent("""\
            prefix dc: <http://purl.org/dc/terms/>
            prefix cat: <http://dbpedia.org/resource/Category:>
            select distinct ?category ?article
            where
            {{
              ?article dc:subject ?category .
              ?category skos:broader{{0,{}}} cat:{} .
            }}
            limit {}
            offset {}""").format(depth, root, limit, offset)


def article_categories_query(article_uri, limit=DBPEDIA_LIMIT, offset=0):
    """Creates a SPARQL query for retrieving the article's categories from DBpedia."""
    return textwrap.dedent("""\
            prefix dc: <http://purl.org/dc/terms/>
            select distinct ?category
            where
            {{
              <{}> dc:subject ?category .
            }}
            limit {}
            offset {}""").format(article_uri, limit, offset)


def supercategories_query(category_uri, limit=DBPEDIA_LIMIT, offset=0):
    """Creates a SPARQL query for retrieving the categorie's supercategories from DBpedia."""
    return textwrap.dedent("""\
            select distinct ?category
            where
            {{
              <{}> skos:broader ?category .
            }}
            limit {}
            offset {}""").format(category_uri, limit, offset)


def subcategories_query(category_uri, limit=DBPEDIA_LIMIT, offset=0):
    """Creates a SPARQL query for retrieving the categorie's supercategories from DBpedia."""
    return textwrap.dedent("""\
            select distinct ?category
            where
            {{
              ?category skos:broader <{}> .
            }}
            limit {}
            offset {}""").format(category_uri, limit, offset)


def query_dbpedia(query, timeout=DBPEDIA_TIMEOUT):
    """Queries the DBpedia SPARQL endpoint."""
    params = urllib.parse.urlencode({'query': query, 'format': TSV_FORMAT,
                                     'timeout': timeout})
    url = DBPEDIA_ENDPOINT + '?' + params
    unquote = lambda s: re.search('^"(.*)"$', s).group(1)
    urldecode = lambda s: urllib.parse.unquote(s)
    def parse(line):
        uris = urldecode(line.decode('UTF-8')).split('\t')
        return list(map(unquote, uris))
    lines = [parse(line) for line in urllib.request.urlopen(url)]
    return lines[1:]  # skipping the header


def page_query_results(query_fn):
    """Pages through the query results of DBpedia.

    Executes the query repeatedly with increasing offsets, returns the
    collected results. Parameter 'query-fn' should be a function that takes
    the offset and returns a query; Parameter 'limit' defines the page size."""
    results = []
    offset = 0
    while True:

        logging.debug('Paging some query:\n' + query_fn(offset))
        more_results = query_dbpedia(query_fn(offset))
        logging.debug('Got {} results'.format(len(more_results)))
        if len(more_results) > 0:
            results.extend(more_results)
            offset += len(more_results)
        else:
            return results


def get_category_pairs(root, depth, limit=DBPEDIA_LIMIT):
    """Retrieves the parent-child category pairs from DBpedia.

    Starts from the root category and goes down the hierarchy
    until the specified depth is reached."""
    logging.info("Retrieving the category pairs for {}".format(root))
    query_fn = lambda offset: category_pairs_query(root, depth, limit, offset)
    return page_query_results(query_fn)


def get_category_articles(root, depth, limit=DBPEDIA_LIMIT):
    """Retrieves the category-article pairs from DBpedia.

    The categories start from the root category and go down the subcategory
    hierarchy until the specified depth is reached."""
    logging.info("Retrieving the category-article pairs for {}".format(root))
    query_fn = lambda offset : category_articles_query(root, depth, limit, offset)
    return page_query_results(query_fn)


def get_article_categories(article_uri):
    """Retrieves the article's parent categories from DBpedia."""
    logging.info("Retrieving the parent categories for {}".format(article_uri))
    results = query_dbpedia(article_categories_query(article_uri))
    return [result[0] for result in results]


def get_supercategories(category_uri):
    """Retrieves the category's supercategories from DBpedia."""
    logging.info("Retrieving the parent categories for {}".format(category_uri))
    results = query_dbpedia(supercategories_query(category_uri))
    return [result[0] for result in results]


def get_subcategories(category_uri):
    """Retrieves the category's subcategories from DBpedia."""
    logging.info("Retrieving the child categories for {}".format(category_uri))
    results = query_dbpedia(subcategories_query(category_uri))
    return [result[0] for result in results]


def save_pairs(pairs, filename):
    logging.info("Saving {} pairs to {}".format(len(pairs), filename))
    lines = (parent + '\t' + child for parent, child in pairs)
    with open(filename, 'w', encoding='utf-8') as file:
        file.write('\n'.join(lines))


def read_tuples(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        return [tuple(line.rstrip().split('\t')) for line in lines]


def get_parents(uri):
    title = strip_dbpedia_prefix(uri)
    uri = title_to_uri(urllib.parse.quote(title, safe='~@#$&()*!+=:;,./\''),
                       category=is_category_uri(uri))
    if is_category_uri(uri):  # important to test the article uri first
        return get_supercategories(uri)
    elif is_article_uri(uri):
        return get_article_categories(uri)
    else:
        raise ValueError('Invalid DBpedia uri: {}'.format(uri))


def get_subcats(uri):
    if not is_category_uri(uri):
        raise ValueError('Invalid DBpedia category uri: {}'.format(uri))
    title = strip_dbpedia_prefix(uri)
    uri = title_to_uri(urllib.parse.quote(title, safe='~@#$&()*!+=:;,./\''),
                       category=True)
    return get_subcategories(uri)
