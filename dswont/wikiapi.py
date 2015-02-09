# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import os
if (os.path.basename(os.getcwd()) == 'dswont'):
    os.chdir(os.path.dirname(os.getcwd()))

# <codecell>

import logging
import re
import requests

from dswont.dbpedia import uri_to_title, title_to_uri

API_URL = 'http://en.wikipedia.org/w/api.php'

#  Would be cool if YOU put your name and address in here.
HEADERS = {
    'User-Agent': 'DSW-ONT-IJCAI (http://ijcai-15.org/)'
}

def page(title=None, pageid=None, text=True):
    params = {
        'action': 'query',
        'format': 'json',
        'redirects': ''
    }

    if title:
        params['titles'] = title
    elif pageid:
        params['pageids'] = pageid
    else:
        raise ValueError('Both page title and pageid are empty.')

    if text:
        params['prop'] = 'extracts'
        params['explaintext'] = ''

    response = requests.get(API_URL, params=params, headers=HEADERS)
    data = response.json()

    page = list(data['query']['pages'].values())[0]

    if page.get('missing') == '':
        logging.warning('Wikipedia page not found: title={}, pageid={}'
                        .format(title, pageid))
        return None

    return {
        'title': page['title'],
        'pageid': page['pageid'],
        'text': page.get('extract')
    }


def supercats(uri):
    title = uri_to_title(uri)

    params = {
        'action': 'query',
        'prop': 'categories',
        'format': 'json',
        'titles': 'Category:{}'.format(title),
        'clshow': '!hidden'
    }

    response = requests.get(API_URL, params=params, headers=HEADERS)
    data = response.json()

    page = list(data['query']['pages'].values())[0]

    if page.get('missing') == '':
        logging.warning('Wikipedia page not found: title={}'
                        .format(title))
        return None

    def get_title(cat):
        return title_to_uri(re.sub('^Category:', '', cat['title']),
                            category=True)

    if 'categories' in page:
        return list(map(get_title, page['categories']))
    else:
        logging.warning('Category: {} has no parent categories.'
                        .format(title))
        return []


def subcats(uri):
    title = uri_to_title(uri)

    params = {
        'action': 'query',
        'list': 'categorymembers',
        'cmtitle': 'Category:{}'.format(title),
        'cmtype': 'subcat',
        'cmlimit': '500',
        'format': 'json'
    }

    response = requests.get(API_URL, params=params, headers=HEADERS)
    data = response.json()

    categories = list(data['query']['categorymembers'])

    def get_uri(cat):
        return title_to_uri(re.sub('^Category:', '', cat['title']),
                            category=True)

    return list(map(get_uri, categories))

# print(list(uri_to_title(cat) for cat in supercats('http://dbpedia.org/resource/Category:WikiLeaks')))

