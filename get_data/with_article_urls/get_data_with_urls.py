import urllib2
from bs4 import BeautifulSoup
import json
import os
import requests
import unidecode
import urlparse
import posixpath
from goose import Goose
import argparse
import sys
# from joblib import Parallel, delayed
from six.moves.html_parser import HTMLParser
# from multiprocessing.dummy import Pool as ThreadPool
from joblib import Parallel, delayed


def resolveComponents(url):
    """
    >>> resolveComponents('http://www.example.com/foo/bar/../../baz/bux/')
    'http://www.example.com/baz/bux/'
    >>> resolveComponents('http://www.example.com/some/path/../file.ext')
    'http://www.example.com/some/file.ext'
    """

    parsed = urlparse.urlparse(url)
    new_path = posixpath.normpath(parsed.path)
    if parsed.path.endswith('/'):
        # Compensate for issue1707768
        new_path += '/'
    cleaned = parsed._replace(path=new_path)

    return cleaned.geturl()


def get_soup(url):
    req = urllib2.Request(url, headers=hdr)
    # try:
    page = urllib2.urlopen(req)
    # except urllib2.HTTPError, e:
    #     print num
    #     print e.fp.read()
    soup = BeautifulSoup(page, 'html.parser')
    [s.extract() for s in soup('script')]
    [s.extract() for s in soup('noscript')]
    #     img = soup.find_all('img')
    figcap = soup.find_all("figcaption")
    return soup, figcap


def retrieve_articles(args):
    #     for m in months:
    # articles = json.load(open(root + "api/nyarticles_%s_%s.json" % (year, m), 'r'))
    thread_num, keys = args
    all_data = {}
    leftovers = []
    # len_articles = len(articles['response']['docs'])
    len_articles = len(keys)

    for num, key in enumerate(keys):
        try:
            data = {}
            # using resolveComponents here, if some bad urls exist.
            url = resolveComponents(article_url[key])
            extract = g.extract(url=url)
            # Save the data before since we might save the data inside the if
            # data['headline'] = a.get('headline', None)
            data['article_url'] = url
            data['article'] = unidecode.unidecode(extract.cleaned_text)
            # data['abstract'] = a.get('abstract', None)
            #             print url
            # if a['multimedia']:
            data['images'] = {}
            soup, figcap = get_soup(url)
            figcap = [c for c in figcap if c.text]
            for ix, cap in enumerate(figcap):
                if cap.parent.attrs.get('itemid', 0):
                    img_url = resolveComponents(cap.parent.attrs['itemid'])
                    #                     urllib.urlretrieve(img_url, root + "/images/%s_%d.%s" % (a['_id'], ix, img_url.split('.')[-1]))
                    img_data = requests.get(img_url, stream=True).content
                    #     , img_url.split('.')[-1]
                    with open(os.path.join(root + "/images/%s_%d.jpg" % (key, ix)), 'wb') as f:
                        #                         shutil.copyfileobj(img_data, f)
                        f.write(img_data)
                    #                     data['images'].update({ix : im['data-mediaviewer-caption']})
                    text = cap.get_text().split('credit')[0]
                    text = text.split('Credit')[0]
                    data['images'].update({ix: text})

            sys.stdout.write('\r%d/%d text documents processed...' % (num, len_articles))
            sys.stdout.flush()
            all_data[key] = data
        except Exception as e:
            leftovers.append(key)
            print e, url

    json.dump(all_data, open(data_root + 'nytimes_data_%s.json' % thread_num, 'wb'))
    if leftovers:
        json.dump(leftovers, open(data_root + 'leftovers_%s' % thread_num, 'wb'))
    return all_data

def merge_two_dicts(x, y):
    """Given two dicts, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input paths
    parser.add_argument('--num_thread', type=int, default=1,
                        help='How many threads you want to use')
    opt = parser.parse_args()

    hdr = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
        'Accept-Encoding': 'none',
        'Accept-Language': 'en-US,en;q=0.8',
        'Connection': 'keep-alive'}

    h = HTMLParser()
    root = "../../"
    data_root = "./data/"
    g = Goose()

    num_thread = opt.num_thread
    leftovers = []

    article_url = json.load(open('article_urls.json'))
    keys = article_url.keys()[:10]
    thread_range = len(keys) / num_thread + 1
    args = [(i + 1, keys[thread_range * i: thread_range * (i + 1)]) for i in xrange(num_thread)]
    results = Parallel(n_jobs=num_thread, verbose=0, backend="loky")(map(delayed(retrieve_articles), args))
    combined_data = {}
    for r in results:
        combined_data = merge_two_dicts(combined_data, r)
    json.dump(combined_data, open('combined_data.json', 'wb'))