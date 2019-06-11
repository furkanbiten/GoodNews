import urllib2
from bs4 import BeautifulSoup
import json
import os
import tqdm
import requests
import unidecode
import urlparse
import multiprocessing
import posixpath
from goose import Goose
import sys
# from joblib import Parallel, delayed
from six.moves.html_parser import HTMLParser
from multiprocessing.dummy import Pool as ThreadPool

def resolveComponents(url):
    """
     resolveComponents('http://www.example.com/foo/bar/../../baz/bux/')
    'http://www.example.com/baz/bux/'
     resolveComponents('http://www.example.com/some/path/../file.ext')
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

def retrieve_articles(m):
    
#     for m in months:
    articles = json.load(open(root + "api/nyarticles_%s_%s.json" % (year, m), 'r'))
    month_data = {}
    leftovers = {}
    len_articles = len(articles['response']['docs'])
    for num, a in enumerate(articles['response']['docs']):
        try:
            data = {}
            # using resolveComponents here, if some bad urls exist.
            url = resolveComponents(a['web_url'])
            extract = g.extract(url=url)
            # Save the data before since we might save the data inside the if
            data['headline'] = a.get('headline', None)
            data['article_url'] = url
            data['article'] = unidecode.unidecode(extract.cleaned_text)
            data['abstract'] = a.get('abstract', None)
    #             print url
            if a['multimedia']:
                data['images'] = {}
                soup, figcap = get_soup(url)
                figcap = [c for c in figcap if c.text]
                for ix, cap in enumerate(figcap):
    #                     if im.attrs.get('data-mediaviewer-caption', 0):
    #                     if cap.text:
    #                     im = cap.find_previous('img')
    #                     img_url = resolveComponents(im.attrs['src'])
                    if cap.parent.attrs.get('itemid', 0):
                        img_url = resolveComponents(cap.parent.attrs['itemid'])
    #                     urllib.urlretrieve(img_url, root + "/images/%s_%d.%s" % (a['_id'], ix, img_url.split('.')[-1]))
                        img_data = requests.get(img_url, stream=True).content
    #     , img_url.split('.')[-1]
                        with open(os.path.join(root + "/images/%s_%d.jpg" % (a['_id'], ix)), 'wb') as f:
    #                         shutil.copyfileobj(img_data, f)
                            f.write(img_data)
    #                     data['images'].update({ix : im['data-mediaviewer-caption']})
                        text = cap.get_text().split('credit')[0]
                        text = text.split('Credit')[0]
                        data['images'].update({ix : text})
            sys.stdout.write('\r%d/%d text documents processed...' % (num, len_articles))
            sys.stdout.flush()
            month_data[a['_id']] = data
        except Exception as e:
            leftovers[a['_id']] = a['web_url']
            print e, url
            
    json.dump(month_data, open(data_root + 'nytimes_%d_%d.json' % (year, m), 'wb'))
    json.dump(leftovers, open(data_root + 'leftovers_%d_%d.json' % (year, m), 'wb'))

def main(num_pool, months):
    pool = ThreadPool(num_pool)
    leftovers = {}
    pool.map(retrieve_articles, months)
    pool.close()
    pool.join()
    json.dump(leftovers, open(root + "leftovers_%s.json" % year, 'w'))

if __name__ == '__main__':
    hdr = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
        'Accept-Encoding': 'none',
        'Accept-Language': 'en-US,en;q=0.8',
        'Connection': 'keep-alive'}

    h = HTMLParser()
    root = "./"
    data_root = root + "data/"
    g = Goose()

    # This is a special case since 2018 had only 6 months by the time the data was getting collected.
    # this year variable used in retrieve articles function
    year = 2018
    main(6, range(1, 7))

    num_pool = 12
    months = range(1, 13)
    years = range(2010, 2018)
    for y in reversed(years):
        year = y
        main(num_pool, months)