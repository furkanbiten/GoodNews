import urllib2
import json
import os
import tqdm
import requests
import unidecode
import argparse
import urlparse
import posixpath
import sys
# from joblib import Parallel, delayed
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

def retrieve_imgs(args):
    #     for m in months:
    # articles = json.load(open(root + "api/nyarticles_%s_%s.json" % (year, m), 'r'))
    thread_num, keys = args
    leftovers = []
    # len_articles = len(articles['response']['docs'])
    len_images = len(keys)

    for num, key in enumerate(keys):
        for ix, url in all_img_url[key].items():
            try:
                img_url = resolveComponents(url)
                img_data = requests.get(img_url, stream=True).content
                with open(os.path.join("../../images/%s_%d.jpg" % (key, int(ix))), 'wb') as f:
                    f.write(img_data)
            except Exception as e:
                leftovers.append(key)
                print e, url

        sys.stdout.write('\r%d/%d text documents processed...' % (num, len_images))
        sys.stdout.flush()
    if leftovers:
        json.dump(leftovers, open(root + 'leftovers_%s' % thread_num, 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input paths
    parser.add_argument('--num_thread', type=int, default=8,
                        help='How many threads you want to use')
    opt = parser.parse_args()
    hdr = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
        'Accept-Encoding': 'none',
        'Accept-Language': 'en-US,en;q=0.8',
        'Connection': 'keep-alive'}

    root = "./"
    num_thread = opt.num_thread
    leftovers = []

    all_img_url = json.load(open('./img_urls_all.json'))
    keys = all_img_url.keys()
    thread_range = len(keys) / num_thread + 1
    args = [(i + 1, keys[thread_range * i: thread_range * (i + 1)]) for i in xrange(num_thread)]
    results = Parallel(n_jobs=num_thread, verbose=0, backend="loky")(map(delayed(retrieve_imgs), args))
