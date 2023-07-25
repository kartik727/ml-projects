import requests
from requests.models import HTTPError
from dataclasses import dataclass
from bs4 import BeautifulSoup
from bs4.element import Comment
from datetime import datetime
from dateutil import parser
from PIL import Image
from urllib.parse import urlparse
import newspaper
import json
from helpers import api_keys
import logging

summarizer_api_params = {
    # You can get this for free
    'api_key'           : api_keys.HUGGINGFACE_API_KEY,
    'endpoint'          : 'https://api-inference.huggingface.co/models/{model}',
    'model'             : 'facebook/bart-large-cnn'
}

news_api_params = {
    # You can get this for free (Free tier API)
    'bing_search' : {
        'api_key'       : api_keys.BING_API_KEY,
        'endpoint'      : 'https://api.bing.microsoft.com/{version}/{endpoint}',
        'version'       : 'v7.0',
        'market'        : 'en-US',
        'search_endpt'  : 'search',
        'news_endpt'    : 'news'
    },

    # You can get this for free
    'news_org' : {
        'api_key'       : api_keys.NEWS_ORG_API_KEY,
        'endpoint'      : 'https://newsapi.org/{version}/{endpoint}',
        'version'       : 'v2',
        'search_endpt'  : 'everything',
        'news_endpt'    : 'top-headlines'
    }
}

class KeyNotFoundError(Exception):
    pass

class NewsSummarizer:
    def __init__(self, endpoint:str, model:str, api_key:str):
        self._endpoint = endpoint
        self._model = model
        self._api_key = api_key

    @property
    def url(self)->str:
        return self._endpoint.format(model=self._model)

    @property
    def headers(self)->dict:
        return {'Authorization': f'Bearer {self._api_key}'}

    @property
    def payload_params(self)->dict:
        return {
            'max_length' : 2048
        }

    def batch_summarize(self, articles:list[str])->list[str]:
        payload = {'inputs': articles, 'parameters': self.payload_params}
        response = requests.post(self.url, headers=self.headers, json=payload)
        response.raise_for_status()
        return [r['summary_text'] for r in json.loads(response.content)]

    def serial_summarize(self, articles:list[str])->list[str]:
        summaries = []
        for article in articles:
            try:
                payload = {'inputs': [article], 'parameters': self.payload_params}
                response = requests.post(self.url, headers=self.headers, json=payload)
                response.raise_for_status()
                summaries.append(json.loads(response.content)[0]['summary_text'])
            except HTTPError:
                summaries.append('Error')
        return summaries

    def __call__(self, article:str)->str:
        return self.batch_summarize([article])[0]

@dataclass
class Article:
    api_src     : str
    url         : str
    title       : str
    source      : str
    time        : datetime
    description : str       = 'No description provided'
    img_url     : str       = None

    def __str__(self):
        return f'''Article from {self.api_src} - url:"{self.url}", \
        title:"{self.title}", src:"{self.source}", dt:"{self.time}", \
        desc:"{self.description[:10]}...", img_url:"{self.img_url}"'''

    def __repr__(self):
        return self.__str__()

    @property
    def time_str(self):
        return self.time.strftime('%Y-%m-%d %H-%M-%S')

class NewsQuery:
    CATEGORIES = ['Business', 'Entertainment', 'General',
                    'Health', 'Science', 'Technology', 'Sports']
    SUBCATEGORIES = {
        'Entertainment' : ['Movies and TV', 'Music'],
        'Sports' : ['Golf', 'MLB', 'NBA', 'NFL', 'NHL',
                    'Soccer', 'Tennis', 'CFB', 'CBB']
    }
    def __init__(self, category:str, subcategory:str=None):
        if category in self.CATEGORIES:
            self._category = category
        else:
            raise ValueError(f'Invalid category: `{category}`')

        if subcategory is not None:
            if subcategory in self.SUBCATEGORIES[category]:
                self._subcategory = subcategory
            else:
                raise ValueError(f'Invalid subcategory `{subcategory}` for category `{category}`')
        else:
            self._subcategory = None

    @property
    def category(self):
        return self._category

    @property
    def subcategory(self):
        return self._subcategory

@dataclass
class NewsAPI:
    api_key: str
    endpoint: str

    def search(self, query:str, **kwargs)->list[Article]:
        raise NotImplementedError('Search method has not been implemented by this class.')

    def news(self, category:str, **kwargs)->list[Article]:
        raise NotImplementedError('News method has not been implemented by this class.')

@dataclass
class BingNewsAPI(NewsAPI):
    CATEGORIES = {
        'Business':{'name':'Business'},
        'Entertainment':{'name':'Entertainment',
            'subcats': {'Movies and TV':'Entertainment_MovieAndTV', 'Music':'Entertainment_Music'}},
        'General':{'name':None},
        'Health':{'name':'Health'},
        'Science':{'name':'Science'},
        'Sports':{'name':'Sports',
            'subcats': {'Golf':'Sports_Golf', 'MLB':'Sports_MLB', 'NBA':'Sports_NBA', 'NFL':'Sports_NFL',
                        'NHL':'Sports_NHL', 'Soccer':'Sports_Soccer', 'Tennis':'Sports_Tennis',
                        'CFB':'Sports_CFB', 'CBB':'Sports_CBB'}},
        'Technology':{'name':'Technology'}
    }
    api_key     : str
    endpoint    : str
    version     : str = 'v7.0'
    market      : str = 'en-US'
    search_endpt: str = 'search'
    news_endpt  : str = 'news'

    @property
    def search_url(self)->str:
        return self.endpoint.format(version=self.version, endpoint=self.search_endpt)

    @property
    def news_url(self)->str:
        return self.endpoint.format(version=self.version, endpoint=self.news_endpt)

    def _parse_article(self, src:str, article_dict:dict)->Article:
        try:
            date_published = parser.parse(article_dict['datePublished'])
        except KeyError:
            date_published = None
        return Article(
            src,
            article_dict['url'],
            article_dict['name'],
            article_dict.get('provider', [dict()])[0].get('name', 'Unknown'),
            date_published,
            description = article_dict.get('description', 'No description provided'),
            img_url = article_dict.get('image', dict()).get('thumbnail', dict()).get('contentUrl')
        )

    def search(self, query:str, **kwargs)->list[Article]:
        params = { 'q': query, 'mkt': self.market }
        headers = { 'Ocp-Apim-Subscription-Key': self.api_key }
        r = requests.get(self.search_url, headers=headers, params=params, **kwargs)
        return [self._parse_article('BingSearch', a) for a in r.json()['webPages']['value']]

    def news(self, news_query:NewsQuery, **kwargs)->list[Article]:
        params = {'mkt': self.market }
        category, subcategory = news_query.category, news_query.subcategory
        cat = self.CATEGORIES[category]['name']
        if cat is not None:
            if subcategory is None:
                params['category'] = cat
            else:
                params['category'] = self.CATEGORIES[category]['subcats'][subcategory]
        headers = { 'Ocp-Apim-Subscription-Key': self.api_key }
        r = requests.get(self.news_url, headers=headers, params=params, **kwargs)
        return [self._parse_article('BingNews', a) for a in r.json()['value']]

@dataclass
class NewsOrgAPI(NewsAPI):
    CATEGORIES = {
        'Business':{'name':'business'},
        'Entertainment':{'name':'entertainment',
            'subcats': {'Movies and TV':'Movies and TV', 'Music':'Music'}},
        'General':{'name':'general'},
        'Health':{'name':'health'},
        'Science':{'name':'science'},
        'Sports':{'name':'sports',
            'subcats': {'Golf':'Golf', 'MLB':'MLB', 'NBA':'NBA', 'NFL':'NFL',
                        'NHL':'NHL', 'Soccer':'Soccer', 'Tennis':'Tennis',
                        'CFB':'CFB', 'CBB':'CBB'}},
        'Technology':{'name':'technology'}
    }
    api_key     : str
    endpoint    : str
    version     : str = 'v2'
    language    : str = 'en'
    sort_by     : str = 'relevancy'
    page_size   : int = 25
    search_endpt: str = 'everything'
    news_endpt  : str = 'top-headlines'

    @property
    def search_url(self)->str:
        return self.endpoint.format(version=self.version, endpoint=self.search_endpt)

    @property
    def news_url(self)->str:
        return self.endpoint.format(version=self.version, endpoint=self.news_endpt)

    def _parse_article(self, article_dict:dict)->Article:
        description = 'No description provided.' if article_dict.get('description') is None else article_dict.get('description')
        return Article(
            'NewsOrg',
            article_dict['url'],
            article_dict['title'],
            article_dict.get('source', dict()).get('name', 'Unknown'),
            parser.parse(article_dict['publishedAt']),
            description = description,
            img_url = article_dict.get('urlToImage')
        )

    def search(self, query:str, **kwargs)->list[Article]:
        params = { 'q': query, 'language': self.language, 'sortBy': self.sort_by, 'pageSize': self.page_size, **kwargs}
        headers = { 'X-Api-Key': self.api_key}
        r = requests.get(self.search_url, headers=headers, params=params)
        r.raise_for_status()
        return [self._parse_article(a) for a in r.json()['articles']]

    def news(self, news_query:NewsQuery, **kwargs)->list[Article]:
        category, subcategory = news_query.category, news_query.subcategory
        cat = self.CATEGORIES[category]['name']
        params = { 'category': cat, 'language': self.language, 'pageSize': self.page_size, **kwargs}
        headers = { 'X-Api-Key': self.api_key}
        if subcategory is not None:
            subcat = self.CATEGORIES[category]['subcats'][subcategory]
            params['q'] = subcat
        r = requests.get(self.news_url, headers=headers, params=params)
        r.raise_for_status()
        return [self._parse_article(a) for a in r.json()['articles']]

class NewsFetcher:
    def __init__(self, summarizer : NewsSummarizer,
            news_apis : list[NewsAPI], /,
        ):
        self.summarizer = summarizer
        self.news_apis = news_apis

    def get_results(self, query:str, api:NewsAPI):
        return api.search(query)

    def _tag_visible(self, element):
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        if isinstance(element, Comment):
            return False
        return True

    def _text_from_html(self, body):
        soup = BeautifulSoup(body, 'html.parser')
        texts = soup.findAll(string=True)
        visible_texts = filter(self._tag_visible, texts)
        return u' '.join(t.strip() for t in visible_texts)

    def _text_from_url(self, url:str):
        article = newspaper.Article(url=url, language='en')
        article.download()
        article.parse()
        return article.text

    def _get_image(self, img_url:str):
        if img_url is None:
            return None

        try:
            r = requests.get(img_url, stream=True, timeout=2)
            return Image.open(r.raw)
        except:
            return None

    def _filter(self, articles:list[Article])->list[Article]:
        filtered = []
        for article in articles:

            # remove msn articles (can't be summarized properly)
            domain = urlparse(article.url).netloc
            if domain == 'www.msn.com':
                continue

            # remove youtube articles
            if domain == 'www.youtube.com':
                continue

            # any other filters go here
            # ...

            # add to list if not yet filtered out
            filtered.append(article)

        return filtered

    def process_articles(self, articles:list[Article])->list[Article]:
        processed_articles = []
        for article in self._filter(articles):
            try:
                webpage = requests.get(article.url, timeout=2)
            except:
                continue
            if webpage.status_code == 200:
                try:
                    article.article_text = self._text_from_url(article.url)
                except:
                    article.article_text = self._text_from_html(webpage.content)

                # Remove articles that are too small
                if len(article.article_text)<100:
                    continue

                article.image = self._get_image(article.img_url)
                processed_articles.append(article)

        if len(processed_articles)==0:
            print('No articles to process')
            return []

        texts = [a.article_text for a in processed_articles]

        summaries = self.summarizer.serial_summarize(texts)

        for pa, summary in zip(processed_articles, summaries):
            pa.summary = summary

        return processed_articles

    def news(self, news_query:NewsQuery, /, *, max_articles:int=10):
        all_articles = {}

        for news_api in self.news_apis:
            article_list = news_api.news(news_query)
            for article in article_list:
                all_articles[article.url] = article
        all_articles = [a for u,a in all_articles.items()][:max_articles]
        all_articles = self.process_articles(all_articles)
        return all_articles

def generate_data(search_params:dict):
    # Initialize the news summarizer
    if summarizer_api_params['api_key'] is not None:
        summarizer = NewsSummarizer(**summarizer_api_params)
    else:
        raise KeyNotFoundError('No API key provided to summarize the news')

    # Initialize the news apis
    news_apis = []
    if news_api_params['news_org']['api_key'] is not None:
        news_org_api = NewsOrgAPI(**news_api_params['news_org'])
        news_apis.append(news_org_api)
    if news_api_params['bing_search']['api_key'] is not None:
        bing_api = BingNewsAPI(**news_api_params['bing_search'])
        news_apis.append(bing_api)

    # Initialize the news fetcher
    if len(news_apis) > 0:
        news_fetcher = NewsFetcher(summarizer, news_apis)
        logging.info('Ready to fetch news')
    else:
        raise KeyNotFoundError('No API keys provided to get the news')

    results = {}
    for qi in search_params['categories']:
        logging.info(f'Fetching news for {qi}')
        query = NewsQuery(qi)
        articles = news_fetcher.news(query, max_articles=search_params['num_articles'])
        results[qi] = [a for a in articles]
        logging.info(f'Fetched {len(articles)} articles for {qi}')
    
    return results

