from flask import Flask, render_template, request
from datetime import datetime, timedelta
import logging
import json
from dataclasses import dataclass
from utils.mongodb import get_client

@dataclass
class Article:
    title:str
    source:str
    time_str:str
    url:str
    summary:str
    img_url:str

class ArticleHolder:
    """This class holds a list of articles of the same category.
    That category is stored in the variable `title`."""
    def __init__(self, title:str, articles:list):
        self.title = title
        self.articles = articles

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)8s:%(name)s:  %(message)s')
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

def get_url(bucket_name:str, source_file:str)->str:
    return f'https://storage.googleapis.com/{bucket_name}/{source_file}'

@app.route('/')
def homepage():
    with open('site_args.json', 'r') as f:
        args = json.load(f)

    date = datetime.now().strftime('%B %d, %Y')

    with get_client('db_info.json', 'conn_info.json') as client:
        collection = client['news']['articles']
        docs = collection.find({'time' : {'$gt' : datetime.now() - timedelta(days=1)}})
        article_holders = {}
        for doc in docs:
            category = doc['category']
            article_holder = article_holders.get(category, ArticleHolder(category, []))
            article_holder.articles.append(Article(
                title=doc['title'],
                source=doc['source'],
                time_str=doc['time'].strftime('%l:%M %p on %b %d, %Y'),
                url=doc['url'],
                summary=doc.get('summary', 'Summary not available'),
                img_url=doc['img_url']
            ))
            article_holders[category] = article_holder

    article_holders = list(article_holders.values())

    bucket_name = args.get('bucket_name')
    return render_template('index.html',
        curr_date = date,
        article_holders = article_holders,
        stylesheet_url = get_url(bucket_name, 'static/main.css'),
        favicon_url = get_url(bucket_name, 'static/favicon.ico'))

if __name__ == '__main__':
    app.run()