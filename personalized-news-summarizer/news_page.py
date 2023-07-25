from flask import Flask, render_template, request
from datetime import datetime
import logging
import json
from helpers.news_fetcher import generate_data

class ArticleHolder:
    """This class holds a list of articles of the same category.
    That category is stored in the variable `title`."""
    def __init__(self, title:str, articles:list):
        self.title = title
        self.articles = articles

logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)8s:%(name)s:  %(message)s')
app = Flask(__name__)

def get_url(bucket_name:str, source_file:str)->str:
    return f'https://storage.googleapis.com/{bucket_name}/{source_file}'

@app.route('/')
def homepage():
    with open('site_args.json', 'r') as f:
        args = json.load(f)

    date = datetime.now().strftime('%B %d, %Y')
    search_params = {
        'num_articles' : int(args.get('num_articles', 2)),
        'categories' : args.get('categories', 'General').split(',')
    }
    article_holders = [ArticleHolder(title, article_list) 
                    for title, article_list 
                    in generate_data(search_params).items()]
    # article_holders = []
    bucket_name = args.get('bucket_name')
    return render_template('index.html', 
        curr_date = date,
        article_holders = article_holders,
        stylesheet_url = get_url(bucket_name, 'static/main.css'),
        favicon_url = get_url(bucket_name, 'static/favicon.ico'))

if __name__ == '__main__':
    app.run()
