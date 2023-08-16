from news_fetcher import generate_data
import functions_framework
import json
import base64
from utils.mongodb import get_client

@functions_framework.cloud_event
def fetch_news(cloud_event):
    request_b64 = cloud_event.data['message']['data']
    request_str = base64.b64decode(request_b64).decode('utf-8')
    request_json = json.loads(request_str)

    # generate data
    data = generate_data({
        'num_articles' : request_json.get('num_articles', 2),
        'categories' : request_json.get('categories', 'General').split(',')
    })
    
    # write data to MongoDB
    with get_client('db_info.json', 'conn_info.json') as client:
        db = client['news']
        collection = db['articles']

        for cateogry, articles in data.items():
            documents = []
            for article in articles:
                documents.append({
                    'category' : cateogry,
                    'api_src' : article.api_src,
                    'url' : article.url,
                    'title' : article.title,
                    'source' : article.source,
                    'time' : article.time,
                    'description' : article.description,
                    'img_url' : article.img_url,
                    'article_text' : article.text
                })
            collection.insert_many(documents)