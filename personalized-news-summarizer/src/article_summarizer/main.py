import os
import requests
from requests.models import HTTPError
import json
import functions_framework
import base64
from bson.objectid import ObjectId
from utils.mongodb import get_client

summarizer_api_params = {
    # You can get this for free
    'api_key'           : os.environ.get('huggingface_api_key'),
    'endpoint'          : 'https://api-inference.huggingface.co/models/{model}',
    'model'             : 'facebook/bart-large-cnn'
}

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
    
@functions_framework.cloud_event
def summarize(cloud_event):

    summarizer = NewsSummarizer(**summarizer_api_params)

    with get_client('db_info.json', 'conn_info.json') as client:
        collection = client['news']['articles']

        article = collection.find_one({'summary' : {'$exists':False}})

        if article is None:
            print('No articles to summarize')
            return
        
        article_text = article['article_text']
        summary = summarizer(article_text)

        update_result = collection.update_one(
            {'_id':article['_id']}, 
            {"$set": {'summary':summary}},
            upsert=False)
        
        assert update_result.matched_count == 1, 'No document found'
        assert update_result.modified_count == 1, 'No document modified'

    return f'Summarized article with id {article["_id"]}'

    