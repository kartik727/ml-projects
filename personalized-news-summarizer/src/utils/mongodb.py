import os
import sys
import json
from urllib.parse import quote, urlencode
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

def quote_dict(d):
    return {k: quote(v, safe='') for k, v in d.items()}

def get_client(db_info_file:str, conn_info_file:str):
    # Load the database and connection info
    with open(db_info_file, 'r')    as f_db,   \
        open(conn_info_file, 'r')   as f_conn:
        
        db_info = json.load(f_db)
        conn_info = json.load(f_conn)
        credentials = {'username': os.environ.get('mongo_username'),
                       'password': os.environ.get('mongo_password')}
    
    # Create the connection URI
    try:
        quoted_kwargs = quote_dict({'cluster':conn_info['cluster'], **credentials})
    except TypeError:
        raise ValueError(f'Invalid connection info: {conn_info}, {credentials}')

    uri = db_info['base_uri'].format(**quoted_kwargs)
    if conn_info.get('parameters'):
        params = urlencode(conn_info['parameters'])
        uri = f'{uri}/?{params}'
    
    # Create the client
    client = MongoClient(uri, server_api=ServerApi(db_info['api_version']))

    # Verify the connection
    client.admin.command('ping')

    return client