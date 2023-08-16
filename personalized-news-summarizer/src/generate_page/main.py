import json
import base64
import functions_framework
from google.cloud import storage
from flask_frozen import Freezer
from news_page import app

def upload_to_bucket(bucket_name, source_files)->str:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    responses = []
    for source_file in source_files:
        blob = bucket.blob(source_file)
        if source_file.endswith('.html'):
            blob.upload_from_filename(source_file, content_type='text/html; charset=utf-8')
        elif source_file.endswith('.css'):
            blob.upload_from_filename(source_file, content_type='text/css; charset=utf-8')
        else:
            blob.upload_from_filename(source_file)

        responses.append(f'Uploaded {source_file} to bucket {bucket_name} as {blob.name}')
    return '\n'.join(responses)

@functions_framework.cloud_event
def generate_site(cloud_event):
    request_b64 = cloud_event.data['message']['data']
    request_str = base64.b64decode(request_b64).decode('utf-8')
    request_json = json.loads(request_str)

    # write args for Flask app
    site_args = {
        'bucket_name' : request_json['bucket_name']
    }
    with open('site_args.json', 'w') as f:
        json.dump(site_args, f, indent=4)

    # generate site
    freezer = Freezer(app)
    freezer.freeze()

    # upload to bucket
    return upload_to_bucket(
        request_json['bucket_name'],
        request_json['source_files']
        )
