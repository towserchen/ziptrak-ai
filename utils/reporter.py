import os
import time
import requests
from utils import verify

API_BASE_URL = os.environ.get('API_BASE_URL', 'http://127.0.0.1:8000')
API_SECRET_KEY = os.environ.get('API_SECRET_KEY', 'ziptrak')

def online():
    data = {
        'token': verify.generate_access_token(API_SECRET_KEY, str(time.time()))
    }

    api_url = API_BASE_URL + '/node/online'
    response = requests.post(api_url, data=data)

    retryCount = 1

    while retryCount < 5:
        if response.status_code != 200:
            time.sleep(3)

            retryCount += 1
            response = requests.post(api_url, data=data)
        else:
            break
    
    return True



def available():
    data = {
        'token': verify.generate_access_token(API_SECRET_KEY, str(time.time()))
    }

    api_url = API_BASE_URL + '/node/idle'
    response = requests.post(api_url, data=data)

    retryCount = 1

    while retryCount < 5:
        if response.status_code != 200:
            time.sleep(3)

            retryCount += 1
            response = requests.post(api_url, data=data)
        else:
            break
    
    return True