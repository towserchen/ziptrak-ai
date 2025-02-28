from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.datastructures import UploadFile
from fastapi.responses import JSONResponse
from utils.helper import ajax
from utils import verify
import json
import time
import os

class APIVerification(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        secret = os.environ.get('API_SECRET_KEY', 'ziptrak')

        if request.method != 'POST':
            return ajax(0, 'Invalid request')

        form_data = await request.form()

        form_field_list = {}
        file_list = {}

        for key, value in form_data.multi_items():
            if isinstance(value, UploadFile):
                file_list[key] = value
            else:
                form_field_list[key] = value
        
        request.state.form_field_list = form_field_list
        request.state.file_list = file_list

        if not 'token' in form_field_list:
            return ajax(0, 'Invalid token')
        
        if not verify.verify_access_token(secret, form_field_list['token'], int(time.time())):
            return ajax(0, 'Invalid token')
        
        response = await call_next(request)

        body = b""

        async for chunk in response.body_iterator:
            body += chunk

        body_str = body.decode()

        try:
            body_data = json.loads(body_str)
            if isinstance(body_data, dict):
                body_data['token'] = verify.generate_access_token(secret, int(time.time()))
                body_str = json.dumps(body_data)
        except json.JSONDecodeError:
            return ajax(0, 'Invalid request')

        new_response = JSONResponse(content=json.loads(body_str))

        return new_response
    

    def _cached_receive(self, request: Request):
        return {
            "type": "http",
            "body": request._body_cache.read(),
            "more_body": False,
        }

        