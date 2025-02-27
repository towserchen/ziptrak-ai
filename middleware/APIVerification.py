from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from io import BytesIO
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

        body = await request.body()
        form_data = await request.form()

        if not 'token' in form_data:
            return ajax(0, 'Invalid token')

        request._body_cache = BytesIO(body)
        request._receive = lambda: self._cached_receive(request)
        
        response = await call_next(request)

        body = b""

        async for chunk in response.body_iterator:
            body += chunk

        body_str = body.decode()

        try:
            body_data = json.loads(body_str)
            if isinstance(body_data, dict):
                body_data['token'] = verify.generate_access_token(secret, time.time())
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

        