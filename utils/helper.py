from fastapi.responses import JSONResponse

def ajax(status: int = 1, msg: str = 'ok', data: any = None) -> JSONResponse:
    return JSONResponse(content={
        "status": status,
        "msg": msg,
        "data": data
    })