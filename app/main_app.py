import fastapi
import uvicorn

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.requests import Request

from vm_versions import get_version
from vm_settings import controller as settings_controller
from processor import Processor
from vm_logging.exceptions import VMBaseException
from db_processing import initialize_connector_by_name


TEST_MODE: bool = bool(settings_controller.get_var('TEST_MODE'))
""" In test mode errors raise but not process """

app = fastapi.FastAPI()


@app.get('/')
async def get():
    """
    Root method returns html ok description
    @return: HTML response with ok micro html
    """
    return fastapi.responses.HTMLResponse('<h2>VBM module v. {}</h2> <br> '.format(get_version()) +
                                          '<h3>Connection established</h3>')


@app.middleware('http')
def test(request: Request, call_next):
    db_name = ''
    url_parts = request.url.path.split('/')
    if len(url_parts) > 1 and url_parts[1].startswith('db_'):
        db_name = url_parts[1]
    if db_name:
        initialize_connector_by_name(db_name)
    result = call_next(request)
    return result


# noinspection PyUnusedLocal
@app.exception_handler(VMBaseException)
def internal_exception_handler(request: fastapi.Request, exc: VMBaseException):
    """
    Handler to process exceptions adds status and error text
    @param request: input request
    @param exc: formed exception
    @return: json response of error
    """

    http_status = None
    http_headers = None
    if isinstance(exc, VMBaseException):
        http_status = exc.get_http_status()
        http_headers = exc.get_http_headers()

    if not http_status:
        http_status = fastapi.status.HTTP_400_BAD_REQUEST

    message = str(exc)

    print(message)

    return JSONResponse(
        status_code=http_status,
        content=message,
        headers=http_headers
    )


# noinspection PyUnusedLocal
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: fastapi.Request,
                                       exc: RequestValidationError):
    """
    Handler to process validation error. Adds status and error text
    @param request: input request
    @param exc: formed exception
    @return: json response of error (422 http status)
    """

    content = jsonable_encoder({'status': 'error',
                'error_text': 'Request validation error!', 'details': exc.errors()})

    return JSONResponse(
        status_code=fastapi.status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=content)


http_processor = Processor()

method_descr_list = http_processor.get_requests_methods_description()

for method_descr in method_descr_list:

    pr_method = method_descr['func']
    if method_descr['http_method'] == 'get':
        api_method = app.get('/{}/'.format(method_descr['path']))(pr_method)
    elif method_descr['http_method'] == 'post':
        api_method = app.post('/{}/'.format(method_descr['path']))(pr_method)

if TEST_MODE:
    uvicorn.run(app, host="127.0.0.1", port=8070, log_level="info")
