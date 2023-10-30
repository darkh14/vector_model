import fastapi
import uvicorn
import traceback

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
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
async def set_db_connector(request: Request, call_next):

    processor = Processor()
    c_method_descr = processor.get_requests_methods_description()

    methods_url_list = [el['path'][10:] for el in c_method_descr if el['path'].startswith('{db_name}/')]

    request_url = request.url.path
    if request_url.startswith('/'):
        request_url = request_url[1:]

    if request_url.endswith('/'):
        request_url = request_url[:-1]

    db_name = ''
    url_parts = request_url.split('/')

    if len(url_parts) > 1 and '/'.join(url_parts[1:]) in methods_url_list:
        db_name = url_parts[0]
    if db_name:
        # noinspection PyBroadException
        try:
            initialize_connector_by_name(db_name)
        except VMBaseException as exc:
            error_text = str(exc)
            print(error_text)
            return JSONResponse(
                status_code=exc.get_http_status() or fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=error_text,
                headers=exc.get_http_headers()
            )
        except Exception:
            return JSONResponse(
                status_code=fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=traceback.format_exc()
            )

    result = await call_next(request)
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
        http_status = fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR

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

    return JSONResponse(
        status_code=fastapi.status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=exc.errors())


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
