import fastapi
import uvicorn

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.openapi import utils

from vm_versions import get_version
from vm_settings import controller as settings_controller
from processor import Processor
from api_types import RequestValidationErrorSchema
from vm_logging.exceptions import GeneralException

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
                                          '<h3>Connection established TEST TEST</h3>')


# noinspection PyUnusedLocal
@app.exception_handler(GeneralException)
async def internal_exception_handler(request: fastapi.Request, exc: GeneralException):
    """
    Handler to process exceptions adds status and error text
    @param request: input request
    @param exc: formed exception
    @return: json response of error
    """
    content = jsonable_encoder({'status': 'error',
                'error_text': exc.message})

    return JSONResponse(
        status_code=fastapi.status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=content)


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

    print(content)

    return JSONResponse(
        status_code=fastapi.status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=content)


def initialize_validation_error_schema():
    """
    Correct validation error schema
    @return: result validation error schema
    """
    validation_schema = RequestValidationErrorSchema.model_json_schema()

    validation_schema.update({"title": "HTTPValidationError"})

    custom_validation_error_response_definition = dict(validation_schema)

    utils.validation_error_response_definition = custom_validation_error_response_definition


if __name__ == "__main__":

    http_processor = Processor()

    method_descr_list = http_processor.get_requests_methods_description()

    for method_descr in method_descr_list:
        if method_descr['http_method'] == 'get':
            api_method = app.get('/{}/'.format(method_descr['path']))(method_descr['func'])
        elif method_descr['http_method'] == 'post':
            api_method = app.post('/{}/'.format(method_descr['path']))(method_descr['func'])

    initialize_validation_error_schema()

    if TEST_MODE:
        uvicorn.run(app, host="127.0.0.1", port=8070, log_level="info")
