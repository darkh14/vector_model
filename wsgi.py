"""The module for starting execution in product mode.
It is started using the wsgi service.

functions:
    application - main function of module. See docs.
    """

# import http_procession
from typing import Callable, Any
from processor import process


def application(environ: dict[str, any], start_response: Callable[[str, list[bytes]], Callable]) -> dict[str, Any]:
    """the main entry point of the program.
    Executed when accessed via http request using the uwsgi service
    Parameters:
        environ: dict - keys and example values:
            REQUEST_METHOD - POST
            CONTENT_TYPE - application/json
            CONTENT_LENGTH - 515
            REQUEST_URI - /
            PATH_INFO - /
            DOCUMENT_ROOT - /usr/share/nginx/html
            SERVER_PROTOCOL - HTTP/1.1
            REQUEST_SCHEME - https
            HTTPS - on
            REMOTE_ADDR - 141.101.76.238
            REMOTE_PORT - 18642
            SERVER_PORT - 443
            SERVER_NAME - smartx.nfp2b.ml
            HTTP_HOST - smartx.nfp2b.ml
            HTTP_CONNECTION - Keep-Alive
            HTTP_ACCEPT_ENCODING - gzip
            HTTP_X_FORWARDED_FOR - 185.246.91.153
            HTTP_CF_RAY - 763dc09d2bff0bf5-AMS
            HTTP_CONTENT_LENGTH - 515
            HTTP_X_FORWARDED_PROTO - https
            HTTP_CF_VISITOR - {"scheme":"https"}
            HTTP_USER_AGENT - 1C+Enterprise/8.3
            HTTP_ACCEPT - */*
            HTTP_CONTENT_TYPE - application/json
            HTTP_CF_CONNECTING_IP - 185.246.91.153
            HTTP_CF_IPCOUNTRY - RU
            HTTP_CDN_LOOP - cloudflare
            wsgi.input - <uwsgi._Input object at 0x7f78dc0aa180>
            wsgi.file_wrapper - <built-in function uwsgi_sendfile>
            wsgi.version - (1, 0)
            wsgi.errors - <_io.TextIOWrapper name=2 mode='w' encoding='UTF-8'>
            wsgi.run_once - False
            wsgi.multithread - False
            wsgi.multiprocess - True
            wsgi.url_scheme - https
            uwsgi.version - b'2.0.19.1'
            uwsgi.node - b'vps-15617'
        start_response: function object - for sending response, has 2 parameters -
            status - str, ex. "200 OK",
            headers - list of tuples, ex. [('Content-Type','application/json')]
        returns dict of response
    """
    output = process(environ, start_response)

    return output
