""" Module for defining actions of vm_security

    functions:
        get_actions() returning dict of actions {request_type, action_object}

"""

__all__ = ['get_actions']

from typing import Callable, Any
from . import api_types
from fastapi import Depends
from fastapi.security import OAuth2PasswordRequestForm
from . import controller
from vm_settings import get_var
from vm_logging.exceptions import SecurityException


def _get_token(form_data: OAuth2PasswordRequestForm = Depends()) -> api_types.OutputToken:

    users = controller.UsersController()

    if not users.get_user(get_var('ROOT_USER')):
        users.create_user(get_var('ROOT_USER'), get_var('ROOT_PASSWORD'))

    user = users.get_user(form_data.username)

    if not user:
        raise SecurityException("Incorrect user or password")

    if not users.validate_password(user['name'], form_data.password):
        raise SecurityException("Incorrect user or password")

    token = users.generate_token(form_data.username)

    return api_types.OutputToken.model_validate({'access_token': token, 'token_type': 'bearer'})


def _set_use_authentication(use: bool) -> str:
    controller.set_use_authentication(use)

    return 'Authentication use is {}. Changes will be applied after service reboot!'.format('ON' if use else 'OFF')


def _get_use_authentication() -> bool:
    return controller.get_use_authentication()


def _get_authentication_enabled() -> bool:
    return controller.get_authentication_enabled()


def _create_user(user: api_types.InputUser) -> str:
    user_dict = user.model_dump()
    controller.create_user(user_dict['username'], user_dict['password'])
    return 'User "{}" created'.format(user_dict['username'])


def _delete_user(username: str) -> str:
    controller.delete_user(username)
    return 'User "{}" is deleted'.format(username)


def _get_user(username: str) -> api_types.User:
    return api_types.User.model_validate(controller.get_user(username))


def _get_all_users() -> list[api_types.User]:
    result = controller.get_all_users()
    return [api_types.User.model_validate(usr) for usr in result]


def _set_password(user: api_types.InputUser) -> str:
    user_dict = user.model_dump()
    controller.set_user_password(user_dict['username'], user_dict['password'])
    return 'Password is set'


def _set_enabled(username: str) -> str:
    controller.set_user_enabled(username)
    return 'User "{}" is enabled now'.format(username)


def _set_disabled(username: str) -> str:
    controller.set_user_disabled(username)
    return 'User "{}" is disabled now'.format(username)


def get_actions() -> list[dict[str, Callable]]:
    """ forms actions dict available for vm_logging
    :return: dict of available actions (functions)
    """

    result = list()

    result.append({'name': 'get_token', 'path': '{db_name}/token', 'func': _get_token, 'http_method': 'post',
                   'requires_authentication': False})

    result.append({'name': 'set_use_authentication', 'path': 'authentication/set_use',
                   'func': _set_use_authentication, 'http_method': 'get',
                   'requires_authentication': True})

    result.append({'name': 'get_use_authentication', 'path': 'authentication/get_use',
                   'func': _get_use_authentication, 'http_method': 'get',
                   'requires_authentication': False})

    result.append({'name': 'get_authentication_enabled', 'path': 'authentication/get_enabled',
                   'func': _get_authentication_enabled, 'http_method': 'get',
                   'requires_authentication': False})

    result.append({'name': 'users_create', 'path': '{db_name}/users/create',
                   'func': _create_user, 'http_method': 'post',
                   'requires_authentication': True})

    result.append({'name': 'users_delete', 'path': '{db_name}/users/delete',
                   'func': _delete_user, 'http_method': 'get',
                   'requires_authentication': True})

    result.append({'name': 'users_get', 'path': '{db_name}/users/get',
                   'func': _get_user, 'http_method': 'get',
                   'requires_authentication': True})

    result.append({'name': 'users_get_all', 'path': '{db_name}/users/get_all',
                   'func': _get_all_users, 'http_method': 'get',
                   'requires_authentication': True})

    result.append({'name': 'users_set_password', 'path': '{db_name}/users/set_password',
                   'func': _set_password, 'http_method': 'post',
                   'requires_authentication': True})

    result.append({'name': 'users_set_enabled', 'path': '{db_name}/users/set_enabled',
                   'func': _set_enabled, 'http_method': 'get',
                   'requires_authentication': True})

    result.append({'name': 'users_set_disabled', 'path': '{db_name}/users/set_disabled',
                   'func': _set_disabled, 'http_method': 'get',
                   'requires_authentication': True})

    return result
