""" Module for controlling users, passwords and tokens
"""
import os
from typing import Any, Optional
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta

from db_processing import get_connector, get_connector_by_name, SETTINGS_DB_NAME
from vm_settings import get_var, set_var
from vm_settings import defaults as settings_defaults
import vm_logging.exceptions as exceptions


TOKEN_ALGORITHM = 'HS256'

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
AUTHENTICATION_ENABLED = None


class UsersController:
    """
        Class for managing users.
        Class properties:
            _db_connector:  connector object to connect to db
        Class methods:
            get_user:  returns user dict (keys: username, disabled, hashed_password (Optional))
    """

    def __init__(self):
        self._db_connector = get_connector()

    def get_user(self, username: str, include_password: bool = False) -> Optional[dict[str, Any]]:
        """
        Returns user dict (keys: username, disabled, hashed_password (Optional))
        :param username: name of required user
        :param include_password: includes hashed password if True
        :return: user dict
        """
        if username == get_var('ROOT_USER'):
            user = AuthenticationController().get_root_user(include_password)
        else:
            user = self._db_connector.get_line('users', {'name': username})

        if user:
            if not include_password and user.get('hashed_password'):
                del user['hashed_password']

        return user

    @staticmethod
    def generate_token(username: str) -> str:
        access_token_expire_minutes = get_var('ACCESS_TOKEN_EXPIRE_MINUTES')
        access_token_expires = timedelta(minutes=access_token_expire_minutes)

        jwt_secret = get_var('JWT_SECRET')

        token_data = {'sub': username, 'exp': datetime.utcnow() + access_token_expires}

        if username != get_var('ROOT_USER'):
            token_data['db'] = get_connector().db_name

        token = jwt.encode(token_data, jwt_secret, algorithm=TOKEN_ALGORITHM)

        return token

    def get_user_by_token(self, token: str) -> dict[str, Any]:

        try:
            payload = jwt.decode(token, get_var('JWT_SECRET'), algorithms=['HS256'])
            username = payload.get('sub')

            if username != get_var('ROOT_USER'):
                connector = get_connector()
                if connector:
                    db_name = get_connector().db_name
                    if payload.get('db') != db_name:
                        raise exceptions.CredentialsException('Token is not allowed in db "{}"'.format(db_name))
                else:
                    raise exceptions.CredentialsException('DB is not defined for token. Use correct DB or system user')

            if username is None:
                raise exceptions.CredentialsException("Could not validate credentials")
        except JWTError as exp:
            raise exceptions.TokenException(str(exp))

        return self.get_user(payload['sub'])

    def validate_password(self, username: str, hashed_password: str) -> bool:
        user = self.get_user(username, include_password=True)

        if not user:
            return False

        if user['disabled']:
            return False

        return self._verify_password(hashed_password, user['hashed_password'])

    def create_user(self, username: str, password: str) -> None:

        existing_user = self.get_user(username)
        if existing_user:
            raise exceptions.UserAlreadyExistsException('User {} already exists'.format(username))

        hashed_password = self._get_hash(password)

        self._db_connector.set_line('users', {'name': username,
                                             'hashed_password': hashed_password, 'disabled': False}, {'name': username})

    def delete_user(self, username) -> None:
        existing_user = self.get_user(username)
        if not existing_user:
            raise exceptions.UserNotFoundException('User {} is not found'.format(username))

        self._db_connector.delete_lines('users', {'name': username})

    def set_password(self, username: str, password: str) -> None:
        existing_user = self.get_user(username)
        if not existing_user:
            raise exceptions.UserNotFoundException('User {} is not found'.format(username))

        existing_user['hashed_password'] = self._get_hash(password)

        self._db_connector.set_line('users', existing_user, {'name': username})

    def set_enabled(self, username: str) -> None:
        self._set_user_status(username, disabled=False)

    def set_disabled(self, username: str) -> None:
        self._set_user_status(username, disabled=True)

    def get_all_users(self, include_disabled: bool = False) -> list[dict[str, Any]]:

        users_filter = {'disabled': False} if not include_disabled else None

        users = self._db_connector.get_lines('users', users_filter)

        for user in users:
            del user['hashed_password']

        return users

    @staticmethod
    def _get_hash(input_string: str) -> str:
        return _get_hash(input_string)

    def _set_user_status(self, username: str, disabled: bool = True):
        existing_user = self.get_user(username, include_password=True)
        if not existing_user:
            raise exceptions.UserNotFoundException('User {} is not found'.format(username))

        existing_user['disabled'] = disabled

        self._db_connector.set_line('users', existing_user, {'name': username})

    @staticmethod
    def _verify_password(plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)


class AuthenticationController:
    """
        Class for controlling users.
        Class properties:
            _db_connector:  connector object to connect to db
    """

    def __init__(self):
        global AUTHENTICATION_ENABLED
        self._db_connector = get_connector_by_name(SETTINGS_DB_NAME)
        if AUTHENTICATION_ENABLED is None:
            AUTHENTICATION_ENABLED = self.get_use()
            if not self.get_root_user():
                self.create_root_user()

    def set_use(self, use: bool) -> None:
        set_var('USE_AUTHENTICATION', use)
        self._db_connector.set_line('settings', {'name': 'use_authentication', 'value': use},
                                    {'name': 'use_authentication'})

    def get_use(self) -> bool:
        use_line = self._db_connector.get_line('settings', {'name': 'use_authentication'})
        if use_line is None:
            use = get_var('USE_AUTHENTICATION')
        else:
            use = use_line['value']

        return use

    # noinspection PyMethodMayBeStatic
    def get_enabled(self) -> bool:
        return bool(AUTHENTICATION_ENABLED)

    def create_root_user(self):
        user_name = get_var('ROOT_USER')
        password = os.environ.get('ROOT_PASSWORD') or settings_defaults.ROOT_PASSWORD

        self._db_connector.set_line('users', {'name': user_name,
                                              'hashed_password': _get_hash(password)},
                                    {'name': user_name})

    def get_root_user(self, include_password: bool = False) -> Optional[dict[str, Any]]:
        root_user = self._db_connector.get_line('users', {'name': get_var('ROOT_USER')})

        if root_user:
            root_user['disabled'] = False
            if not include_password:
                del root_user['hashed_password']
        return root_user


def _get_hash(input_string: str) -> str:
    return pwd_context.hash(input_string)


def get_current_user(token: str) -> dict[str, Any]:
    users = UsersController()
    return users.get_user_by_token(token)


def set_use_authentication(use: bool = False) -> None:

    auth = AuthenticationController()

    auth.set_use(use)


def get_use_authentication() -> bool:
    auth = AuthenticationController()
    return auth.get_use()


def get_authentication_enabled() -> bool:
    auth = AuthenticationController()
    return auth.get_enabled()


def create_user(username, password) -> None:
    users = UsersController()
    users.create_user(username, password)


def get_user(username) -> dict[str, Any]:
    users = UsersController()
    result = users.get_user(username)
    return result


def get_all_users() -> list[dict[str, Any]]:
    users = UsersController()
    result = users.get_all_users(include_disabled=True)
    return result


def set_user_enabled(username: str) -> None:
    users = UsersController()
    users.set_enabled(username)


def set_user_disabled(username: str) -> None:
    users = UsersController()
    users.set_disabled(username)


def set_user_password(username: str, password: str) -> None:
    users = UsersController()
    users.set_password(username, password)


def delete_user(username: str) -> None:
    users = UsersController()
    users.delete_user(username)
