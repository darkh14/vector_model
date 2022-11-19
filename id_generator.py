""" Module for generating different variants of id
    classes:
        IdGenerator - main class for generation ids
"""

import uuid
import hashlib


class IdGenerator:
    """ Class for generation ids
        methods:
            get_random_id - for random id
            get_id_by_name - for id from string
            get_id_from_dict - for id from dict
    """
    @classmethod
    def get_random_id(cls) -> str:
        """Returns random id string (UUID) in xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx where x - hex digit"""
        return str(uuid.uuid4())

    @classmethod
    def get_id_by_name(cls, name: str) -> str:
        """Returns id string (UUID) generates by name in xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx where x - hex digit"""
        return str(uuid.uuid3(uuid.NAMESPACE_DNS, name))

    @classmethod
    def get_short_id_from_dict_id_type(cls, dict_value: dict[str, str]) -> str:
        """Returns short id string generates by dict in xxxxxxx where x - hex digit"""
        str_val = dict_value['id'] + dict_value.get('type') or ''
        return cls._get_hash(str_val)

    @classmethod
    def get_short_id_from_list_of_dict_short_id(cls, list_value: list[dict[str, str]]) -> str:
        """Returns short id string generates by dict in xxxxxxx where x - hex digit"""
        if list_value:
            short_id_list = [el['short_id'] for el in list_value]
            short_id_list.sort()
            str_val = ''.join(short_id_list)
            return cls._get_hash(str_val)
        else:
            return ''

    @classmethod
    def _get_hash(cls, value: str) -> str:
        if not value.replace(' ', ''):
            return ''
        data_hash = hashlib.md5(value.encode())
        return data_hash.hexdigest()[-7:]
