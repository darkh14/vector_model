
from pydantic import BaseModel


class User(BaseModel):
    name: str
    disabled: bool


class InputUser(BaseModel):
    username: str
    password: str


class OutputToken(BaseModel):
    access_token: str
    token_type: str
