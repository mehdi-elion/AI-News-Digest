from pydantic import BaseModel, Field, HttpUrl


class APIResult(BaseModel):  # noqa: D101
    url: HttpUrl
    title: str = Field(max_length=64, pattern=r"^[A-Z].*$")
    content: str
