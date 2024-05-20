from contextlib import nullcontext

import pytest
from pydantic import AnyUrl, ValidationError

from src.ai_news_digest.models.core import APIResult


def test_api_result():  # noqa: D103
    api_result = APIResult(url="http://example.com/", title="My title", content="This is an example content.")

    assert api_result.url == AnyUrl("http://example.com/")
    assert api_result.title == "My title"
    assert api_result.content == "This is an example content."


@pytest.mark.parametrize(
    "url,expectation",
    [
        pytest.param("foo bar", pytest.raises(ValidationError), id="it should be an url"),
        pytest.param(
            "http://example.com/",
            nullcontext(APIResult(url="http://example.com", title="My title", content="This is an example content")),
            id="http with TLD and Host should pass",
        ),
        pytest.param(
            "https://example.com/",
            nullcontext(APIResult(url="https://example.com", title="My title", content="This is an example content")),
            id="https with TLD and Host should pass",
        ),
    ],
)
def test_api_result_validation_url(url, expectation):  # noqa: D103
    with expectation as e:
        assert APIResult(url=url, title="My title", content="This is an example content") == e


@pytest.mark.parametrize(
    "title,expectation",
    [
        pytest.param("foo", pytest.raises(ValidationError), id="it should start with an uppercase letter"),
        pytest.param("T" * 65, pytest.raises(ValidationError), id="it should be max 64 characters long"),
        pytest.param(
            "Foo",
            nullcontext(APIResult(url="http://example.com", title="Foo", content="This is an example content")),
            id="title with correct length that starts with upper should pass",
        ),
    ],
)
def test_api_result_validation_title(title, expectation):  # noqa: D103
    with expectation as e:
        assert APIResult(url="http://example.com/", title=title, content="This is an example content") == e
