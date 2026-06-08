from types import SimpleNamespace

import pytest

pytest.importorskip("huggingface_hub")
hub_errors = pytest.importorskip("huggingface_hub.errors")
httpx = pytest.importorskip("httpx")

HfHubHTTPError = hub_errors.HfHubHTTPError
from scripts.upload_hf_adapter import ensure_model_repo, upload_adapter


class DummyApi:
    def __init__(self, *, fail_create: bool = False) -> None:
        self.fail_create = fail_create
        self.model_info_called = False

    def create_repo(self, **kwargs):
        if self.fail_create:
            response = httpx.Response(
                403,
                request=httpx.Request("POST", "https://huggingface.co/api/repos/create"),
            )
            raise HfHubHTTPError("403 Forbidden", response=response)
        return "created"

    def model_info(self, repo_id):
        self.model_info_called = True
        return SimpleNamespace(id=repo_id)


def test_ensure_model_repo_accepts_existing_repo_when_create_is_forbidden():
    api = DummyApi(fail_create=True)

    ensure_model_repo(api, "micic-mihajlo/adapter", private=False)

    assert api.model_info_called is True


def test_upload_adapter_requires_token(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    args = SimpleNamespace(
        token=None,
        folder=tmp_path,
        repo_id="micic-mihajlo/adapter",
        private=False,
        commit_message="Upload adapter",
        create_pr=True,
    )

    with pytest.raises(ValueError, match="HF token"):
        upload_adapter(args)
