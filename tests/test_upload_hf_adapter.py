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
        self.create_token = None
        self.model_info_token = None

    def create_repo(self, **kwargs):
        self.create_token = kwargs.get("token")
        if self.fail_create:
            response = httpx.Response(
                403,
                request=httpx.Request("POST", "https://huggingface.co/api/repos/create"),
            )
            raise HfHubHTTPError("403 Forbidden", response=response)
        return "created"

    def model_info(self, repo_id, **kwargs):
        self.model_info_called = True
        self.model_info_token = kwargs.get("token")
        return SimpleNamespace(id=repo_id)


class UploadDummyApi(DummyApi):
    last_instance = None

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.init_token = kwargs.get("token")
        self.upload_token = None
        UploadDummyApi.last_instance = self

    def upload_folder(self, **kwargs):
        self.upload_token = kwargs.get("token")
        return SimpleNamespace(commit_url="https://huggingface.co/commit")


def test_ensure_model_repo_accepts_existing_repo_when_create_is_forbidden():
    api = DummyApi(fail_create=True)

    ensure_model_repo(api, "micic-mihajlo/adapter", private=False, token="hf_test")

    assert api.model_info_called is True
    assert api.create_token == "hf_test"
    assert api.model_info_token == "hf_test"


def test_upload_adapter_passes_token_to_upload_folder(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.upload_hf_adapter.HfApi", UploadDummyApi)
    args = SimpleNamespace(
        token="hf_test",
        folder=tmp_path,
        repo_id="micic-mihajlo/adapter",
        private=False,
        commit_message="Upload adapter",
        create_pr=True,
    )

    upload_adapter(args)

    api = UploadDummyApi.last_instance
    assert api.init_token == "hf_test"
    assert api.create_token == "hf_test"
    assert api.upload_token == "hf_test"


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
