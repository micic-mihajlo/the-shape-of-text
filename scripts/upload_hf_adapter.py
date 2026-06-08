#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError


def ensure_model_repo(api: HfApi, repo_id: str, *, private: bool) -> None:
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        return
    except HfHubHTTPError as exc:
        message = str(exc)
        if "403 Forbidden" not in message:
            raise
        # Some Hugging Face Jobs tokens can open PRs but cannot create repos or
        # commit directly to main. If the repo already exists, keep going and
        # let upload_folder(create_pr=True) handle the write path.
        api.model_info(repo_id)


def upload_adapter(args: argparse.Namespace):
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF token is required via --token or HF_TOKEN")
    if not args.folder.exists():
        raise FileNotFoundError(args.folder)

    api = HfApi(token=token)
    ensure_model_repo(api, args.repo_id, private=args.private)
    return api.upload_folder(
        folder_path=str(args.folder),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=args.commit_message,
        create_pr=args.create_pr,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a LoRA adapter folder to HF Hub")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--token", default=None)
    parser.add_argument("--commit-message", default="Upload trained social-post LoRA adapter")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--create-pr", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = upload_adapter(parse_args())
    print(f"upload_result={result}")
    commit_url = getattr(result, "commit_url", None)
    pr_url = getattr(result, "pr_url", None)
    if commit_url:
        print(f"commit_url={commit_url}")
    if pr_url:
        print(f"pr_url={pr_url}")
    print("HF_ADAPTER_UPLOAD_OK")


if __name__ == "__main__":
    main()
