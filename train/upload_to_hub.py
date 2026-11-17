#!/usr/bin/env python3
"""Upload the prepared folder to your Hugging Face model repo and print the
block to paste into js/config.js.

    hf auth login                      # once; needs a token with write access
    python upload_to_hub.py --folder out/hf-repo --repo your-name/your-model

The repo must be public: the browser downloads the weights without a token.

Needs: pip install huggingface_hub
"""

import argparse
import pathlib
import sys

from huggingface_hub import HfApi


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", default="out/hf-repo", help="what prepare_repo.py wrote")
    p.add_argument("--repo", required=True, help="your-name/your-model")
    p.add_argument("--dtype", default="q4f16")
    p.add_argument("--private", action="store_true", help="create the repo private (BLIK cannot load it until you make it public)")
    args = p.parse_args()

    folder = pathlib.Path(args.folder)
    graph = folder / "onnx" / f"model_{args.dtype}.onnx"
    for needed in (folder / "config.json", folder / "tokenizer.json", folder / "LICENSE", graph):
        if not needed.exists():
            sys.exit(f"{needed} is missing. Run prepare_repo.py first.")

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    commit = api.upload_folder(folder_path=str(folder), repo_id=args.repo, repo_type="model", commit_message="Upload ONNX model for BLIK llm")
    revision = commit.oid
    print(f"\nUploaded to https://huggingface.co/{args.repo} at commit {revision}\n")
    print("Put this in js/config.js:\n")
    print("export const MODEL = {")
    print(f"  repo: '{args.repo}',")
    print(f"  revision: '{revision}',")
    print(f"  dtype: '{args.dtype}',")
    print("};")
    if args.private:
        print("\nThe repo is private. Make it public before loading it in BLIK.")


if __name__ == "__main__":
    main()
