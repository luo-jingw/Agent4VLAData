"""Push the staged release tree to the Hugging Face Hub.

The tree under `hf_export/` is uploaded verbatim, so whatever `export_hf.py` checked is what lands.
`upload_large_folder` is used rather than `upload_folder`: it shards the commit, runs the transfers
in parallel and resumes where it left off, which matters for ~86 GiB across three checkpoints.

The token is read from a file and never printed, logged, or passed on a command line where `ps`
would show it.

  cd tmp/tmp_train_G2
  ../../train/openpi/.venv/bin/python upload_hf.py --dry-run     # list what would go
  ../../train/openpi/.venv/bin/python upload_hf.py
"""

from __future__ import annotations

import dataclasses
import pathlib

import huggingface_hub
import tyro

ROOT = pathlib.Path(__file__).resolve().parent


@dataclasses.dataclass(frozen=True)
class Args:
    folder: pathlib.Path = ROOT / "hf_export"
    repo_id: str = "EmbodyX/G2-Orion-suction-cup-2subtasks"
    token_file: pathlib.Path = ROOT.parent / "HF_TOKEN"
    # The repo is expected to exist already, with its visibility set by whoever made it. Creating
    # it here is opt-in so that an upload can never be the thing that changes those settings.
    create: bool = False
    private: bool = True  # only consulted when --create is passed
    dry_run: bool = False


def main(args: Args) -> None:
    files = sorted(p for p in args.folder.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    print(f"{args.folder} -> {args.repo_id}")
    print(f"  {len(files)} files, {total / 2**30:.1f} GiB")
    for part in sorted({p.relative_to(args.folder).parts[0] for p in files}):
        sub = [p for p in files if p.relative_to(args.folder).parts[0] == part]
        print(f"    {part:22s} {len(sub):3d} files  {sum(p.stat().st_size for p in sub) / 2**30:7.2f} GiB")
    if args.dry_run:
        return

    token = args.token_file.read_text().strip()
    if not token:
        raise RuntimeError(f"{args.token_file} is empty")
    api = huggingface_hub.HfApi(token=token)
    if args.create:
        api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    else:
        info = api.repo_info(args.repo_id, repo_type="model")
        print(f"  target exists, private={info.private}")
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(args.folder),
        num_workers=8,
    )
    print(f"done: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main(tyro.cli(Args))
