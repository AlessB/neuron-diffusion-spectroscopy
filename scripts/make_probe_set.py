"""
make_probe_set.py
=================

Build the fixed probe text once (run on a machine with internet; ~340 MB download, one time):

    python scripts/make_probe_set.py --sequences 1024 --length 256

Writes probes/probe_tokens_pile-val_<N>x<T>.npy and the matching .json manifest. Commit both:
they are small (about 1 MB for 1,024 x 256) and make every later run use byte-identical input.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nds.probe_text import build_probe_set
from nds.probe_text import download_pile_validation
from nds.probe_text import iterate_pile_documents
from nds.probe_text import save_probe_set


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequences", type=int, default=1024)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--tokenizer", default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--no-separator",
        action="store_true",
        help="do not start rows with <|endoftext|> (proposed D17 says: do start with it)",
    )
    arguments = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(arguments.tokenizer)

    if arguments.no_separator:
        separator_id = None
    else:
        separator_id = tokenizer.eos_token_id
        print(f"rows start with {tokenizer.eos_token!r} (id {separator_id})")

    local_path = download_pile_validation(cache_dir=arguments.cache_dir)
    print(f"Pile validation file: {local_path}")

    documents = iterate_pile_documents(local_path)

    tokens, manifest_rows = build_probe_set(
        tokenizer,
        documents,
        number_of_sequences=arguments.sequences,
        sequence_length=arguments.length,
        document_separator_id=separator_id,
    )

    output_folder = Path("probes")
    output_folder.mkdir(exist_ok=True)

    stem = f"probe_tokens_pile-val_{arguments.sequences}x{arguments.length}"

    # the position-0 control (PLAN P2.3) must not overwrite the main probe set
    if arguments.no_separator:
        stem = stem + "_no-separator"

    save_probe_set(
        tokens,
        manifest_rows,
        output_folder / f"{stem}.npy",
        output_folder / f"{stem}.json",
        source_description=(
            "monology/pile-uncopyrighted val.jsonl.zst, first documents with enough "
            f"tokens for rows of {arguments.length}, tokenizer {arguments.tokenizer}, "
            f"row separator id {separator_id}"
        ),
    )

    subsets = {}
    for row in manifest_rows:
        subsets[row["pile_set_name"]] = subsets.get(row["pile_set_name"], 0) + 1

    print(f"saved {tokens.shape} to probes/{stem}.npy")
    print("documents per Pile subset:", subsets)


if __name__ == "__main__":
    main()
