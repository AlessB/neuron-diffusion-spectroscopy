"""
probe_text.py
=============

Build the fixed probe set once, save it, and reuse the exact same tokens at every checkpoint.

Why fixed: if the text changed between checkpoints, we could not tell a change in the model from a
change in the data. The token ids are saved in the repo (`probes/probe_tokens_*.npy`, about 200 KB),
so every machine and every agent uses byte-identical input.

Where the text comes from (DECISIONS.md D6)
-------------------------------------------
Pythia was trained on the Pile's TRAIN split. We take probe text from the Pile's VALIDATION split,
which Pythia never trained on. The original Pile host is gone; we use the public mirror
`monology/pile-uncopyrighted` on the Hugging Face Hub, file `val.jsonl.zst` (338 MB). That mirror
removed the subsets not cleared for AI training (Books3, BookCorpus2, OpenSubtitles, YTSubtitles,
OpenWebText2), so our probe text covers the remaining Pile domains. Each document carries its Pile
subset name (`meta.pile_set_name`), which we keep: it lets us ask later whether neuron modules line
up with text domains.

How sequences are made
----------------------
One sequence per document: tokenise the document, skip it if it is shorter than
`sequence_length` tokens, otherwise keep its first `sequence_length` tokens. No sequence crosses a
document boundary. Documents are read in file order, so the selection is deterministic.

Position 0 (proposed DECISIONS D17). Pythia never saw a document "from nothing": its training
stream is documents joined by the end-of-document token `<|endoftext|>` (id 0 in its tokenizer),
so every document it trained on starts right after that token. With `document_separator_id` set,
each row starts with that token followed by the first `sequence_length - 1` document tokens, which
matches what the model saw in training. Collection then leaves position 0 out of the neuron
statistics (`activations.collect_neuron_moments(skip_first_positions=1)`): the first position has
no context and in many LLMs carries unusually large activations (attention sinks / massive
activations, Sun et al. 2024). Gurnee et al. (2024) filter the same kind of tokens.

Credit
------
- L. Gao, S. Biderman, S. Black, et al. (2020). "The Pile: An 800GB dataset of diverse text for
  language modeling." arXiv:2101.00027.
- Mirror: https://huggingface.co/datasets/monology/pile-uncopyrighted
- M. Sun, X. Chen, J. Z. Kolter, Z. Liu (2024). "Massive activations in large language models."
  COLM 2024. https://github.com/locuslab/massive-activations (why position 0 is set aside)
- W. Gurnee, T. Horsley, Z. C. Guo, et al. (2024). "Universal neurons in GPT2 language models."
  TMLR. https://github.com/wesg52/universal-neurons (filters BOS, padding and newline tokens)
- The position-0 issue was found in this project's code study (research/S2_code_llm_geometry.md).
"""

import hashlib
import io
import json

import numpy as np


def iterate_pile_documents(local_path):
    """
    Yield (text, pile_set_name) from a Pile-style .jsonl.zst file, one document at a time,
    without unpacking the whole file.
    """
    import zstandard

    with open(local_path, "rb") as compressed_file:
        decompressor = zstandard.ZstdDecompressor()

        with decompressor.stream_reader(compressed_file) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")

            for line in text_stream:
                record = json.loads(line)

                text = record["text"]

                meta = record.get("meta", {})
                subset = meta.get("pile_set_name", "unknown")

                yield text, subset


def download_pile_validation(cache_dir=None):
    """
    Download val.jsonl.zst from the mirror (once; the Hub caches it) and return its local path.
    """
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        repo_id="monology/pile-uncopyrighted",
        filename="val.jsonl.zst",
        repo_type="dataset",
        cache_dir=cache_dir,
    )

    return local_path


def build_probe_set(
    tokenizer,
    documents,
    number_of_sequences=200,
    sequence_length=256,
    document_separator_id=None,
):
    """
    Turn an iterator of (text, subset) into a token array of shape
    (number_of_sequences, sequence_length) plus a manifest describing where each row came from.

    document_separator_id   None: rows are the document's first `sequence_length` tokens.
                            An id (Pythia: tokenizer.eos_token_id, which is 0): rows are that id
                            followed by the document's first `sequence_length - 1` tokens.
    """
    rows = []
    manifest_rows = []

    if document_separator_id is None:
        document_tokens_per_row = sequence_length
    else:
        document_tokens_per_row = sequence_length - 1

    for document_index, (text, subset) in enumerate(documents):
        token_ids = tokenizer(text)["input_ids"]

        if len(token_ids) < document_tokens_per_row:
            continue

        row = list(token_ids[:document_tokens_per_row])

        if document_separator_id is not None:
            row = [int(document_separator_id)] + row

        rows.append(row)

        manifest_rows.append(
            {
                "document_index": document_index,
                "pile_set_name": subset,
            }
        )

        if len(rows) == number_of_sequences:
            break

    if len(rows) < number_of_sequences:
        raise ValueError(
            f"only found {len(rows)} documents with at least {sequence_length} tokens"
        )

    tokens = np.array(rows, dtype=np.int32)

    return tokens, manifest_rows


def token_fingerprint(tokens):
    """
    SHA-256 of the token array, so any run can check it used the right probe set.
    """
    contiguous = np.ascontiguousarray(tokens, dtype=np.int32)

    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def save_probe_set(tokens, manifest_rows, tokens_path, manifest_path, source_description):
    """
    Save the tokens (.npy) and a JSON manifest with the fingerprint and the source.
    """
    np.save(tokens_path, tokens)

    manifest = {
        "source": source_description,
        "shape": list(tokens.shape),
        "sha256": token_fingerprint(tokens),
        "rows": manifest_rows,
    }

    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=1)


def load_probe_set(tokens_path, manifest_path):
    """
    Load the tokens and check them against the manifest's fingerprint.
    """
    tokens = np.load(tokens_path)

    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    if token_fingerprint(tokens) != manifest["sha256"]:
        raise ValueError(f"{tokens_path} does not match the fingerprint in {manifest_path}")

    return tokens, manifest
