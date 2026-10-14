"""Load and clean the Stanford IMDB movie review sentiment data, and freeze
one split.

Source: the HuggingFace dataset `stanfordnlp/imdb`
(https://huggingface.co/datasets/stanfordnlp/imdb). It is public, but
`load_dataset()` still accepts the project's HF token for consistency with
the (gated) model checkpoint - same credential, one source of truth.

HF split "train" (25k, labeled) -> the POOL. Split once into train/val/test.
HF split "test"  (25k, labeled) -> the HOLDOUT. Never split, never trained on.
("unsupervised", 50k unlabeled reviews, is not used anywhere in this project.)

`load_clean()`   - the HF pool split -> a canonical, deduplicated 2-class
                   DataFrame (`positive` / `negative`).
`load_holdout()` - same cleaning applied to the HF test split, minus any row
                   that also appears in the pool (that overlap would be leakage).
`build_split()`  - split the pool into train/val/test once and write CSVs.
                   Later runs just reuse the files already on disk.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import get_hf_token, get_logger, read_json, resolve, write_json

logger = get_logger(__name__)

SPLIT_FILES = {"train": "train.csv", "val": "val.csv", "test": "test.csv"}
MANIFEST_FILE = "split_manifest.json"
POOL_FILE = "clean_pool.csv"
HOLDOUT_FILE = "holdout.csv"

# stanfordnlp/imdb's ClassLabel feature order: 0 = neg, 1 = pos.
_LABEL_NAMES = {0: "negative", 1: "positive"}


# --------------------------------------------------------------------------
# 1. Load + clean
# --------------------------------------------------------------------------
def _load_hf_split(dataset_id: str, split: str, hf_token: str | None) -> pd.DataFrame:
    """Pull one split of the dataset from the Hub as a plain DataFrame."""
    from datasets import load_dataset

    token = hf_token or get_hf_token()
    ds = load_dataset(dataset_id, split=split, token=token)
    return ds.to_pandas()


def _read_and_clean(
    dcfg: dict[str, Any], split: str, tag: str, hf_token: str | None
) -> pd.DataFrame:
    """Shared cleaning for both HF splits: normalise labels, strip HTML line
    breaks, drop bad rows, dedupe. Returns columns: uid, text, label."""
    df = _load_hf_split(dcfg["hf_dataset"], split, hf_token)
    n_raw = len(df)

    # newer pandas leaves NaN as NaN under plain .astype(str), so fill first
    def _as_text(col: pd.Series) -> pd.Series:
        return col.astype(object).fillna("").astype(str).str.strip()

    # The raw reviews keep literal "<br /><br />" line breaks from the source
    # HTML; collapse them (and any other run of whitespace) to single spaces.
    text = _as_text(df["text"]).str.replace("<br />", " ", regex=False)
    df["text"] = text.str.replace(r"\s+", " ", regex=True).str.strip()

    # label is an int ClassLabel (0/1); -1 in the "unsupervised" split means
    # "no gold label" and must be dropped, not silently mapped to something.
    df["label"] = df["label"].map(_LABEL_NAMES)
    n_dropped_label = int(df["label"].isna().sum())
    df = df[df["label"].notna()]

    # a literal empty cell reads back as the string "nan" after astype(str)
    too_short = df["text"].isin(["", "nan", "none"]) | (
        df["text"].str.len() < int(dcfg["min_chars"])
    )
    n_dropped_text = int(too_short.sum())
    df = df[~too_short]

    n_before_dedupe = len(df)
    df = df.drop_duplicates(subset=list(dcfg["dedupe_on"]), keep="first")
    n_dropped_dupe = n_before_dedupe - len(df)

    # stable row id: content hash, so uid survives a row-order change
    df = df.reset_index(drop=True)
    df["uid"] = [hashlib.sha1(t.encode("utf-8")).hexdigest()[:16] for t in df["text"]]
    df = df[["uid", "text", "label"]].reset_index(drop=True)

    logger.info(
        "[%s] %d raw -> %d kept (dropped %d unlabeled, %d short, %d dup) | %s",
        tag, n_raw, len(df), n_dropped_label, n_dropped_text, n_dropped_dupe,
        df["label"].value_counts().to_dict(),
    )
    return df


def load_clean(cfg: dict[str, Any], hf_token: str | None = None) -> pd.DataFrame:
    """Clean the HF `data.pool_split` (default: "train") into the canonical pool."""
    dcfg = cfg["data"]
    return _read_and_clean(dcfg, dcfg["pool_split"], tag="pool", hf_token=hf_token)


def load_holdout(
    cfg: dict[str, Any], pool: pd.DataFrame | None = None, hf_token: str | None = None
) -> pd.DataFrame | None:
    """Clean the HF `data.holdout_split` (default: "test") into an independent set.

    Returns None when no split is configured. Rows also present in the pool
    are dropped so the holdout stays independent.
    """
    dcfg = cfg["data"]
    holdout_split = dcfg.get("holdout_split")
    if not holdout_split:
        return None

    df = _read_and_clean(dcfg, holdout_split, tag="holdout", hf_token=hf_token)

    if pool is not None:
        overlap = df["uid"].isin(set(pool["uid"]))
        n_overlap = int(overlap.sum())
        if n_overlap:
            logger.warning(
                "[holdout] %d row(s) also occur in the pool - removed to keep the "
                "holdout independent.", n_overlap,
            )
            df = df[~overlap].reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# 2. Split once, then freeze
# --------------------------------------------------------------------------
def split_exists(cfg: dict[str, Any]) -> bool:
    """True once every split file (+ pool) is already on disk."""
    out_dir = resolve(cfg["split"]["out_dir"])
    return all((out_dir / f).exists() for f in (*SPLIT_FILES.values(), POOL_FILE))


def _freeze_holdout(out_dir, holdout: pd.DataFrame | None) -> dict[str, Any]:
    """Write holdout.csv and return its stats block for the manifest."""
    if holdout is None or holdout.empty:
        return {"present": False, "n_rows": 0}
    holdout.reset_index(drop=True).to_csv(out_dir / HOLDOUT_FILE, index=False)
    return {
        "present": True,
        "n_rows": len(holdout),
        "file": HOLDOUT_FILE,
        "label_counts": holdout["label"].value_counts().to_dict(),
        "note": "From the HF `test` split of stanfordnlp/imdb. Never split, "
                "never trained on, never used for model selection - an "
                "independent cross-check only.",
    }


def build_split(
    cfg: dict[str, Any],
    df: pd.DataFrame,
    holdout: pd.DataFrame | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Create the frozen train/val/test split once, or reuse it if it already
    exists. Pass force=True to discard an existing split and redo it."""
    scfg = cfg["split"]
    out_dir = resolve(scfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if split_exists(cfg) and not force:
        logger.info("Split already exists in %s - reusing.", out_dir)
        manifest = read_json(out_dir / MANIFEST_FILE)
        # holdout is refreshed on every run since it's never part of the split
        manifest["holdout"] = _freeze_holdout(out_dir, holdout)
        write_json(manifest, out_dir / MANIFEST_FILE)
        return manifest

    ratios = (scfg["train_ratio"], scfg["val_ratio"], scfg["test_ratio"])
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(ratios)}")

    strat_col = scfg["stratify_on"]
    seed = cfg["seed"]

    # carve off train, then split the rest into val/test at the right ratio
    train_df, rest_df = train_test_split(
        df, train_size=ratios[0], random_state=seed, shuffle=True,
        stratify=df[strat_col],
    )
    val_share = ratios[1] / (ratios[1] + ratios[2])
    val_df, test_df = train_test_split(
        rest_df, train_size=val_share, random_state=seed, shuffle=True,
        stratify=rest_df[strat_col],
    )

    for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
        part.reset_index(drop=True).to_csv(out_dir / SPLIT_FILES[name], index=False)
    df.reset_index(drop=True).to_csv(out_dir / POOL_FILE, index=False)

    manifest = {
        "seed": seed,
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "stratify_on": strat_col,
        "pool_size": len(df),
        "sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "label_counts": {
            "pool": df[strat_col].value_counts().to_dict(),
            "train": train_df[strat_col].value_counts().to_dict(),
            "val": val_df[strat_col].value_counts().to_dict(),
            "test": test_df[strat_col].value_counts().to_dict(),
        },
        "files": {**SPLIT_FILES, "pool": POOL_FILE},
        "holdout": _freeze_holdout(out_dir, holdout),
        "note": "Created once from the HF `train` split of stanfordnlp/imdb. "
                "Task 1 uses the full pool; the splits are for the fine-tuning "
                "tasks so every task shares identical partitions. The holdout "
                "comes from the HF `test` split and is never part of the split.",
    }
    write_json(manifest, out_dir / MANIFEST_FILE)
    logger.info("Split created: %s | holdout: %d rows",
                manifest["sizes"], manifest["holdout"]["n_rows"])
    return manifest


# Every partition name accepted by `load_split` / `--source`.
PARTITIONS = ("all", "train", "val", "test", "holdout")


def load_split(cfg: dict[str, Any], which: str) -> pd.DataFrame:
    """Load one frozen partition. 'all' = the whole cleaned pool, 'holdout' =
    the independent HF `test` split, otherwise one of train/val/test."""
    out_dir = resolve(cfg["split"]["out_dir"])
    fname = {"all": POOL_FILE, "holdout": HOLDOUT_FILE}.get(which) or SPLIT_FILES.get(which)
    if fname is None:
        raise ValueError(f"Unknown partition '{which}'. Use one of {'|'.join(PARTITIONS)}.")

    path = out_dir / fname
    if not path.exists():
        if which == "holdout":
            raise FileNotFoundError(
                f"{path} not found. Re-run `python scripts/01_prepare_data.py` "
                "with `data.holdout_split` set in the config to enable "
                "`--source holdout`."
            )
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/01_prepare_data.py` first."
        )
    return pd.read_csv(path)


def sample_for_eval(
    df: pd.DataFrame, n: int | None, seed: int, stratified: bool = True
) -> pd.DataFrame:
    """Take the evaluation subset for Task 1. `n=None` (or n >= len(df)) means
    use everything. Stratified sampling keeps the pool's label proportions."""
    if n is None or n >= len(df):
        return df.reset_index(drop=True)
    if n < 200:
        logger.warning("n_samples=%d is below the required minimum of 200.", n)

    if stratified:
        subset, _ = train_test_split(
            df, train_size=n, random_state=seed, shuffle=True, stratify=df["label"]
        )
    else:
        subset = df.sample(n=n, random_state=seed)
    return subset.reset_index(drop=True)
