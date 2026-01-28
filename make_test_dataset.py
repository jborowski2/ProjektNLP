from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from data_loading import _read_csv_with_fallback
from relation_extractor import RelationExtractor


def _safe_write(path: Path, df: pd.DataFrame, *, sep: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path, index=False, sep=sep, encoding="utf-8")
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_alt{path.suffix}")
        df.to_csv(alt, index=False, sep=sep, encoding="utf-8")
        return alt


def _strip_leading_preps(text: str, *, preps: set[str]) -> str:
    s = (text or "").strip()
    if not s:
        return s
    parts = s.split()
    if len(parts) >= 2 and parts[0].lower() in preps:
        return " ".join(parts[1:]).strip()
    return s


def _single_token_lemma(rel: RelationExtractor, text: str) -> str:
    s = (text or "").strip()
    if not s:
        return s
    doc = rel.nlp(s)
    toks = [t for t in doc if (not t.is_space) and t.pos_ != "PUNCT"]
    if len(toks) != 1:
        return s
    t = toks[0]
    lemma = (t.lemma_ or t.text).strip()
    if not lemma:
        return s
    if s.isupper() and len(s) <= 6:
        return s
    if s[:1].isupper():
        return lemma[:1].upper() + lemma[1:]
    return lemma


def _normalize_like_tagged(
    rel: RelationExtractor,
    text: str | None,
    *,
    preps: set[str],
    lemma_single_token: bool,
) -> str:
    s = _strip_leading_preps(str(text or ""), preps=preps)
    if lemma_single_token:
        s = _single_token_lemma(rel, s)
    return s.strip()


def _complexity_features(doc) -> dict[str, int]:
    n_tokens = 0
    n_punct = 0
    n_commas = 0
    n_verbs = 0
    n_conj = 0
    n_subclauses = 0

    sub_deps = {"advcl", "ccomp", "xcomp", "acl", "relcl"}
    for t in doc:
        if t.is_space:
            continue
        if t.pos_ == "PUNCT":
            n_punct += 1
            if t.text == ",":
                n_commas += 1
            continue

        n_tokens += 1
        if t.pos_ in {"VERB", "AUX"}:
            n_verbs += 1
        if t.pos_ in {"CCONJ", "SCONJ"} or t.dep_ in {"cc", "mark"}:
            n_conj += 1
        if t.dep_ in sub_deps:
            n_subclauses += 1

    return {
        "n_tokens": n_tokens,
        "n_punct": n_punct,
        "n_commas": n_commas,
        "n_verbs": n_verbs,
        "n_conj": n_conj,
        "n_subclauses": n_subclauses,
    }


def _complexity_score(feat: dict[str, int]) -> int:
    # Lower = simpler.
    return (
        feat["n_tokens"]
        + 2 * feat["n_verbs"]
        + 3 * feat["n_subclauses"]
        + 2 * feat["n_commas"]
        + 1 * feat["n_conj"]
        + max(0, feat["n_punct"] - feat["n_commas"])
    )


def _has_nonempty(text: str | None) -> bool:
    return bool((text or "").strip())


def _greedy_select_simple_rich(
    df: pd.DataFrame,
    *,
    n: int,
    min_trigger_rate: float,
    min_where_rate: float,
) -> pd.DataFrame:
    if "complexity_score" not in df.columns:
        raise ValueError("Expected complexity_score in dataframe")
    if "has_trigger" not in df.columns or "has_where" not in df.columns:
        raise ValueError("Expected has_trigger/has_where in dataframe")

    req_trigger = int((n * min_trigger_rate) + 0.999999)
    req_where = int((n * min_where_rate) + 0.999999)

    df_sorted = df.sort_values(["complexity_score", "id"], ascending=[True, True]).reset_index(drop=True)

    selected_idx: list[int] = []
    used = set()
    trigger_count = 0
    where_count = 0

    # Pass 1: satisfy quotas with minimal complexity.
    for i, row in df_sorted.iterrows():
        if len(selected_idx) >= n:
            break
        has_trigger = bool(row["has_trigger"])
        has_where = bool(row["has_where"])
        need_trigger = trigger_count < req_trigger
        need_where = where_count < req_where
        helps = (need_trigger and has_trigger) or (need_where and has_where)
        if not helps:
            continue
        selected_idx.append(i)
        used.add(i)
        trigger_count += 1 if has_trigger else 0
        where_count += 1 if has_where else 0
        if trigger_count >= req_trigger and where_count >= req_where:
            break

    # Pass 2: fill remaining with simplest remaining.
    for i, row in df_sorted.iterrows():
        if len(selected_idx) >= n:
            break
        if i in used:
            continue
        selected_idx.append(i)
        used.add(i)

    chosen = df_sorted.iloc[selected_idx].reset_index(drop=True)
    chosen["quota_req_trigger"] = req_trigger
    chosen["quota_req_where"] = req_where
    chosen["quota_met_trigger"] = int(chosen["has_trigger"].sum())
    chosen["quota_met_where"] = int(chosen["has_where"].sum())
    return chosen


def _svo_features(doc, trigger_token) -> dict[str, bool]:
    if trigger_token is None:
        return {"has_subj": False, "has_obj": False, "has_svo": False}
    has_subj = False
    has_obj = False
    for t in doc:
        if t.head != trigger_token:
            continue
        if t.dep_ in {"nsubj", "nsubj:pass", "csubj"}:
            has_subj = True
        if t.dep_ in {"obj", "iobj", "obl", "ccomp", "xcomp"}:
            has_obj = True
    has_svo = bool(has_subj and has_obj)
    return {"has_subj": has_subj, "has_obj": has_obj, "has_svo": has_svo}


def _ease_score(row: dict[str, Any]) -> int:
    # Higher = easier to answer KTO/CO/TRIGGER (plus GDZIE/KIEDY).
    score = 0
    score += 8 if row.get("has_trigger") else 0
    score += 6 if row.get("has_who") else 0
    score += 6 if row.get("has_what") else 0
    score += 3 if row.get("has_svo") else 0
    score += 2 if row.get("has_where") else 0
    score += 1 if row.get("has_when") else 0

    # Penalties for complexity.
    score -= int(row.get("n_commas", 0))
    score -= 2 * int(row.get("n_subclauses", 0))
    score -= 1 * int(row.get("n_conj", 0))
    score -= max(0, int(row.get("n_tokens", 0)) - 14)
    return int(score)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample 100 headlines and tag them from scratch (rule-based).")
    parser.add_argument("--headlines", default="datasets/id_and_headline_first_sentence (1).csv")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--strategy",
        default="random",
        choices=["random", "simple", "simple-rich", "easy-relations"],
        help="How to select N headlines: random sample or syntactically simplest.",
    )

    parser.add_argument("--min-trigger-rate", type=float, default=0.60)
    parser.add_argument("--min-where-rate", type=float, default=0.60)
    parser.add_argument(
        "--easy-min-core-rate",
        type=float,
        default=0.85,
        help="For --strategy easy-relations: minimum fraction of selected rows that must have (KTO+CO+TRIGGER).",
    )

    parser.add_argument(
        "--selection-report-out",
        default=None,
        help="Optional CSV to write selection scores/features for the chosen rows.",
    )

    parser.add_argument(
        "--prefilter-k",
        type=int,
        default=5000,
        help=(
            "For strategies that require parsing many sentences (simple/simple-rich/easy-relations), "
            "first keep only K cheapest-by-string-complexity candidates to speed up selection. "
            "Set to 0 to disable."
        ),
    )

    # Requested output
    parser.add_argument("--out", default="test_data_set.csv", help="Output CSV (comma-separated) with sentence + tags.")

    # Helper outputs to make evaluation easier
    parser.add_argument("--tagged-out", default="results/test_data_set_tagged.csv")
    parser.add_argument("--headlines-out", default="results/test_data_set_headlines.csv")
    args = parser.parse_args()

    headlines = _read_csv_with_fallback(args.headlines, sep=",")
    if "id" not in headlines.columns or "headline" not in headlines.columns:
        raise ValueError(f"Expected columns id,headline in {args.headlines}, got {list(headlines.columns)}")

    headlines = headlines[["id", "headline"]].copy()
    headlines["id"] = pd.to_numeric(headlines["id"], errors="coerce")
    headlines = headlines.dropna(subset=["id", "headline"]).copy()
    headlines["headline"] = headlines["headline"].astype(str).str.strip()
    headlines = headlines[headlines["headline"] != ""].copy()

    n = int(args.n)
    if n <= 0:
        raise ValueError("--n must be > 0")

    if n > len(headlines):
        n = len(headlines)

    # Fast prefilter to avoid parsing the whole corpus for selection strategies.
    if args.strategy in {"simple", "simple-rich", "easy-relations"}:
        k = int(args.prefilter_k)
        if k > 0 and len(headlines) > k:
            tmp = headlines.copy()
            s = tmp["headline"].astype(str)
            tmp["_wc"] = s.str.split().map(len)
            tmp["_commas"] = s.str.count(",")
            tmp["_seps"] = s.str.count("[;:]")
            tmp["_len"] = s.str.len()
            tmp["_quick_score"] = tmp["_wc"] + 2 * tmp["_commas"] + tmp["_seps"] + (tmp["_len"] // 80)
            headlines = tmp.nsmallest(k, "_quick_score")[["id", "headline"]].reset_index(drop=True)

    rel = RelationExtractor()

    if args.strategy in {"simple", "simple-rich", "easy-relations"}:
        docs_all = rel.nlp.pipe(headlines["headline"].tolist(), batch_size=64)
        feats_list: list[dict[str, int]] = []
        scores: list[int] = []
        has_trigger_list: list[bool] = []
        has_where_list: list[bool] = []
        has_who_list: list[bool] = []
        has_what_list: list[bool] = []
        has_when_list: list[bool] = []
        has_subj_list: list[bool] = []
        has_obj_list: list[bool] = []
        has_svo_list: list[bool] = []
        ease_scores: list[int] = []
        for doc in docs_all:
            feat = _complexity_features(doc)
            feats_list.append(feat)
            scores.append(_complexity_score(feat))

            who, trigger_lemma, what, where, when = rel.extract_relations_from_doc(doc)
            trigger_token = rel._pick_trigger_token(doc)  # type: ignore[attr-defined]
            trigger_text = trigger_token.text if trigger_token is not None else (trigger_lemma or "")
            has_trigger_list.append(_has_nonempty(trigger_text))
            has_where_list.append(_has_nonempty(where))
            has_who_list.append(_has_nonempty(who))
            has_what_list.append(_has_nonempty(what))
            has_when_list.append(_has_nonempty(when))

            svo = _svo_features(doc, trigger_token)
            has_subj_list.append(bool(svo["has_subj"]))
            has_obj_list.append(bool(svo["has_obj"]))
            has_svo_list.append(bool(svo["has_svo"]))

            # Ease score uses both extraction presence and syntactic/complexity signals.
            ease_row = {
                **feat,
                "has_trigger": _has_nonempty(trigger_text),
                "has_who": _has_nonempty(who),
                "has_what": _has_nonempty(what),
                "has_where": _has_nonempty(where),
                "has_when": _has_nonempty(when),
                **svo,
            }
            ease_scores.append(_ease_score(ease_row))

        feats_df = pd.DataFrame(feats_list)
        headlines = pd.concat([headlines.reset_index(drop=True), feats_df], axis=1)
        headlines["complexity_score"] = scores
        headlines["has_trigger"] = has_trigger_list
        headlines["has_where"] = has_where_list
        headlines["has_who"] = has_who_list
        headlines["has_what"] = has_what_list
        headlines["has_when"] = has_when_list
        headlines["has_subj"] = has_subj_list
        headlines["has_obj"] = has_obj_list
        headlines["has_svo"] = has_svo_list
        headlines["ease_score"] = ease_scores

        if args.strategy == "simple":
            sample = headlines.nsmallest(n, "complexity_score").reset_index(drop=True)
        elif args.strategy == "simple-rich":
            sample = _greedy_select_simple_rich(
                headlines,
                n=n,
                min_trigger_rate=float(args.min_trigger_rate),
                min_where_rate=float(args.min_where_rate),
            )
        else:
            # easy-relations: prefer clear KTO+CO+TRIGGER and SVO, then lowest complexity.
            core_mask = headlines["has_who"] & headlines["has_what"] & headlines["has_trigger"]
            svo_mask = headlines["has_svo"]

            candidates = headlines[core_mask & svo_mask].copy()
            if len(candidates) < n:
                candidates = headlines[core_mask].copy()
            if len(candidates) < n:
                candidates = headlines.copy()

            candidates = candidates.sort_values(
                ["ease_score", "complexity_score", "id"], ascending=[False, True, True]
            ).reset_index(drop=True)

            sample = candidates.head(n).reset_index(drop=True)

            # If the head-N doesn't meet core quota, do a small greedy fix: ensure at least easy-min-core-rate have core.
            req_core = int((n * float(args.easy_min_core_rate)) + 0.999999)
            if "has_who" in sample.columns and "has_what" in sample.columns and "has_trigger" in sample.columns:
                core_selected = int((sample["has_who"] & sample["has_what"] & sample["has_trigger"]).sum())
                if core_selected < req_core:
                    core_pool = candidates[candidates["has_who"] & candidates["has_what"] & candidates["has_trigger"]]
                    noncore_pool = candidates[~(candidates["has_who"] & candidates["has_what"] & candidates["has_trigger"])]
                    core_pool = core_pool.sort_values(["ease_score", "complexity_score"], ascending=[False, True])
                    noncore_pool = noncore_pool.sort_values(["ease_score", "complexity_score"], ascending=[False, True])
                    new_rows = []
                    new_rows.append(core_pool.head(req_core))
                    new_rows.append(noncore_pool.head(n - req_core))
                    sample = pd.concat(new_rows, axis=0).head(n).reset_index(drop=True)
    else:
        sample = headlines.sample(n=n, random_state=int(args.seed)).reset_index(drop=True)
    loc_preps = set(rel._location_preps)  # type: ignore[attr-defined]
    time_preps = set(rel._time_preps)  # type: ignore[attr-defined]
    docs = rel.nlp.pipe(sample["headline"].tolist(), batch_size=32)

    rows: list[dict[str, Any]] = []
    tagged_rows: list[dict[str, Any]] = []

    for (id_val, sent), doc in zip(sample[["id", "headline"]].itertuples(index=False, name=None), docs, strict=False):
        who, trigger_lemma, what, where, when = rel.extract_relations_from_doc(doc)
        trigger_token = rel._pick_trigger_token(doc)  # type: ignore[attr-defined]
        # TRIGGER in tagged.csv is typically surface-form (not lemma)
        trigger_text = trigger_token.text if trigger_token is not None else (trigger_lemma or "")

        # Match tagged.csv conventions:
        # - strip prepositions from GDZIE/KIEDY
        # - lemma single-token fields like "Holandii" -> "Holandia", "czerwcu" -> "czerwiec"
        kto_out = _normalize_like_tagged(rel, who, preps=loc_preps, lemma_single_token=True)
        gdzie_out = _normalize_like_tagged(rel, where, preps=loc_preps, lemma_single_token=True)
        kiedy_out = _normalize_like_tagged(rel, when, preps=time_preps, lemma_single_token=True)

        # CO in tagged.csv is often a noun phrase; keep surface phrase as-is.
        co_out = (what or "").strip()

        rows.append(
            {
                "id": int(id_val),
                "sentence": sent,
                "KTO": kto_out,
                "CO": co_out,
                "TRIGGER": trigger_text or "",
                "GDZIE": gdzie_out,
                "KIEDY": kiedy_out,
            }
        )

        tagged_rows.append(
            {
                "id": int(id_val),
                "kategoria": "TEST",
                "KTO": kto_out,
                "CO": co_out,
                "TRIGGER": trigger_text or "",
                "GDZIE": gdzie_out,
                "KIEDY": kiedy_out,
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = _safe_write(Path(args.out), out_df, sep=",")

    tagged_df = pd.DataFrame(tagged_rows)
    tagged_path = _safe_write(Path(args.tagged_out), tagged_df, sep=";")

    headlines_out = pd.DataFrame({"id": out_df["id"], "headline": out_df["sentence"]})
    headlines_path = _safe_write(Path(args.headlines_out), headlines_out, sep=",")

    if args.selection_report_out:
        report_cols = ["id", "headline"]
        for c in [
            "ease_score",
            "complexity_score",
            "n_tokens",
            "n_verbs",
            "n_subclauses",
            "n_commas",
            "n_conj",
            "n_punct",
            "has_trigger",
            "has_where",
            "has_who",
            "has_what",
            "has_when",
            "has_svo",
            "has_subj",
            "has_obj",
            "quota_req_trigger",
            "quota_req_where",
            "quota_met_trigger",
            "quota_met_where",
        ]:
            if c in sample.columns:
                report_cols.append(c)
        report_df = sample[report_cols].copy()
        report_path = _safe_write(Path(args.selection_report_out), report_df, sep=",")
        print(f"Wrote: {report_path}")

    print(f"Wrote: {out_path}")
    print(f"Wrote: {tagged_path}")
    print(f"Wrote: {headlines_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
