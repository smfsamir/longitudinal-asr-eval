import argparse
import json
from pathlib import Path

from .manifest import read_manifest, write_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(prog="asr-eval")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Create a sample manifest")
    prepare.add_argument(
        "dataset", choices=["l1-arctic", "l2-arctic", "suitcase", "saa", "openslr83", "allstar"]
    )
    prepare.add_argument(
        "--root", required=True, help="Corpus directory, L2 archive, or ALLSTAR index CSV"
    )
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--cache", default="data/cache/l2-arctic")
    prepare.add_argument("--selection", help="CSV of stable sample IDs")
    prepare.add_argument("--accents", nargs="+", help="Keep only these accent labels")
    prepare.add_argument(
        "--balanced", action="store_true", help="Draw the paper's OpenSLR 550-sample subset"
    )
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--metadata", help="SAA speaker spreadsheet")
    prepare.add_argument(
        "--skip-invalid-intervals",
        action="store_true",
        help="Exclude ALLSTAR sentences extending beyond the recording, with a warning",
    )
    prepare.add_argument("--min-speakers", type=int, default=20)
    prepare.add_argument("--control-audio")
    prepare.add_argument("--control-text")
    merge = sub.add_parser("merge", help="Combine sample manifests")
    merge.add_argument("manifests", nargs="+")
    merge.add_argument("--output", required=True)
    validate = sub.add_parser("validate", help="Check sample metadata and audio")
    validate.add_argument("manifest")
    run = sub.add_parser("run", help="Transcribe with one configured model")
    run.add_argument("manifest")
    run.add_argument("--config", default="configs/paper.json")
    run.add_argument("--model", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--device", default="auto")
    run.add_argument("--limit", type=int)
    run.add_argument("--keep-going", action="store_true")
    score = sub.add_parser("score", help="Recompute WER/CER from a prediction CSV")
    score.add_argument("input")
    score.add_argument("--output", required=True)
    analyze = sub.add_parser("analyze", help="Write paper figures and tables")
    analyze.add_argument("--scores", nargs="+", required=True)
    analyze.add_argument("--config", default="configs/paper.json")
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--datasets", nargs="+")
    analyze.add_argument("--allow-partial", action="store_true")
    analyze.add_argument("--bootstrap", type=int)
    analyze.add_argument("--seed", type=int)
    audit = sub.add_parser("audit", help="Compare analysis summaries to reported paper values")
    audit.add_argument("summary")
    audit.add_argument("--reference", default="data/paper_values.csv")
    audit.add_argument("--output", required=True)
    demo = sub.add_parser(
        "demo", help="Exercise the pipeline with synthetic audio and replayed predictions"
    )
    demo.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        dispatch(args)
    except (ValueError, FileNotFoundError, KeyError, RuntimeError) as error:
        parser.exit(1, f"asr-eval: {error}\n")


def dispatch(args):
    if args.command == "prepare":
        from . import datasets as ds

        if args.balanced and (args.dataset != "openslr83" or args.selection):
            raise ValueError(
                "--balanced is only for OpenSLR and cannot be combined with --selection"
            )
        if bool(args.control_audio) != bool(args.control_text):
            raise ValueError("Supply both --control-audio and --control-text")
        if args.min_speakers < 1:
            raise ValueError("--min-speakers must be positive")
        if args.skip_invalid_intervals and args.dataset != "allstar":
            raise ValueError("--skip-invalid-intervals is only for ALLSTAR")
        loaders = {
            "l1-arctic": lambda: ds.l1_arctic(args.root),
            "l2-arctic": lambda: ds.l2_arctic(args.root, args.cache),
            "suitcase": lambda: ds.l2_arctic(args.root, args.cache, spontaneous=True),
            "saa": lambda: ds.speech_accent(args.root, args.metadata, args.min_speakers),
            "openslr83": lambda: ds.openslr83(args.root),
            "allstar": lambda: ds.allstar(args.root, args.skip_invalid_intervals),
        }
        samples = list(loaders[args.dataset]())
        if args.selection:
            samples = ds.select_samples(samples, args.selection)
        if args.balanced:
            samples = ds.balanced_openslr(samples, args.seed)
        if args.accents:
            samples = [s for s in samples if s.accent in args.accents]
        if args.control_audio:
            if args.dataset != "suitcase":
                raise ValueError("The native control belongs to the suitcase partition")
            samples.append(ds.suitcase_control(args.control_audio, args.control_text))
        # As upstream, skip references consisting only of non-speech annotations.
        from .normalization import normalize_english

        retained = [s for s in samples if normalize_english(s.text).strip()]
        print(json.dumps({"skipped_empty_reference": len(samples) - len(retained)}))
        report = ds.validate_samples(retained)
        write_manifest(retained, args.output)
        print(json.dumps(report, indent=2))
    elif args.command == "merge":
        samples = [sample for path in args.manifests for sample in read_manifest(path)]
        write_manifest(samples, args.output)
    elif args.command == "validate":
        from .datasets import validate_samples

        print(json.dumps(validate_samples(read_manifest(args.manifest)), indent=2))
    elif args.command == "run":
        from .runner import run

        config = json.loads(Path(args.config).read_text())
        models = {m["id"]: m for m in config["models"]}
        if args.model not in models:
            raise ValueError(f"Choose a model from {', '.join(models)}")
        print(
            json.dumps(
                run(
                    args.manifest,
                    models[args.model],
                    args.output,
                    args.device,
                    config.get("seed", 0),
                    args.limit,
                    args.keep_going,
                )
            )
        )
    elif args.command == "score":
        import pandas as pd

        from .metrics import score

        df = pd.read_csv(args.input, keep_default_na=False, dtype={"sample_id": str})
        records = []
        for row in df.to_dict("records"):
            metrics = score(row["groundtruth"], row["prediction"])
            if metrics is not None:
                records.append({**row, **metrics})
        if not records:
            raise ValueError("No nonempty references")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(args.output, index=False)
    elif args.command == "analyze":
        from .analysis import analyze, read_scores

        config = json.loads(Path(args.config).read_text())
        summary = analyze(
            read_scores(args.scores),
            config,
            args.output,
            args.datasets,
            args.allow_partial,
            args.bootstrap,
            args.seed,
        )
        print(
            summary[["dataset", "minority_md", "standard_md", "spearman_rho"]].to_string(
                index=False
            )
        )
    elif args.command == "audit":
        import pandas as pd

        actual = pd.read_csv(args.summary).set_index("dataset")
        expected = pd.read_csv(args.reference)
        rows = []
        for row in expected.to_dict("records"):
            value = (
                float(actual.loc[row["dataset"], row["metric"]])
                if row["dataset"] in actual.index
                else None
            )
            error = value - row["value"] if value is not None else None
            rows.append(
                {
                    **row,
                    "actual": value,
                    "difference": error,
                    "matches": abs(error) <= row["tolerance"] if error is not None else False,
                }
            )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(pd.DataFrame(rows).to_string(index=False))
    elif args.command == "demo":
        from .demo import demo

        demo(args.output)
