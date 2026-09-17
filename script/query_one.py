#!/usr/bin/env python
import argparse
import ast
import os
import sys

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultra.pipeline import UltraQueryPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Execute a single complex logical query with UltraQuery.")
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="FB15k237LogicalQuery",
        help="Dataset name ('FB15k237LogicalQuery', 'InductiveFB15k237Query', 'FB15kLogicalQuery', 'NELL995LogicalQuery')",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="550",
        help="Version for inductive datasets (e.g. 550, 300, 217).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="Graph split to evaluate on.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="ckpts/ultraquery.pth",
        help="Path to UltraQuery checkpoint.",
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        default=None,
        help="Logical query in BetaE nested-tuple format, e.g. \"(927, (202,))\" or \"('/m/01k2wn', ('+/education/...',))\".",
    )
    parser.add_argument(
        "-k", "--top_k",
        type=int,
        default=10,
        help="Number of top predictions to display (default: 10).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Confidence threshold (0.0 for ultraquery.pth, 0.8 for vanilla Ultra).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on ('cuda:0' or 'cpu').",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n========================================================")
    print(f" UltraQuery Single Complex Query Pipeline")
    print(f"========================================================")
    print(f"Dataset   : {args.dataset} (version={args.version}, split={args.split})")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Loading pipeline...")

    pipeline = UltraQueryPipeline.from_pretrained(
        dataset=args.dataset,
        ckpt_path=args.ckpt,
        version=args.version,
        split=args.split,
        device=args.device,
        threshold=args.threshold,
    )
    print(f"Graph loaded: {pipeline.graph.num_nodes} nodes, {pipeline.graph.num_relations} relations.")
    print(f"Device      : {pipeline.device}")

    # If query not provided via CLI, pick a representative query or prompt user
    if args.query is None:
        print("\nNo query specified via --query (-q). Using demo query:")
        if "237" in args.dataset:
            # Major field of study for educational institution /m/01k2wn
            demo_query = (927, (202,))
        else:
            # Fallback to first available entity and relation
            demo_query = (0, (0,))
        query_val = demo_query
    else:
        try:
            query_val = ast.literal_eval(args.query)
        except Exception:
            query_val = args.query

    print(f"\nQuery input: {query_val}")
    print("\nExecuting inference...")
    df_results = pipeline.ask(query_val, top_k=args.top_k)

    print("\n" + "=" * 70)
    print(f" Top {args.top_k} Candidate Answers:")
    print("=" * 70)
    # Print formatted table
    print(df_results.to_string(index=False))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

