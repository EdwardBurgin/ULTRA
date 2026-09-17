#!/usr/bin/env python
"""
Demonstration script showing how UltraQueryPipeline answers an assortment
of naturally phrased questions using symbolic First-Order Logic (FOL) reasoning.

Covers:
  - 1-hop path projections (1p)
  - 2-hop chained path queries (2p)
  - 2-hop conjunctions / intersections (2i)
  - 2-hop intersections with negation (2in)
  - 2-hop disjunctions / unions (2u)
"""

import os
import sys

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra.pipeline import UltraQueryPipeline


def main():
    print("=" * 80)
    print(" UltraQuery: Natural Language Question Answering Demonstration")
    print("=" * 80)

    # 1. Initialize pipeline with pre-trained UltraQuery on FB15k-237
    print("\n[Step 1] Loading pre-trained UltraQuery pipeline with text grounding...")
    pipeline = UltraQueryPipeline.from_pretrained(
        dataset="FB15k237LogicalQuery",
        ckpt_path="ckpts/ultraquery.pth",
        threshold=0.0,
    )
    print(f"Graph loaded with {pipeline.graph.num_nodes} entities and {pipeline.graph.num_relations} relations.")
    print(f"Device: {pipeline.device}")

    # Assortment of diverse natural language questions
    questions = [
        # --- 1-hop Path Queries (1p) ---
        (
            "1. 1-hop Path Query (1p)",
            "What films did Christopher Nolan direct?",
            5
        ),
        (
            "2. 1-hop Path Query (1p)",
            "Where was Christopher Nolan born?",
            3
        ),
        (
            "3. 1-hop Path Query (1p)",
            "What are the major fields of study at University of California, Berkeley?",
            5
        ),
        (
            "4. 1-hop Path Query (1p)",
            "What movies star Christian Bale?",
            5
        ),

        # --- 2-hop Chained Path Query (2p) ---
        (
            "5. 2-hop Chained Path Query (2p)",
            "Where was the director of Inception born?",
            3
        ),

        # --- 2-hop Conjunction / Intersection Query (2i) ---
        (
            "6. Conjunction / Intersection Query (2i)",
            "Which movies star Christian Bale and were directed by Christopher Nolan?",
            5
        ),

        # --- 2-hop Intersection with Negation Query (2in) ---
        (
            "7. Intersection with Negation Query (2in)",
            "Which movies star Christian Bale and were not directed by Christopher Nolan?",
            5
        ),

        # --- 2-hop Disjunction / Union Query (2u) ---
        (
            "8. Disjunction / Union Query (2u)",
            "Which movies star Christian Bale or were directed by Christopher Nolan?",
            5
        ),
    ]

    for title, question, top_k in questions:
        print("\n" + "=" * 80)
        print(f" {title}")
        print("=" * 80)
        df = pipeline.ask_natural(question, top_k=top_k, explain=True)
        print(df[["rank", "entity", "entity_mid", "probability", "logit"]].to_string(index=False))

    print("\n" + "=" * 80)
    print(" Demonstration Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
