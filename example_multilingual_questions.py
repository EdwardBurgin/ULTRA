#!/usr/bin/env python
"""
Demonstration script showing how UltraQueryPipeline answers an assortment
of naturally phrased questions in Chinese and English using multilingual-e5,
jieba, and first-order logical reasoning.
"""

import os
import sys

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra.pipeline import UltraQueryPipeline


def main():
    print("=" * 80)
    print(" UltraQuery: Multilingual Natural Language Reasoning (Chinese & English)")
    print("=" * 80)

    # 1. Initialize pipeline with pre-trained UltraQuery on FB15k-237
    print("\n[Step 1] Loading pre-trained UltraQuery pipeline with multilingual-e5 grounding...")
    pipeline = UltraQueryPipeline.from_pretrained(
        dataset="FB15k237LogicalQuery",
        ckpt_path="ckpts/ultraquery.pth",
        threshold=0.0,
    )
    print(f"Graph loaded with {pipeline.graph.num_nodes} entities and {pipeline.graph.num_relations} relations.")
    print(f"Device: {pipeline.device}")

    # Register standard Chinese transliterations for demonstration test entities
    zh_entities = {
        "/m/0184dt": "克里斯托弗·诺兰",
        "/m/01wy5m": "克里斯蒂安·贝尔",
        "/m/0661ql3": "盗梦空间",
        "/m/04jpl": "伦敦",
        "/m/02zd460": "加州大学伯克利分校",
    }
    for mid, zh_name in zh_entities.items():
        if mid in pipeline.ent2id:
            eid = pipeline.ent2id[mid]
            pipeline.nl_parser.name2id[zh_name.lower()] = eid
            pipeline.id2name[eid] = f"{pipeline.id2name.get(eid, mid)} ({zh_name})"
    pipeline.nl_parser._sorted_names = sorted(pipeline.nl_parser.name2id.keys(), key=lambda x: len(x), reverse=True)

    # Multilingual questions across Chinese and English
    questions = [
        # --- Chinese Questions ---
        (
            "1. Chinese 1-hop Path (1p)",
            "克里斯托弗·诺兰导演了哪些电影？",
            5
        ),
        (
            "2. Chinese 1-hop Birthplace (1p)",
            "克里斯托弗·诺兰出生在哪里？",
            3
        ),
        (
            "3. Chinese 2-hop Conjunction / Intersection (2i)",
            "哪些电影由克里斯蒂安·贝尔主演并且由克里斯托弗·诺兰导演？",
            5
        ),
        (
            "4. Chinese Intersection with Negation (2in)",
            "哪些电影由克里斯蒂安·贝尔主演且并非由克里斯托弗·诺兰导演？",
            5
        ),
        (
            "5. Chinese Disjunction / Union (2u)",
            "哪些电影由克里斯蒂安·贝尔主演或者由克里斯托弗·诺兰导演？",
            5
        ),
        (
            "6. Chinese 2-hop Chained Path (2p)",
            "盗梦空间的导演出生在哪里？",
            3
        ),

        # --- English Questions ---
        (
            "7. English 1-hop Path (1p)",
            "What films did Christopher Nolan direct?",
            5
        ),
        (
            "8. English Intersection with Negation (2in)",
            "Which movies star Christian Bale and were not directed by Christopher Nolan?",
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
    print(" Multilingual Demonstration Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
