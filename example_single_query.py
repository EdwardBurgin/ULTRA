#!/usr/bin/env python
"""
Example demonstrating how to ask single complex logical queries using UltraQueryPipeline.

Supports:
- 1p: 1-hop path query
- 2p: 2-hop path query
- 2i: 2-hop intersection / conjunction
- 2in: 2-hop intersection with negation
- Custom nested tuple queries using string entity/relation names
"""

import os
import sys

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultra.pipeline import UltraQueryPipeline


def main():
    print("================================================================")
    print(" UltraQuery: Single Complex Query Answering Demonstration")
    print("================================================================")

    # 1. Initialize pipeline for FB15k-237 complex queries on GPU
    print("\n[Step 1] Loading pre-trained UltraQuery pipeline on FB15k-237...")
    pipeline = UltraQueryPipeline.from_pretrained(
        dataset="FB15k237LogicalQuery",
        ckpt_path="ckpts/ultraquery.pth",
        threshold=0.0,
    )
    print(f"Loaded graph with {pipeline.graph.num_nodes} entities and {pipeline.graph.num_relations} relations.")
    print(f"Device: {pipeline.device}")

    # -------------------------------------------------------------
    # Example 1: 1-hop Projection (1p)
    # Question: "What are the major fields of study associated with institution /m/01k2wn?"
    # -------------------------------------------------------------
    print("\n----------------------------------------------------------------")
    print("Example 1: 1-hop Path Query (1p)")
    print("Query: ?X : major_field_of_study('/m/01k2wn', ?X)")
    print("----------------------------------------------------------------")
    entity_id = 927  # '/m/01k2wn'
    rel_id = 202     # '+/education/educational_institution/students_graduates./education/education/major_field_of_study'

    df_1p = pipeline.ask_1p(entity_id, rel_id, top_k=5)
    print(df_1p.to_string(index=False))

    # -------------------------------------------------------------
    # Example 2: 2-hop Projection (2p)
    # Question: "What are the entities reachable via 2 hops from entity 927?"
    # -------------------------------------------------------------
    print("\n----------------------------------------------------------------")
    print("Example 2: 2-hop Path Query (2p)")
    print("Query: ?X : rel2(rel1(entity, ?Y), ?X)")
    print("----------------------------------------------------------------")
    rel2_id = 95  # e.g. parent relation or classification
    df_2p = pipeline.ask_2p(entity_id, rel_id, rel2_id, top_k=5)
    print(df_2p.to_string(index=False))

    # -------------------------------------------------------------
    # Example 3: 2-hop Intersection / Conjunction (2i)
    # Question: "What entities satisfy BOTH r1(e1, ?X) AND r2(e2, ?X)?"
    # -------------------------------------------------------------
    print("\n----------------------------------------------------------------")
    print("Example 3: Conjunction / Intersection Query (2i)")
    print("Query: ?X : r1(e1, ?X) AND r2(e2, ?X)")
    print("----------------------------------------------------------------")
    e1, r1 = 32, 97
    e2, r2 = 11188, 39
    df_2i = pipeline.ask_2i(e1, r1, e2, r2, top_k=5)
    print(df_2i.to_string(index=False))

    # -------------------------------------------------------------
    # Example 4: Intersection with Negation (2in)
    # Question: "What entities satisfy r1(e1, ?X) AND NOT r2(e2, ?X)?"
    # -------------------------------------------------------------
    print("\n----------------------------------------------------------------")
    print("Example 4: Intersection with Negation Query (2in)")
    print("Query: ?X : r1(e1, ?X) AND NOT r2(e2, ?X)")
    print("----------------------------------------------------------------")
    df_2in = pipeline.ask_2in(e1, r1, e2, r2, top_k=5)
    print(df_2in.to_string(index=False))

    # -------------------------------------------------------------
    # Example 5: Asking with Human-Readable String Identifiers
    # -------------------------------------------------------------
    print("\n----------------------------------------------------------------")
    print("Example 5: Query with String Entity & Relation Names")
    print("----------------------------------------------------------------")
    str_query = (
        "/m/01k2wn",
        ("+/education/educational_institution/students_graduates./education/education/major_field_of_study",)
    )
    df_str = pipeline.ask(str_query, top_k=5)
    print(df_str.to_string(index=False))

    print("\n================================================================")
    print(" Demonstration Complete!")
    print("================================================================")


if __name__ == "__main__":
    main()

