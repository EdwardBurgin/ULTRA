import pandas as pd
from ultra.export_utils import (
    dataframe_to_graph,
    graph_to_dataframe,
    relation_graph_to_dataframe,
    graph_summary,
    dataframes_to_dataset,
    dataset_to_dataframes
)
from ultra.datasets import CoDExSmall

def run_tests():
    print("=== Test 1: DataFrame to Graph & Summary ===")
    df = pd.DataFrame([
        {"head": "Alice", "relation": "friends_with", "tail": "Bob"},
        {"head": "Bob", "relation": "lives_in", "tail": "Wonderland"},
        {"head": "Alice", "relation": "born_in", "tail": "Wonderland"},
    ])
    graph, ent_vocab, rel_vocab = dataframe_to_graph(df)
    summary = graph_summary(graph)
    print("Graph Summary:")
    print(summary)
    assert summary["num_nodes"] == 3
    assert summary["num_relations"] == 6  # 3 direct + 3 inverse
    assert summary["has_relation_graph"] == True

    print("\n=== Test 2: Graph to DataFrame ===")
    df_exported = graph_to_dataframe(graph, entity_vocab=ent_vocab, rel_vocab=rel_vocab)
    print("Exported DataFrame:")
    print(df_exported)
    assert len(df_exported) == 3
    assert set(df_exported.columns) >= {"head", "relation", "tail", "head_id", "relation_id", "tail_id"}

    print("\n=== Test 3: Relation Graph to DataFrame ===")
    rel_df = relation_graph_to_dataframe(graph.relation_graph, rel_vocab=rel_vocab)
    print("Relation Graph DataFrame head:")
    print(rel_df.head())
    assert "interaction_type" in rel_df.columns
    assert "src_relation" in rel_df.columns

    print("\n=== Test 4: DataFrames to Dataset (train, valid, test) ===")
    valid_df = pd.DataFrame([
        {"head": "Alice", "relation": "lives_in", "tail": "Wonderland"}
    ])
    splits = dataframes_to_dataset(df, valid_df=valid_df)
    print("Splits keys:", list(splits.keys()))
    assert splits["train_data"] is not None
    assert splits["valid_data"] is not None
    assert splits["valid_data"].target_edge_index.shape[1] == 1

    print("\n=== Test 5: Existing ULTRA Dataset to DataFrames (CoDExSmall) ===")
    dataset = CoDExSmall(root="~/git/ULTRA/kg-datasets/")
    codex_dfs = dataset_to_dataframes(dataset)
    print("CoDExSmall train shape:", codex_dfs["train"].shape)
    print("CoDExSmall valid shape:", codex_dfs["valid"].shape)
    print("CoDExSmall test shape:", codex_dfs["test"].shape)
    print(codex_dfs["train"].head(3))
    assert codex_dfs["train"].shape[0] == 32888
    assert codex_dfs["valid"].shape[0] == 1827
    assert codex_dfs["test"].shape[0] == 1828

    print("\nALL 5 TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()
