import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Data
from ultra.tasks import build_relation_graph


RELATION_GRAPH_INTERACTIONS = {
    0: "head_to_head",
    1: "tail_to_tail",
    2: "head_to_tail",
    3: "tail_to_head",
}


def _ensure_vocab_map(vocab: Optional[Union[Dict, List, Tuple]]) -> Optional[Dict[int, str]]:
    """Normalize vocab into a dict of {int_id: str_name} or None."""
    if vocab is None:
        return None
    if isinstance(vocab, dict):
        if not vocab:
            return {}
        first_k = next(iter(vocab.keys()))
        if isinstance(first_k, str):
            return {v: k for k, v in vocab.items()}
        return vocab
    if isinstance(vocab, (list, tuple)):
        return {i: str(name) for i, name in enumerate(vocab)}
    return None


def graph_to_dataframe(
    data: Data,
    entity_vocab: Optional[Union[Dict, List]] = None,
    rel_vocab: Optional[Union[Dict, List]] = None,
    target_edges: bool = False,
    include_inverses: bool = False,
) -> pd.DataFrame:
    """
    Export a PyG Data graph structure from ULTRA into a pandas DataFrame.

    Parameters:
        data (Data): PyG Data object (e.g. dataset[0], train_data, valid_data, test_data).
        entity_vocab (dict or list, optional): Entity ID-to-name mapping or name-to-ID mapping.
        rel_vocab (dict or list, optional): Relation ID-to-name mapping or name-to-ID mapping.
        target_edges (bool): If True, exports target_edge_index/target_edge_type instead of edge_index/edge_type.
        include_inverses (bool): If False, excludes automatically appended inverse edges (rel_id >= original num_relations).

    Returns:
        pd.DataFrame: DataFrame with columns ['head', 'relation', 'tail', 'head_id', 'relation_id', 'tail_id'].
    """
    if target_edges and hasattr(data, "target_edge_index") and data.target_edge_index is not None:
        edge_index = data.target_edge_index.cpu()
        edge_type = data.target_edge_type.cpu()
    else:
        edge_index = data.edge_index.cpu()
        edge_type = data.edge_type.cpu()

    heads = edge_index[0].tolist()
    tails = edge_index[1].tolist()
    rels = edge_type.tolist()

    df = pd.DataFrame({
        "head_id": heads,
        "relation_id": rels,
        "tail_id": tails,
    })

    total_rels = getattr(data, "num_relations", None)
    if not include_inverses and total_rels is not None:
        # In ULTRA, transductive graphs double relations by adding inverse edges (rel_id + num_relations)
        # So base relations are [0, total_rels // 2)
        base_num_rels = int(total_rels) // 2
        df = df[df["relation_id"] < base_num_rels].copy()

    id2ent = _ensure_vocab_map(entity_vocab)
    id2rel = _ensure_vocab_map(rel_vocab)

    if id2ent is not None:
        df["head"] = df["head_id"].map(lambda x: id2ent.get(x, str(x)))
        df["tail"] = df["tail_id"].map(lambda x: id2ent.get(x, str(x)))
    else:
        df["head"] = df["head_id"]
        df["tail"] = df["tail_id"]

    if id2rel is not None:
        df["relation"] = df["relation_id"].map(lambda x: id2rel.get(x, str(x)))
    else:
        df["relation"] = df["relation_id"]

    cols = ["head", "relation", "tail", "head_id", "relation_id", "tail_id"]
    return df[cols]


def relation_graph_to_dataframe(
    rel_graph: Data,
    rel_vocab: Optional[Union[Dict, List]] = None,
) -> pd.DataFrame:
    """
    Export ULTRA's custom relation graph (graph.relation_graph) into a pandas DataFrame.

    Parameters:
        rel_graph (Data): Relation graph object with edge_index and edge_type (0..3).
        rel_vocab (dict or list, optional): Relation ID-to-name mapping.

    Returns:
        pd.DataFrame: DataFrame describing relation interactions (head-to-head, tail-to-tail, etc.).
    """
    if hasattr(rel_graph, "relation_graph"):
        rel_graph = rel_graph.relation_graph

    if rel_graph is None or not hasattr(rel_graph, "edge_index"):
        raise ValueError("Provided object does not contain a relation_graph.")

    edge_index = rel_graph.edge_index.cpu()
    edge_type = rel_graph.edge_type.cpu()

    src_rels = edge_index[0].tolist()
    dst_rels = edge_index[1].tolist()
    type_ids = edge_type.tolist()

    df = pd.DataFrame({
        "src_rel_id": src_rels,
        "dst_rel_id": dst_rels,
        "type_id": type_ids,
        "interaction_type": [RELATION_GRAPH_INTERACTIONS.get(t, "unknown") for t in type_ids],
    })

    id2rel = _ensure_vocab_map(rel_vocab)
    if id2rel is not None:
        df["src_relation"] = df["src_rel_id"].map(lambda x: id2rel.get(x, str(x)))
        df["dst_relation"] = df["dst_rel_id"].map(lambda x: id2rel.get(x, str(x)))
        cols = ["src_relation", "dst_relation", "interaction_type", "src_rel_id", "dst_rel_id", "type_id"]
    else:
        cols = ["src_rel_id", "dst_rel_id", "interaction_type", "type_id"]

    return df[cols]


def dataframe_to_graph(
    df: pd.DataFrame,
    head_col: str = "head",
    rel_col: str = "relation",
    tail_col: str = "tail",
    entity_vocab: Optional[Dict[str, int]] = None,
    rel_vocab: Optional[Dict[str, int]] = None,
    add_inverse_edges: bool = True,
    build_rel_graph: bool = True,
) -> Tuple[Data, Dict[str, int], Dict[str, int]]:
    """
    Import a pandas DataFrame of triples into an ULTRA-compatible PyG Data graph structure.

    Parameters:
        df (pd.DataFrame): DataFrame containing triples.
        head_col (str): Column name for head entity.
        rel_col (str): Column name for relation.
        tail_col (str): Column name for tail entity.
        entity_vocab (dict, optional): Mapping of {entity_name: entity_id}. Built if not provided.
        rel_vocab (dict, optional): Mapping of {rel_name: rel_id}. Built if not provided.
        add_inverse_edges (bool): If True, appends inverse edges (t, h, r + num_relations) as standard in ULTRA.
        build_rel_graph (bool): If True, constructs graph.relation_graph via ULTRA's build_relation_graph.

    Returns:
        tuple: (graph: Data, entity_vocab: dict, rel_vocab: dict)
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    # Build or normalize entity vocabulary {name: int_id}
    if entity_vocab is None:
        unique_entities = sorted(pd.concat([df[head_col], df[tail_col]]).astype(str).unique())
        entity_vocab = {ent: i for i, ent in enumerate(unique_entities)}
    else:
        first_k = next(iter(entity_vocab.keys())) if entity_vocab else None
        if isinstance(first_k, int):
            entity_vocab = {v: k for k, v in entity_vocab.items()}

    # Build or normalize relation vocabulary {name: int_id}
    if rel_vocab is None:
        unique_rels = sorted(df[rel_col].astype(str).unique())
        rel_vocab = {r: i for i, r in enumerate(unique_rels)}
    else:
        first_k = next(iter(rel_vocab.keys())) if rel_vocab else None
        if isinstance(first_k, int):
            rel_vocab = {v: k for k, v in rel_vocab.items()}

    num_nodes = len(entity_vocab)
    num_relations = len(rel_vocab)

    head_ids = df[head_col].astype(str).map(entity_vocab).values
    tail_ids = df[tail_col].astype(str).map(entity_vocab).values
    rel_ids = df[rel_col].astype(str).map(rel_vocab).values

    if pd.isna(head_ids).any() or pd.isna(tail_ids).any() or pd.isna(rel_ids).any():
        raise ValueError("Some entities or relations in the DataFrame could not be mapped using the provided vocabularies.")

    direct_edges = torch.as_tensor(np.stack([head_ids, tail_ids]), dtype=torch.long)
    direct_types = torch.as_tensor(rel_ids, dtype=torch.long)

    if add_inverse_edges:
        inverse_edges = direct_edges.flip(0)
        inverse_types = direct_types + num_relations
        all_edges = torch.cat([direct_edges, inverse_edges], dim=1)
        all_types = torch.cat([direct_types, inverse_types], dim=0)
        total_num_relations = num_relations * 2
    else:
        all_edges = direct_edges
        all_types = direct_types
        total_num_relations = num_relations

    graph = Data(
        edge_index=all_edges,
        edge_type=all_types,
        num_nodes=num_nodes,
        num_relations=total_num_relations,
        target_edge_index=direct_edges,
        target_edge_type=direct_types,
    )

    if build_rel_graph:
        graph = build_relation_graph(graph)

    return graph, entity_vocab, rel_vocab


def dataframes_to_dataset(
    train_df: pd.DataFrame,
    valid_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    head_col: str = "head",
    rel_col: str = "relation",
    tail_col: str = "tail",
    entity_vocab: Optional[Dict[str, int]] = None,
    rel_vocab: Optional[Dict[str, int]] = None,
    build_rel_graph: bool = True,
) -> Dict[str, Union[Data, Dict[str, int], None]]:
    """
    Import train, valid, and test DataFrames into standard transductive ULTRA dataset splits.

    Parameters:
        train_df (pd.DataFrame): Training triples DataFrame.
        valid_df (pd.DataFrame, optional): Validation triples DataFrame.
        test_df (pd.DataFrame, optional): Test triples DataFrame.
        head_col, rel_col, tail_col (str): Column names in the DataFrames.
        entity_vocab, rel_vocab (dict, optional): Shared vocabularies. Built across all splits if omitted.
        build_rel_graph (bool): If True, generates relation graphs for each split.

    Returns:
        dict: {
            'train_data': Data,
            'valid_data': Data (or None),
            'test_data': Data (or None),
            'entity_vocab': dict,
            'rel_vocab': dict
        }
    """
    dfs = [df for df in [train_df, valid_df, test_df] if df is not None]
    if entity_vocab is None:
        all_entities = sorted(pd.concat([pd.concat([d[head_col], d[tail_col]]) for d in dfs]).astype(str).unique())
        entity_vocab = {ent: i for i, ent in enumerate(all_entities)}
    if rel_vocab is None:
        all_rels = sorted(pd.concat([d[rel_col] for d in dfs]).astype(str).unique())
        rel_vocab = {r: i for i, r in enumerate(all_rels)}

    train_data, _, _ = dataframe_to_graph(
        train_df, head_col, rel_col, tail_col,
        entity_vocab=entity_vocab, rel_vocab=rel_vocab,
        add_inverse_edges=True, build_rel_graph=build_rel_graph,
    )

    result = {
        "train_data": train_data,
        "valid_data": None,
        "test_data": None,
        "entity_vocab": entity_vocab,
        "rel_vocab": rel_vocab,
    }

    # Transductive setup: valid and test evaluate on their target edges using the train graph as background
    for split_name, split_df in [("valid_data", valid_df), ("test_data", test_df)]:
        if split_df is not None:
            v_heads = split_df[head_col].astype(str).map(entity_vocab).values
            v_tails = split_df[tail_col].astype(str).map(entity_vocab).values
            v_rels = split_df[rel_col].astype(str).map(rel_vocab).values

            target_edges = torch.as_tensor(np.stack([v_heads, v_tails]), dtype=torch.long)
            target_types = torch.as_tensor(v_rels, dtype=torch.long)

            split_data = Data(
                edge_index=train_data.edge_index,
                edge_type=train_data.edge_type,
                num_nodes=train_data.num_nodes,
                num_relations=train_data.num_relations,
                target_edge_index=target_edges,
                target_edge_type=target_types,
            )
            if build_rel_graph:
                split_data.relation_graph = train_data.relation_graph
            result[split_name] = split_data

    return result


def dataset_to_dataframes(
    dataset,
    entity_vocab: Optional[Union[Dict, List]] = None,
    rel_vocab: Optional[Union[Dict, List]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Extract train, valid, and test splits of an ULTRA dataset (e.g. CoDExSmall, FB15k237)
    into a dictionary of pandas DataFrames.

    Parameters:
        dataset: ULTRA dataset instance (e.g. CoDExSmall(root=...)).
        entity_vocab, rel_vocab (dict or list, optional): Custom vocabularies.

    Returns:
        dict: {'train': pd.DataFrame, 'valid': pd.DataFrame, 'test': pd.DataFrame}
    """
    train_data = dataset[0]
    valid_data = dataset[1] if len(dataset) > 1 else None
    test_data = dataset[2] if len(dataset) > 2 else None

    # Check for vocabs attached to dataset
    if entity_vocab is None and hasattr(dataset, "inv_entity_vocab"):
        entity_vocab = dataset.inv_entity_vocab
    if rel_vocab is None and hasattr(dataset, "inv_rel_vocab"):
        rel_vocab = dataset.inv_rel_vocab

    dfs = {
        "train": graph_to_dataframe(train_data, entity_vocab, rel_vocab, target_edges=True),
    }
    if valid_data is not None:
        dfs["valid"] = graph_to_dataframe(valid_data, entity_vocab, rel_vocab, target_edges=True)
    if test_data is not None:
        dfs["test"] = graph_to_dataframe(test_data, entity_vocab, rel_vocab, target_edges=True)

    return dfs


def graph_summary(data: Data) -> pd.Series:
    """
    Return a pandas Series summarizing nodes, edges, relations, and relation graph status.
    Useful for quick introspection in Jupyter Notebooks.
    """
    info = {
        "num_nodes": getattr(data, "num_nodes", None),
        "num_edges": data.edge_index.shape[1] if hasattr(data, "edge_index") and data.edge_index is not None else 0,
        "num_relations": getattr(data, "num_relations", None),
        "has_target_edges": hasattr(data, "target_edge_index") and data.target_edge_index is not None,
        "num_target_edges": data.target_edge_index.shape[1] if hasattr(data, "target_edge_index") and data.target_edge_index is not None else 0,
        "has_relation_graph": hasattr(data, "relation_graph") and data.relation_graph is not None,
    }
    if info["has_relation_graph"]:
        rg = data.relation_graph
        info["rel_graph_nodes"] = rg.num_nodes
        info["rel_graph_edges"] = rg.edge_index.shape[1]
    return pd.Series(info)
