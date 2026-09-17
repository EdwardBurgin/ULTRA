import os
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch_geometric.data import Data

from ultra.models import Ultra
from ultra.query_utils import Query
from ultra.tasks import build_relation_graph
from ultra.ultraquery import UltraQuery


class UltraQueryPipeline:
    """
    High-level pipeline for asking single complex logical queries against knowledge graphs.
    
    Supports:
      - 1-hop projection (1p): (e, (r,))
      - 2-hop projection (2p): (e, (r1, r2))
      - 3-hop projection (3p): (e, (r1, r2, r3))
      - 2-hop intersection (2i): ((e1, (r1,)), (e2, (r2,)))
      - 3-hop intersection (3i): ((e1, (r1,)), (e2, (r2,)), (e3, (r3,)))
      - 2-hop intersection with negation (2in): ((e1, (r1,)), (e2, (r2, -2)))
      - 2-hop union (2u): ((e1, (r1,)), (e2, (r2,)), (-1,))
      - Arbitrary nested BetaE logical formulas
    """

    def __init__(
        self,
        model: UltraQuery,
        graph: Data,
        id2ent: Optional[Dict[int, str]] = None,
        id2rel: Optional[Dict[int, str]] = None,
        ent2id: Optional[Dict[str, int]] = None,
        rel2id: Optional[Dict[str, int]] = None,
        device: Optional[torch.device] = None,
        dataset_name: Optional[str] = None,
        easy_answers: Optional[Dict[Tuple, Any]] = None,
        hard_answers: Optional[Dict[Tuple, Any]] = None,
    ):
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

        if not hasattr(graph, "relation_graph") or graph.relation_graph is None:
            graph = build_relation_graph(graph.cpu())
        self.graph = graph.to(self.device)

        # Build vocabulary lookups
        self.id2ent = id2ent or {}
        self.id2rel = id2rel or {}
        self.ent2id = ent2id or {v: k for k, v in self.id2ent.items()}
        self.rel2id = rel2id or {v: k for k, v in self.id2rel.items()}

        self.dataset_name = dataset_name or "CustomGraph"
        self.easy_answers = easy_answers or {}
        self.hard_answers = hard_answers or {}

    @classmethod
    def from_pretrained(
        cls,
        dataset: Union[str, Any] = "FB15k237LogicalQuery",
        ckpt_path: str = "ckpts/ultraquery.pth",
        version: Optional[Union[str, int]] = 550,
        split: str = "test",
        device: Optional[Union[str, torch.device]] = None,
        threshold: float = 0.0,
        root: str = "~/git/ULTRA/query-datasets/",
    ) -> "UltraQueryPipeline":
        """
        Factory method to initialize the pipeline from a pre-trained checkpoint and dataset.

        Parameters:
            dataset: Dataset name ('FB15k237LogicalQuery', 'InductiveFB15k237Query', 'FB15kLogicalQuery', 'NELL995LogicalQuery'),
                     or an existing LogicalQueryDataset object, or a PyG Data graph.
            ckpt_path: Path to UltraQuery checkpoint ('ckpts/ultraquery.pth' or 'ckpts/ultra_4g.pth').
            version: Version for inductive datasets (e.g. 550, 300, 217).
            split: Graph split to use ('test', 'valid', or 'train').
            device: 'cuda:0' or 'cpu'.
            threshold: Score threshold (0.0 for ultraquery.pth, ~0.8 for vanilla Ultra).
            root: Root path for query datasets.
        """
        device = torch.device(device if device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        root = os.path.expanduser(root)

        id2ent = {}
        id2rel = {}
        easy_answers = {}
        hard_answers = {}
        dataset_name = dataset if isinstance(dataset, str) else getattr(dataset, "name", "custom")

        # 1. Load or resolve graph and vocabularies
        if isinstance(dataset, str):
            ds_name = dataset.lower()
            if "inductive" in ds_name or "fb15k237inductive" in ds_name:
                from ultra.datasets_query import InductiveFB15k237Query
                ds = InductiveFB15k237Query(root=root, version=version)
                if split == "train":
                    graph = ds.train_graph
                elif split == "valid":
                    graph = ds.valid_graph
                else:
                    graph = ds.test_graph
                id2ent = ds.inv_entity_vocab
                id2rel = ds.inv_relation_vocab
                dataset_name = f"InductiveFB15k237Query:{version}"
            else:
                # Transductive datasets (BetaE format: FB15k-237-betae, etc.)
                if "237" in ds_name:
                    dir_name = "FB15k-237-betae"
                    dataset_name = "FB15k237LogicalQuery"
                elif "nell" in ds_name:
                    dir_name = "NELL-betae"
                    dataset_name = "NELL995LogicalQuery"
                else:
                    dir_name = "FB15k-betae"
                    dataset_name = "FB15kLogicalQuery"

                path = os.path.join(root, dir_name)
                if not os.path.exists(path):
                    # Trigger download via dataset class
                    from ultra.datasets_query import FB15k237LogicalQuery, NELL995LogicalQuery, FB15kLogicalQuery
                    cls_map = {
                        "FB15k237LogicalQuery": FB15k237LogicalQuery,
                        "NELL995LogicalQuery": NELL995LogicalQuery,
                        "FB15kLogicalQuery": FB15kLogicalQuery,
                    }
                    cls_map[dataset_name](root=root)

                with open(os.path.join(path, "id2ent.pkl"), "rb") as f:
                    id2ent = pickle.load(f)
                with open(os.path.join(path, "id2rel.pkl"), "rb") as f:
                    id2rel = pickle.load(f)

                triplets = []
                triplet_file = os.path.join(path, "train.txt")
                with open(triplet_file) as f:
                    for line in f:
                        h, r, t = [int(x) for x in line.split()]
                        triplets.append((h, t, r))

                train_edges = torch.tensor([[t[0], t[1]] for t in triplets], dtype=torch.long).t()
                train_edge_types = torch.tensor([t[2] for t in triplets], dtype=torch.long)
                graph = Data(
                    edge_index=train_edges,
                    edge_type=train_edge_types,
                    num_nodes=len(id2ent),
                    num_relations=len(id2rel),
                    inverse_rel_plus_one=True,
                )

                # Load easy/hard test answers if available
                easy_path = os.path.join(path, f"{split}-easy-answers.pkl")
                hard_path = os.path.join(path, f"{split}-hard-answers.pkl")
                if os.path.exists(easy_path):
                    with open(easy_path, "rb") as f:
                        easy_answers = pickle.load(f)
                if os.path.exists(hard_path):
                    with open(hard_path, "rb") as f:
                        hard_answers = pickle.load(f)

        elif hasattr(dataset, "train_graph") or hasattr(dataset, "test_graph"):
            # Existing dataset instance
            if split == "train":
                graph = dataset.train_graph
            elif split == "valid":
                graph = dataset.valid_graph
            else:
                graph = dataset.test_graph
            id2ent = getattr(dataset, "inv_entity_vocab", {})
            id2rel = getattr(dataset, "inv_relation_vocab", {})
            dataset_name = str(dataset)
        elif isinstance(dataset, Data):
            graph = dataset
        else:
            raise ValueError(f"Unsupported dataset format: {type(dataset)}")

        # 2. Build model
        model = UltraQuery(
            model=Ultra(
                rel_model_cfg={
                    "class": "RelNBFNet",
                    "input_dim": 64,
                    "hidden_dims": [64] * 6,
                    "message_func": "distmult",
                    "aggregate_func": "sum",
                    "short_cut": True,
                    "layer_norm": True,
                },
                entity_model_cfg={
                    "class": "QueryNBFNet",
                    "input_dim": 64,
                    "hidden_dims": [64] * 6,
                    "message_func": "distmult",
                    "aggregate_func": "sum",
                    "short_cut": True,
                    "layer_norm": True,
                },
            ),
            logic="product",
            threshold=threshold,
        )

        if not os.path.exists(ckpt_path):
            # Try prepending repo root if relative
            cand = os.path.join(os.path.dirname(os.path.dirname(__file__)), ckpt_path)
            if os.path.exists(cand):
                ckpt_path = cand

        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["model"])

        return cls(
            model=model,
            graph=graph,
            id2ent=id2ent,
            id2rel=id2rel,
            device=device,
            dataset_name=dataset_name,
            easy_answers=easy_answers,
            hard_answers=hard_answers,
        )

    def _resolve_term(self, val: Any, vocab_map: Dict[str, int], term_name: str) -> int:
        """Resolve a single entity or relation term (int ID or string name) to int ID."""
        if isinstance(val, int):
            return val
        s = str(val).strip()
        if s in vocab_map:
            return vocab_map[s]
        # Check without '+' or '-' prefix for relations
        if term_name == "relation":
            for prefix in ["+", "-"]:
                if prefix + s in vocab_map:
                    return vocab_map[prefix + s]
                if s.startswith(prefix) and s[1:] in vocab_map:
                    return vocab_map[s[1:]]
        # Check if s is numeric string
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
        raise ValueError(f"Unknown {term_name} '{val}'. Available examples: {list(vocab_map.keys())[:5]}")

    def resolve_query(self, nested: Any) -> Tuple:
        """
        Recursively resolve entity/relation string names into integer IDs in a BetaE nested tuple.
        Also translates string operator shortcuts ('n' -> -2, 'u' -> -1).
        """
        if not isinstance(nested, tuple):
            return self._resolve_term(nested, self.ent2id, "entity")

        # Unary operations (projection, negation): (var, (op1, op2, ...))
        if len(nested) == 2 and isinstance(nested[1], tuple) and len(nested[1]) > 0 and not isinstance(nested[1][-1], tuple):
            var, unary_ops = nested
            resolved_var = self.resolve_query(var)
            resolved_ops = []
            for op in unary_ops:
                if op in (-2, "n", "-2", "negation"):
                    resolved_ops.append(-2)
                elif op in (-1, "u", "-1", "union"):
                    resolved_ops.append(-1)
                else:
                    resolved_ops.append(self._resolve_term(op, self.rel2id, "relation"))
            return (resolved_var, tuple(resolved_ops))

        # N-ary operations (conjunction/intersection or union)
        resolved_elements = []
        for elem in nested:
            if elem in (-1, "u", "-1", "union"):
                resolved_elements.append((-1,))
            elif isinstance(elem, tuple):
                resolved_elements.append(self.resolve_query(elem))
            else:
                resolved_elements.append(self._resolve_term(elem, self.ent2id, "entity"))
        return tuple(resolved_elements)

    def query_to_string(self, numeric_query: Tuple) -> str:
        """Render a readable mathematical/logical representation of a query."""
        try:
            q = Query.from_nested(numeric_query)
            readable = q.to_readable()
            # Replace raw integer IDs with entity and relation names if available
            lines = []
            for line in readable.split("\n"):
                for ent_id, ent_name in self.id2ent.items():
                    line = line.replace(f"({ent_id})", f"({ent_name})")
                    line = line.replace(f" {ent_id}", f" {ent_name}")
                for rel_id, rel_name in self.id2rel.items():
                    line = line.replace(f"projection_{rel_id}(", f"projection_{rel_name}(")
                lines.append(line)
            return "\n".join(lines)
        except Exception:
            return str(numeric_query)

    @torch.no_grad()
    def ask(
        self,
        query: Union[Tuple, str, Query],
        top_k: int = 10,
        return_dataframe: bool = True,
    ) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Execute a single complex query and retrieve top-K ranked entity answers.

        Parameters:
            query: Nested tuple query in BetaE format (with string names or integer IDs),
                   or a string representation of a tuple, or a Query object.
            top_k: Number of ranked candidate entities to return (default: 10).
            return_dataframe: If True, returns a pandas DataFrame; otherwise a dict.

        Returns:
            pd.DataFrame or dict with ranked entities, IDs, probabilities, and logits.
        """
        if isinstance(query, str):
            # Parse string representation of tuple if passed as string
            import ast
            try:
                query = ast.literal_eval(query)
            except Exception:
                pass

        if not isinstance(query, Query):
            numeric_query = self.resolve_query(query)
            q_tensor = Query.from_nested(numeric_query)
        else:
            numeric_query = tuple()
            q_tensor = query

        q_batch = q_tensor.unsqueeze(0).to(self.device)

        # Forward pass (eval mode, symbolic traversal disabled)
        logits = self.model(self.graph, q_batch, symbolic_traversal=False)
        probs = torch.sigmoid(logits[0])

        k = min(top_k, len(probs))
        top_probs, top_indices = torch.topk(probs, k=k)
        top_logits = logits[0][top_indices]

        ranks = list(range(1, k + 1))
        entity_ids = [idx.item() for idx in top_indices]
        entity_names = [self.id2ent.get(idx, str(idx)) for idx in entity_ids]
        prob_values = [p.item() for p in top_probs]
        logit_values = [l.item() for l in top_logits]

        # Check ground truth annotations if available
        is_easy = []
        is_hard = []
        has_gt = False
        if numeric_query in self.easy_answers or numeric_query in self.hard_answers:
            has_gt = True
            easy_set = self.easy_answers.get(numeric_query, set())
            hard_set = self.hard_answers.get(numeric_query, set())
            for idx in entity_ids:
                is_easy.append(idx in easy_set)
                is_hard.append(idx in hard_set)

        data = {
            "rank": ranks,
            "entity": entity_names,
            "entity_id": entity_ids,
            "probability": prob_values,
            "logit": logit_values,
        }
        if has_gt:
            data["is_easy_answer"] = is_easy
            data["is_hard_answer"] = is_hard

        if return_dataframe:
            df = pd.DataFrame(data)
            return df
        return data

    # ---------------- Shorthand Query Builders ---------------- #

    def ask_1p(self, entity: Union[str, int], relation: Union[str, int], top_k: int = 10) -> pd.DataFrame:
        """1-hop path query: ?X : relation(entity, ?X)."""
        return self.ask((entity, (relation,)), top_k=top_k)

    def ask_2p(self, entity: Union[str, int], rel1: Union[str, int], rel2: Union[str, int], top_k: int = 10) -> pd.DataFrame:
        """2-hop path query: ?X : rel2(rel1(entity), ?X)."""
        return self.ask((entity, (rel1, rel2)), top_k=top_k)

    def ask_3p(self, entity: Union[str, int], rel1: Union[str, int], rel2: Union[str, int], rel3: Union[str, int], top_k: int = 10) -> pd.DataFrame:
        """3-hop path query: ?X : rel3(rel2(rel1(entity))), ?X)."""
        return self.ask((entity, (rel1, rel2, rel3)), top_k=top_k)

    def ask_2i(
        self,
        e1: Union[str, int],
        r1: Union[str, int],
        e2: Union[str, int],
        r2: Union[str, int],
        top_k: int = 10,
    ) -> pd.DataFrame:
        """2-hop intersection: ?X : r1(e1, ?X) AND r2(e2, ?X)."""
        return self.ask(((e1, (r1,)), (e2, (r2,))), top_k=top_k)

    def ask_3i(
        self,
        e1: Union[str, int],
        r1: Union[str, int],
        e2: Union[str, int],
        r2: Union[str, int],
        e3: Union[str, int],
        r3: Union[str, int],
        top_k: int = 10,
    ) -> pd.DataFrame:
        """3-hop intersection: ?X : r1(e1, ?X) AND r2(e2, ?X) AND r3(e3, ?X)."""
        return self.ask(((e1, (r1,)), (e2, (r2,)), (e3, (r3,))), top_k=top_k)

    def ask_2in(
        self,
        e1: Union[str, int],
        r1: Union[str, int],
        e2: Union[str, int],
        r2: Union[str, int],
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Intersection with negation: ?X : r1(e1, ?X) AND NOT r2(e2, ?X)."""
        return self.ask(((e1, (r1,)), (e2, (r2, -2))), top_k=top_k)

    def ask_2u(
        self,
        e1: Union[str, int],
        r1: Union[str, int],
        e2: Union[str, int],
        r2: Union[str, int],
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Union query: ?X : r1(e1, ?X) OR r2(e2, ?X)."""
        return self.ask(((e1, (r1,)), (e2, (r2,)), (-1,)), top_k=top_k)

    # ---------------- Inspection Helpers ---------------- #

    def search_entities(self, pattern: str, limit: int = 10) -> List[Tuple[int, str]]:
        """Search entity vocabulary by substring."""
        pattern = pattern.lower()
        results = []
        for idx, name in self.id2ent.items():
            if pattern in name.lower():
                results.append((idx, name))
                if len(results) >= limit:
                    break
        return results

    def search_relations(self, pattern: str, limit: int = 10) -> List[Tuple[int, str]]:
        """Search relation vocabulary by substring."""
        pattern = pattern.lower()
        results = []
        for idx, name in self.id2rel.items():
            if pattern in name.lower():
                results.append((idx, name))
                if len(results) >= limit:
                    break
        return results

    def sample_queries(self, query_type: str = "1p", n: int = 5) -> List[Tuple]:
        """Fetch sample queries of a given structure from the loaded dataset."""
        type2struct = {
            "1p": ("e", ("r",)),
            "2p": ("e", ("r", "r")),
            "3p": ("e", ("r", "r", "r")),
            "2i": (("e", ("r",)), ("e", ("r",))),
            "3i": (("e", ("r",)), ("e", ("r",)), ("e", ("r",))),
            "2in": (("e", ("r",)), ("e", ("r", "n"))),
            "3in": (("e", ("r",)), ("e", ("r",)), ("e", ("r", "n"))),
            "2u": (("e", ("r",)), ("e", ("r",)), ("u",)),
        }
        struct = type2struct.get(query_type)
        if struct is None or not self.easy_answers:
            return []
        matching = [q for q in self.easy_answers.keys() if isinstance(q, tuple)]
        return matching[:n]

