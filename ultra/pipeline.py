import os
import pickle
import urllib.request
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch_geometric.data import Data

from ultra.models import Ultra
from ultra.nl_parser import NLQueryParser
from ultra.query_utils import Query
from ultra.tasks import build_relation_graph
from ultra.ultraquery import UltraQuery


class UltraQueryPipeline:
    """
    High-level pipeline for asking complex logical and natural language queries against knowledge graphs.
    
    Supports:
      - Natural language questions via .ask_natural("What films did Christopher Nolan direct?")
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
        ent2text: Optional[Dict[str, str]] = None,
        rel2text: Optional[Dict[str, str]] = None,
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
        
        self.ent2text = ent2text or {}
        self.rel2text = rel2text or {}

        # Human-readable entity names
        self.id2name = {eid: self.ent2text.get(mid, mid) for eid, mid in self.id2ent.items()}

        # Natural Language Parser (multilingual-e5 + jieba)
        self.nl_parser = NLQueryParser(self.id2ent, self.id2rel, self.ent2text, self.rel2text, device=self.device)

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
        ent2text = {}
        rel2text = {}
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

                # Check / download human-readable entity and relation names for FB15k-237
                ent2text_file = os.path.join(path, "entity2text.txt")
                rel2text_file = os.path.join(path, "relation2text.txt")
                if "237" in dir_name:
                    if not os.path.exists(ent2text_file):
                        try:
                            url = "https://raw.githubusercontent.com/yao8839836/kg-bert/master/data/FB15k-237/entity2text.txt"
                            urllib.request.urlretrieve(url, ent2text_file)
                        except Exception:
                            pass
                    if not os.path.exists(rel2text_file):
                        try:
                            url = "https://raw.githubusercontent.com/yao8839836/kg-bert/master/data/FB15k-237/relation2text.txt"
                            urllib.request.urlretrieve(url, rel2text_file)
                        except Exception:
                            pass

                if os.path.exists(ent2text_file):
                    with open(ent2text_file, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split("\t")
                            if len(parts) >= 2:
                                ent2text[parts[0].strip()] = parts[1].strip()

                if os.path.exists(rel2text_file):
                    with open(rel2text_file, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split("\t")
                            if len(parts) >= 2:
                                rel2text[parts[0].strip()] = parts[1].strip()

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

                # Optionally load test answers for ground-truth checks
                easy_path = os.path.join(path, "test-easy-answers.pkl")
                hard_path = os.path.join(path, "test-hard-answers.pkl")
                if os.path.exists(easy_path):
                    with open(easy_path, "rb") as f:
                        easy_answers = pickle.load(f)
                if os.path.exists(hard_path):
                    with open(hard_path, "rb") as f:
                        hard_answers = pickle.load(f)

        elif hasattr(dataset, "edge_index"):
            graph = dataset
            id2ent = getattr(dataset, "inv_entity_vocab", {i: str(i) for i in range(graph.num_nodes)})
            id2rel = getattr(dataset, "inv_relation_vocab", {i: str(i) for i in range(graph.num_relations)})
        else:
            raise ValueError(f"Unsupported dataset format: {type(dataset)}")

        # 2. Build model architecture
        base_model = Ultra(
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
        )

        model = UltraQuery(
            model=base_model,
            logic="product",
            threshold=threshold,
        )

        # 3. Load checkpoint
        ckpt_path = os.path.expanduser(ckpt_path)
        if not os.path.isabs(ckpt_path):
            candidates = [
                ckpt_path,
                os.path.join(os.path.dirname(os.path.dirname(__file__)), ckpt_path),
                os.path.join("/root/ULTRA", ckpt_path),
            ]
            for c in candidates:
                if os.path.exists(c):
                    ckpt_path = c
                    break

        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")
            if "model" in state:
                model.load_state_dict(state["model"], strict=False)
            else:
                model.load_state_dict(state, strict=False)

        return cls(
            model=model,
            graph=graph,
            id2ent=id2ent,
            id2rel=id2rel,
            ent2text=ent2text,
            rel2text=rel2text,
            device=device,
            dataset_name=dataset_name,
            easy_answers=easy_answers,
            hard_answers=hard_answers,
        )

    def _resolve_term(self, term: Any, vocab: Dict[str, int], name: str) -> int:
        """Resolve a string or int to a vocabulary index."""
        if isinstance(term, int):
            return term
        if isinstance(term, str):
            if term in vocab:
                return vocab[term]
            norm_term = term.strip()
            if norm_term in vocab:
                return vocab[norm_term]
            for prefix in ["+", "-"]:
                alt = prefix + norm_term.lstrip("+-")
                if alt in vocab:
                    return vocab[alt]
        raise KeyError(f"Unknown {name}: '{term}'. Available count: {len(vocab)}")

    def resolve_query(self, nested: Tuple) -> Tuple:
        """Recursively resolve entity and relation string names to integer IDs."""
        if len(nested) == 2 and isinstance(nested[1], tuple) and (len(nested[1]) == 0 or not isinstance(nested[1][-1], tuple)):
            entity, relations = nested
            entity_id = self._resolve_term(entity, self.ent2id, "entity")
            resolved_rel = []
            for r in relations:
                if r == -1 or r == "n":
                    resolved_rel.append(-1)
                else:
                    resolved_rel.append(self._resolve_term(r, self.rel2id, "relation"))
            return (entity_id, tuple(resolved_rel))

        resolved_elements = []
        for elem in nested:
            if elem == -1 or elem == -2 or elem == "u":
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

        logits = self.model(self.graph, q_batch, symbolic_traversal=False)
        probs = torch.sigmoid(logits[0])

        k = min(top_k, len(probs))
        top_probs, top_indices = torch.topk(probs, k=k)
        top_logits = logits[0][top_indices]

        ranks = list(range(1, k + 1))
        entity_ids = [idx.item() for idx in top_indices]
        # Human-readable names if available, fallback to MID / ID
        entity_names = [self.id2name.get(idx, self.id2ent.get(idx, str(idx))) for idx in entity_ids]
        entity_mids = [self.id2ent.get(idx, str(idx)) for idx in entity_ids]
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
            "entity_mid": entity_mids,
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

    # ---------------- Natural Language Interface ---------------- #

    def parse_natural(self, question: str) -> Dict[str, Any]:
        """Parse a natural language question into a BetaE logical query."""
        return self.nl_parser.parse(question)

    def ask_natural(
        self,
        question: str,
        top_k: int = 10,
        return_dataframe: bool = True,
        explain: bool = True,
    ) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Answer a naturally phrased question against the knowledge graph.

        Parameters:
            question: Natural language question (e.g. "What films did Christopher Nolan direct?")
            top_k: Number of candidate answers to retrieve.
            return_dataframe: Return a pandas DataFrame if True.
            explain: If True, prints parsing diagnostics and a natural language answer summary.
        """
        parsed = self.parse_natural(question)
        if explain:
            print(f"\n" + "-" * 70)
            print(f" Natural Language Question: \"{parsed['question']}\"")
            print(f"-" * 70)
            print(f"  Query Pattern : {parsed['query_type']}")
            print(f"  Logical AST   : {parsed['logical_query']}")
            print(f"  Explanation   : {parsed['explanation']}")
            print(f"-" * 70)

        df = self.ask(parsed["logical_query"], top_k=top_k, return_dataframe=True)

        if explain and isinstance(df, pd.DataFrame) and len(df) > 0:
            top_answers = [f"{row['entity']} (p={row['probability']:.3f})" for _, row in df.head(3).iterrows()]
            print(f"  Top Answers   : {', '.join(top_answers)}\n")

        if not return_dataframe:
            return df.to_dict(orient="records")
        return df

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
        return self.ask(((e1, (r1,)), (e2, (r2, -1))), top_k=top_k)

    def ask_2u(
        self,
        e1: Union[str, int],
        r1: Union[str, int],
        e2: Union[str, int],
        r2: Union[str, int],
        top_k: int = 10,
    ) -> pd.DataFrame:
        """Union query: ?X : r1(e1, ?X) OR r2(e2, ?X)."""
        return self.ask(((e1, (r1,)), (e2, (r2,)), -2), top_k=top_k)

    # ---------------- Inspection Helpers ---------------- #

    def search_entities(self, pattern: str, limit: int = 10) -> List[Tuple[int, str, str]]:
        """Search entity vocabulary by substring."""
        pattern = pattern.lower()
        results = []
        for idx, mid in self.id2ent.items():
            name = self.id2name.get(idx, mid)
            if pattern in name.lower() or pattern in mid.lower():
                results.append((idx, name, mid))
                if len(results) >= limit:
                    break
        return results

    def search_relations(self, pattern: str, limit: int = 10) -> List[Tuple[int, str, str]]:
        """Search relation vocabulary by substring."""
        pattern = pattern.lower()
        results = []
        for idx, uri in self.id2rel.items():
            desc = self.rel2text.get(uri.lstrip("+-"), uri)
            if pattern in desc.lower() or pattern in uri.lower():
                results.append((idx, desc, uri))
                if len(results) >= limit:
                    break
        return results
