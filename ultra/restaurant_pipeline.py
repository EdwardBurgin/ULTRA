import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch_geometric.data import Data

from ultra.export_utils import dataframe_to_graph
from ultra.models import Ultra
from ultra.query_utils import Query
from ultra.tasks import build_relation_graph
from ultra.ultraquery import UltraQuery


class ShenzhenRestaurantPipeline:
    """
    Dedicated pipeline for Chinese Shenzhen Restaurant Review Knowledge Graphs.
    
    Given:
      - A graph DataFrame with columns: ['id', 'subject', 'predicate', 'object']
      - Synthetic 2-hop questions of the form: (x) [p1] (o1) [p2] (o2)
    
    The pipeline:
      1. Extracts entities o1 and o2 from the question text.
      2. Discovers the starting triple (o1) [p2] (o2) and its graph 'id'.
      3. Passes the reasoning query into the UltraQuery foundation model on the custom graph.
      4. Solves for the missing entity x.
      5. Retrieves the exact graph 'id' of the target triple (x) [p1] (o1).
      6. Exports all structured results to an Excel (.xlsx) file.
    """

    def __init__(
        self,
        graph_df: pd.DataFrame,
        ckpt_path: str = "ckpts/ultraquery.pth",
        device: Optional[Union[str, torch.device]] = None,
        threshold: float = 0.0,
    ):
        self.device = torch.device(device if device else ("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.threshold = threshold

        # Ensure required columns
        required_cols = {"id", "subject", "predicate", "object"}
        missing = required_cols - set(graph_df.columns)
        if missing:
            raise ValueError(f"graph_df is missing required columns: {missing}")

        self.graph_df = graph_df.copy()
        # Ensure string types for entity and relation names
        self.graph_df["subject"] = self.graph_df["subject"].astype(str).str.strip()
        self.graph_df["predicate"] = self.graph_df["predicate"].astype(str).str.strip()
        self.graph_df["object"] = self.graph_df["object"].astype(str).str.strip()

        # Build fast lookup indexes
        self._build_indexes()

        # Build UltraQuery graph representation
        self._build_ultra_graph()

        # Load UltraQuery neural reasoner
        self._load_model(ckpt_path)

    def _build_indexes(self):
        """Construct fast lookup maps from the graph DataFrame."""
        self.triple2id: Dict[Tuple[str, str, str], Any] = {}
        self.pair2triples: Dict[Tuple[str, str], List[Tuple[str, Any]]] = {}
        self.incoming_to_obj: Dict[str, List[Tuple[str, str, Any]]] = {}
        self.outgoing_from_subj: Dict[str, List[Tuple[str, str, Any]]] = {}

        for _, row in self.graph_df.iterrows():
            tid = row["id"]
            s = row["subject"]
            p = row["predicate"]
            o = row["object"]

            self.triple2id[(s, p, o)] = tid

            if (s, o) not in self.pair2triples:
                self.pair2triples[(s, o)] = []
            self.pair2triples[(s, o)].append((p, tid))

            if o not in self.incoming_to_obj:
                self.incoming_to_obj[o] = []
            self.incoming_to_obj[o].append((s, p, tid))

            if s not in self.outgoing_from_subj:
                self.outgoing_from_subj[s] = []
            self.outgoing_from_subj[s].append((o, p, tid))

        # Distinct entities sorted by length descending for greedy substring matching
        unique_entities = set(self.graph_df["subject"]).union(set(self.graph_df["object"]))
        self.sorted_entities = sorted(unique_entities, key=lambda x: len(x), reverse=True)

        # Distinct predicates
        self.predicates = sorted(self.graph_df["predicate"].unique(), key=lambda x: len(x), reverse=True)

    def _build_ultra_graph(self):
        """Convert graph DataFrame into an ULTRA PyG graph object with inverse relations."""
        # Convert using dataframe_to_graph
        pyg_data, ent_vocab, rel_vocab = dataframe_to_graph(
            self.graph_df,
            head_col="subject",
            rel_col="predicate",
            tail_col="object",
            add_inverse_edges=True,
            build_rel_graph=True,
        )

        self.graph = pyg_data.to(self.device)
        self.ent2id = ent_vocab
        self.id2ent = {v: k for k, v in ent_vocab.items()}
        self.rel2id = rel_vocab
        self.id2rel = {v: k for k, v in rel_vocab.items()}

        self.num_base_relations = len(rel_vocab)
        self.num_nodes = len(ent_vocab)

    def _load_model(self, ckpt_path: str):
        """Load pre-trained UltraQuery model."""
        if not os.path.isabs(ckpt_path):
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            full_ckpt = os.path.join(repo_root, ckpt_path)
            if os.path.exists(full_ckpt):
                ckpt_path = full_ckpt

        print(f"[ShenzhenRestaurantPipeline] Loading UltraQuery checkpoint from {ckpt_path}...")
        state = torch.load(ckpt_path, map_location="cpu")
        model_state = state.get("model", state)

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
        self.model = UltraQuery(
            model=base_model,
            logic="product",
            threshold=self.threshold,
        )
        self.model.load_state_dict(model_state, strict=False)
        self.model.to(self.device)
        self.model.eval()
        print(f"[ShenzhenRestaurantPipeline] Initialized on {self.device} with {self.num_nodes} nodes, {self.num_base_relations} base relations.")

    def extract_entities_and_starting_triple(self, question: str) -> Dict[str, Any]:
        """
        Extracts candidate entities o1 and o2 from question text and resolves
        the starting triple (o1) [p2] (o2) and its ID.
        """
        q = question.strip()
        
        # 1. Greedy entity detection
        matched_spans = []
        for ent in self.sorted_entities:
            if len(ent) < 2:
                continue
            # Look for entity mention in question
            idx = 0
            while True:
                pos = q.find(ent, idx)
                if pos == -1:
                    break
                s, e = pos, pos + len(ent)
                # Ensure no overlapping with previously matched longer entities
                if not any(not (e <= ms or s >= me) for ms, me, _ in matched_spans):
                    matched_spans.append((s, e, ent))
                idx = pos + 1

        matched_spans.sort(key=lambda x: x[0])
        candidate_entities = [m[2] for m in matched_spans]

        # 2. Find the pair (o1, o2) that forms a valid starting triple in the graph
        # In (x) [p1] (o1) [p2] (o2), the starting triple is (o1, p2, o2)
        valid_pairs = []
        for i, ent_a in enumerate(candidate_entities):
            for j, ent_b in enumerate(candidate_entities):
                if i == j:
                    continue
                if (ent_a, ent_b) in self.pair2triples:
                    for pred, tid in self.pair2triples[(ent_a, ent_b)]:
                        valid_pairs.append({
                            "o1": ent_a,
                            "p2": pred,
                            "o2": ent_b,
                            "starting_triple_id": tid,
                        })

        if not valid_pairs:
            # Fallback: check if reverse pair exists
            for i, ent_a in enumerate(candidate_entities):
                for j, ent_b in enumerate(candidate_entities):
                    if i == j:
                        continue
                    if (ent_b, ent_a) in self.pair2triples:
                        for pred, tid in self.pair2triples[(ent_b, ent_a)]:
                            valid_pairs.append({
                                "o1": ent_b,
                                "p2": pred,
                                "o2": ent_a,
                                "starting_triple_id": tid,
                            })

        if not valid_pairs:
            raise ValueError(
                f"Could not find a valid starting triple (o1, p2, o2) in question: '{question}'. "
                f"Matched candidate entities: {candidate_entities}"
            )

        # Return the best matching pair
        return valid_pairs[0]

    def resolve_p1(self, question: str, o1: str) -> Optional[str]:
        """
        Determines predicate p1 connecting x to o1 in (x) [p1] (o1).
        Checks candidate incoming predicates to o1 and finds mention in question.
        """
        incoming = self.incoming_to_obj.get(o1, [])
        candidate_preds = list({p for _, p, _ in incoming})

        # Match against question text
        for pred in candidate_preds:
            if pred in question:
                return pred

        # If only one incoming predicate type exists for o1, use it
        if len(candidate_preds) == 1:
            return candidate_preds[0]

        # Check all graph predicates
        for pred in self.predicates:
            if pred in question and pred in candidate_preds:
                return pred

        if candidate_preds:
            return candidate_preds[0]

        return None

    @torch.no_grad()
    def query_ultra(
        self,
        anchor_name: str,
        rel_chain_names: List[str],
        top_k: int = 5,
    ) -> List[Tuple[str, float, float]]:
        """
        Executes a neural logical query on the custom graph using UltraQuery.
        Returns ranked list of (entity_name, probability, logit).
        """
        if anchor_name not in self.ent2id:
            raise ValueError(f"Anchor entity '{anchor_name}' not in graph vocabulary.")

        anchor_id = self.ent2id[anchor_name]
        numeric_rels = []
        for r_name in rel_chain_names:
            if r_name.startswith("inv:"):
                base_r = r_name[4:]
                if base_r not in self.rel2id:
                    raise ValueError(f"Relation '{base_r}' not in graph relation vocabulary.")
                # Inverse relation ID in ULTRA = base_id + num_base_relations
                numeric_rels.append(self.rel2id[base_r] + self.num_base_relations)
            else:
                if r_name not in self.rel2id:
                    raise ValueError(f"Relation '{r_name}' not in graph relation vocabulary.")
                numeric_rels.append(self.rel2id[r_name])

        query_tuple = (anchor_id, tuple(numeric_rels))
        q_tensor = Query.from_nested(query_tuple)
        q_batch = q_tensor.unsqueeze(0).to(self.device)

        logits = self.model(self.graph, q_batch, symbolic_traversal=False)
        probs = torch.sigmoid(logits[0])

        k = min(top_k, len(probs))
        top_probs, top_indices = torch.topk(probs, k=k)
        top_logits = logits[0][top_indices]

        results = []
        for p, idx, l in zip(top_probs, top_indices, top_logits):
            ent_name = self.id2ent[idx.item()]
            results.append((ent_name, p.item(), l.item()))

        return results

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Answers a 2-hop question (x) [p1] (o1) [p2] (o2) to find missing entity x
        and retrieve the graph ID of triple (x) [p1] (o1).
        """
        # Step 1: Extract o1, p2, o2, and starting triple id
        start_info = self.extract_entities_and_starting_triple(question)
        o1 = start_info["o1"]
        p2 = start_info["p2"]
        o2 = start_info["o2"]
        starting_triple_id = start_info["starting_triple_id"]

        # Step 2: Determine p1
        p1 = self.resolve_p1(question, o1)
        if p1 is None:
            raise ValueError(f"Could not determine predicate p1 for object '{o1}' in question: '{question}'")

        # Step 3: Run UltraQuery neural reasoner
        # Query 1: 1-hop reverse from o1: o1 -> p1_inv -> x
        top_predictions_1hop = self.query_ultra(anchor_name=o1, rel_chain_names=[f"inv:{p1}"], top_k=5)

        # Query 2: 2-hop reverse from o2: o2 -> p2_inv -> o1 -> p1_inv -> x
        top_predictions_2hop = self.query_ultra(anchor_name=o2, rel_chain_names=[f"inv:{p2}", f"inv:{p1}"], top_k=5)

        # Pick top predicted x
        # Check against ground truth incoming edges into o1 to verify exact match
        incoming = self.incoming_to_obj.get(o1, [])
        valid_x_triples = {s: tid for s, p, tid in incoming if p == p1}

        predicted_x = None
        target_triple_id = None
        confidence = 0.0

        # Prioritize candidates from 1-hop and 2-hop predictions
        for candidate_x, prob, _ in top_predictions_1hop:
            if candidate_x in valid_x_triples:
                predicted_x = candidate_x
                target_triple_id = valid_x_triples[candidate_x]
                confidence = prob
                break

        if predicted_x is None and top_predictions_1hop:
            predicted_x = top_predictions_1hop[0][0]
            confidence = top_predictions_1hop[0][1]
            if (predicted_x, p1, o1) in self.triple2id:
                target_triple_id = self.triple2id[(predicted_x, p1, o1)]

        return {
            "question": question,
            "o1": o1,
            "p2": p2,
            "o2": o2,
            "starting_triple_id": starting_triple_id,
            "p1": p1,
            "predicted_x": predicted_x,
            "target_triple_id": target_triple_id,
            "confidence": confidence,
            "status": "SUCCESS" if (predicted_x and target_triple_id is not None) else "PARTIAL",
        }

    def process_questions(
        self,
        questions_df: pd.DataFrame,
        output_xlsx_path: str = "shenzhen_2hop_results.xlsx",
    ) -> pd.DataFrame:
        """
        Processes a DataFrame containing a 'question' column and writes the results to an Excel file.
        """
        if "question" not in questions_df.columns:
            raise ValueError("questions_df must contain a 'question' column.")

        results = []
        for idx, row in questions_df.iterrows():
            q = str(row["question"])
            try:
                res = self.answer_question(q)
                results.append(res)
            except Exception as e:
                results.append({
                    "question": q,
                    "o1": None,
                    "p2": None,
                    "o2": None,
                    "starting_triple_id": None,
                    "p1": None,
                    "predicted_x": None,
                    "target_triple_id": None,
                    "confidence": 0.0,
                    "status": f"ERROR: {str(e)}",
                })

        res_df = pd.DataFrame(results)

        # Save to Excel
        output_dir = os.path.dirname(os.path.abspath(output_xlsx_path))
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        res_df.to_excel(output_xlsx_path, index=False, engine="openpyxl")
        print(f"[ShenzhenRestaurantPipeline] Results successfully written to {output_xlsx_path}")

        return res_df
