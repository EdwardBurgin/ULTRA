import os
from typing import Dict, List, Optional, Tuple, Union

import torch


# High-precision canonical descriptors for common relations
CANONICAL_RELATION_DESCRIPTORS = {
    "+/film/director/film": "films directed by, directed by, directed, movie director, film director, direct, 导演, 执导, 导演的作品",
    "-/film/director/film": "director of film, directed by whom, 电影的导演, 谁导演的",
    "+/film/actor/film./film/performance/film": "movies starring, films starring, movies star, starred in, actor in, starring, star, cast member, 主演, 出演, 参演, 扮演, 演的电影",
    "-/film/actor/film./film/performance/film": "actors in the film, cast of movie, 电影的主演, 演员表",
    "+/people/person/place_of_birth": "place of birth, born in, birthplace, where was born, 出生地, 出生于, 出生在哪里, 籍贯",
    "-/people/person/place_of_birth": "people born here, birthplace of, 在这里出生的人",
    "+/people/person/nationality": "nationality, country of citizenship, citizen, what nationality, 国籍, 国家, 哪国人",
    "-/people/person/nationality": "citizens of country, 该国公民",
    "+/people/person/profession": "profession, occupation, career, job, 职业, 从事什么工作",
    "+/education/educational_institution/students_graduates./education/education/major_field_of_study": "major, field of study, degree, studied, 专业, 主修, 研究领域",
    "+/film/film/genre": "film genre, movie category, style, 电影流派, 类型, 题材",
}


class MultilingualRelationMatcher:
    """
    Cross-lingual relation matcher using dense multilingual embeddings (intfloat/multilingual-e5-small).
    
    Encodes knowledge graph relations and semantic relation queries (Chinese, English, etc.)
    into a unified embedding space to perform nearest-neighbor semantic search.
    """

    def __init__(
        self,
        id2rel: Dict[int, str],
        rel2text: Optional[Dict[str, str]] = None,
        model_name: str = "intfloat/multilingual-e5-small",
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.id2rel = id2rel
        self.rel2text = rel2text or {}
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.model_name = model_name
        self._model = None
        self._relation_ids: List[int] = []
        self._relation_uris: List[str] = []
        self._relation_texts: List[str] = []
        self._relation_embeddings: Optional[torch.Tensor] = None

        self._build_relation_catalog()

    def _build_relation_catalog(self):
        """Prepare textual descriptions for all relations with cross-lingual concept anchors."""
        self._relation_ids = []
        self._relation_uris = []
        self._relation_texts = []

        for rid, uri in sorted(self.id2rel.items()):
            self._relation_ids.append(rid)
            self._relation_uris.append(uri)
            
            if uri in CANONICAL_RELATION_DESCRIPTORS:
                desc = CANONICAL_RELATION_DESCRIPTORS[uri]
            else:
                clean_uri = uri.lstrip("+-")
                desc = self.rel2text.get(clean_uri, clean_uri.replace("/", " ").replace(".", " ").replace("_", " "))
                if uri.startswith("-"):
                    desc = f"reverse relation of {desc}"
            
            passage_text = f"passage: {desc}"
            self._relation_texts.append(passage_text)

    def _ensure_embeddings(self):
        """Load model and compute relation embeddings if not already computed."""
        if self._relation_embeddings is not None:
            return

        from sentence_transformers import SentenceTransformer

        if self._model is None:
            self._model = SentenceTransformer(self.model_name, device=self.device)

        with torch.no_grad():
            self._relation_embeddings = self._model.encode(
                self._relation_texts,
                batch_size=64,
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=self.device,
            )

    def match(self, query_text: str, top_k: int = 3, prefer_forward: bool = True) -> List[Tuple[int, str, float]]:
        """
        Matches a natural language query phrase (in Chinese, English, etc.)
        against the relation catalog.
        
        Returns a ranked list of (relation_id, relation_uri, similarity_score).
        """
        self._ensure_embeddings()

        query_input = f"query: {query_text.strip()}"

        with torch.no_grad():
            query_embedding = self._model.encode(
                [query_input],
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=self.device,
            )[0]

            sims = torch.mv(self._relation_embeddings, query_embedding)
            
            # If prefer_forward is requested, slightly boost forward relations (+/...)
            if prefer_forward:
                forward_mask = torch.tensor([not uri.startswith("-") for uri in self._relation_uris], device=self.device)
                sims = torch.where(forward_mask, sims + 0.02, sims)

            top_vals, top_idxs = torch.topk(sims, k=min(top_k, len(sims)))

            results = []
            for val, idx in zip(top_vals, top_idxs):
                rid = self._relation_ids[idx.item()]
                uri = self._relation_uris[idx.item()]
                score = val.item()
                results.append((rid, uri, score))

            return results

    def find_best_relation(self, query_text: str, threshold: float = 0.6, prefer_forward: bool = True) -> Optional[Tuple[int, str, float]]:
        """Find single best matching relation if score >= threshold."""
        matches = self.match(query_text, top_k=1, prefer_forward=prefer_forward)
        if matches and matches[0][2] >= threshold:
            return matches[0]
        return None
