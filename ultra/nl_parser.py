import re
from typing import Any, Dict, List, Optional, Tuple, Union

import jieba
import torch

from ultra.relation_matcher import MultilingualRelationMatcher


# Schema stopwords to ignore during entity linking in English and Chinese
SCHEMA_STOP_WORDS = {
    # English
    "star", "stars", "starring", "starred", "major", "majors", "majored", "director",
    "directors", "directed", "actor", "actors", "actress", "actresses", "film", "films",
    "movie", "movies", "study", "studies", "education", "field", "fields", "degree",
    "degrees", "student", "students", "born", "birth", "birthplace", "nationality",
    "citizen", "citizenship", "award", "awards", "genre", "genres", "music", "author",
    "authors", "written", "writer", "writers", "country", "countries", "city", "cities",
    "place", "places", "what", "which", "where", "who", "when", "how", "whose", "whom",
    "both", "either", "neither", "also", "well", "name", "names", "kind", "kinds", "type",
    "types", "one", "ones", "associated", "located", "contained", "people", "person",
    
    # Chinese
    "电影", "影片", "作品", "哪部", "哪些", "哪个", "谁", "什么", "哪里", "何处",
    "导演", "执导", "主演", "出演", "扮演", "参演", "出生", "出生地", "出生于",
    "国籍", "专业", "就读", "毕业", "学院", "大学", "学生", "学者", "奖项", "获得",
    "并且", "而且", "且", "和", "与", "同", "或者", "或", "还是", "不是", "并非",
    "未曾", "未由", "不曾", "没有", "排除", "不包括", "的", "了", "在", "是", "由"
}

# Regex to detect negation connectives across Chinese and English
NEG_CONNECTIVES_REGEX = (
    r"(?:"
    r"\b(?:and|but)\s+(?:were\s+|are\s+|is\s+|was\s+|did\s+)?not\b"
    r"|\bwithout\b|\bexcluding\b"
    r"|且并非由|且并非|并且不是由|并且不是|而不是由|而不是|并非由|并非|不是由|不是|未由|未曾|不曾|排除|不包括"
    r")"
)

# Regex to detect conjunction / intersection connectives
CONJ_CONNECTIVES_REGEX = (
    r"(?:"
    r"\b(?:and|both|as well as)\b"
    r"|并且|而且|同时|且|和|与"
    r")"
)

# Regex to detect disjunction / union connectives
DISJ_CONNECTIVES_REGEX = (
    r"(?:"
    r"\b(?:or|either)\b"
    r"|或者|或|还是"
    r")"
)

# Canonical intent lexicon for instant high-precision relation resolution
INTENT_LEXICON = [
    # Actor / starring
    (r"(?:主演|出演|参演|扮演|演员|starring|starred|star\b|stars\b|actor|actress)", "+/film/actor/film./film/performance/film"),
    # Director / directed
    (r"(?:导演|执导|directed|director|direct\b)", "+/film/director/film"),
    # Place of birth
    (r"(?:出生地|出生在|出生于|出生|born|birthplace|place of birth)", "+/people/person/place_of_birth"),
    # Nationality
    (r"(?:国籍|哪国人|nationality|citizenship|citizen)", "+/people/person/nationality"),
    # Field of study / Major
    (r"(?:专业|主修|研究领域|major|field of study|studied)", "+/education/educational_institution/students_graduates./education/education/major_field_of_study"),
    # Genre
    (r"(?:流派|风格|题材|类型|genre)", "+/film/film/genre"),
]


class NLQueryParser:
    """
    Generic Multilingual Query Parser for Knowledge Graph Reasoning.
    
    Translates natural language questions in Chinese, English, and other languages
    into formal BetaE First-Order Logical (FOL) query structures.
    
    Key Features:
      - Generic Entity Extraction: Matches graph entities directly with CJK-safe boundary matching.
      - Cross-Lingual Semantic Matching: Hybrid resolution combining canonical multilingual intent
        lexicon with dense intfloat/multilingual-e5-small embeddings for zero-shot generalization.
      - Multilingual Connective Lexicon: Accurately classifies and splits 2in, 2i, 2u, 2p, and 1p queries.
    """

    def __init__(
        self,
        id2ent: Dict[int, str],
        id2rel: Dict[int, str],
        ent2text: Optional[Dict[str, str]] = None,
        rel2text: Optional[Dict[str, str]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.id2ent = id2ent
        self.id2rel = id2rel
        self.ent2id = {v: k for k, v in id2ent.items()}
        self.rel2id = {v: k for k, v in id2rel.items()}
        
        self.ent2text = ent2text or {}
        self.rel2text = rel2text or {}
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        # Dense cross-lingual relation matcher (multilingual-e5)
        self.relation_matcher = MultilingualRelationMatcher(
            id2rel=self.id2rel,
            rel2text=self.rel2text,
            device=self.device,
        )

        # Build generic entity lookup dictionary from whatever entities exist in the graph
        self.name2id: Dict[str, int] = {}
        for mid, name in self.ent2text.items():
            clean_mid = mid.replace("_zh", "")
            if clean_mid in self.ent2id:
                eid = self.ent2id[clean_mid]
                self.name2id[name.lower().strip()] = eid

        # Add raw MIDs / node IDs
        for mid, eid in self.ent2id.items():
            self.name2id[mid.lower().strip()] = eid

        # Sort candidate entity names by length descending for greedy matching
        self._sorted_names = sorted(self.name2id.keys(), key=lambda x: len(x), reverse=True)

    def find_entities(self, text: str) -> List[Tuple[int, str, int, int]]:
        """
        Extracts entity mentions from natural text (English, Chinese, etc.).
        Returns list of (entity_id, matched_text, start_idx, end_idx).
        """
        text_lower = text.lower()
        matches = []
        occupied_spans = []

        # 1. Check quotes first: "...", '...', 《...》 (Chinese book/film quotes)
        quoted = re.finditer(r'["\'《]([^"\'》]+)["\'》]', text)
        for m in quoted:
            phrase = m.group(1).lower().strip()
            if phrase in self.name2id:
                eid = self.name2id[phrase]
                matches.append((eid, m.group(1), m.start(1), m.end(1)))
                occupied_spans.append((m.start(1), m.end(1)))

        # 2. Greedy longest matching over the graph's entity catalog with CJK-safe boundaries
        for candidate in self._sorted_names:
            if candidate in SCHEMA_STOP_WORDS or len(candidate) < 2:
                continue

            # For Latin / English text, use ASCII boundary assertions; for CJK text, match direct substrings
            is_cjk = any('\u4e00' <= ch <= '\u9fff' for ch in candidate)
            if is_cjk:
                pattern = re.escape(candidate)
            else:
                pattern = r'(?<![a-zA-Z0-9])' + re.escape(candidate) + r'(?![a-zA-Z0-9])'

            for m in re.finditer(pattern, text_lower):
                s, e = m.start(), m.end()
                if any(not (e <= os_ or s >= oe_) for os_, oe_ in occupied_spans):
                    continue

                eid = self.name2id[candidate]
                orig_text = text[s:e]
                matches.append((eid, orig_text, s, e))
                occupied_spans.append((s, e))

        # 3. Chinese word segmentation with jieba as an entity discovery fallback
        has_chinese = any('\u4e00' <= ch <= '\u9fff' for ch in text)
        if has_chinese and not matches:
            tokens = list(jieba.cut(text))
            for tok in tokens:
                tok_clean = tok.lower().strip()
                if tok_clean in self.name2id and tok_clean not in SCHEMA_STOP_WORDS:
                    eid = self.name2id[tok_clean]
                    matches.append((eid, tok, 0, len(tok)))

        matches.sort(key=lambda x: x[2])
        return matches

    def find_relation(self, text: str, min_score: float = 0.55) -> Optional[Tuple[int, str]]:
        """
        Extracts relation intent using hybrid resolution:
        1. Fast, exact matching via multilingual intent lexicon.
        2. Dense multilingual-e5 embeddings for unseen / generalized relation phrases.
        """
        text_clean = text.lower()
        
        # 1. Check canonical intent lexicon
        for pattern, uri in INTENT_LEXICON:
            if re.search(pattern, text_clean):
                if uri in self.rel2id:
                    return (self.rel2id[uri], uri)

        # 2. Dense multilingual-e5 fallback
        clean_text = re.sub(r"[？?！!，,。.\-_/\\\"'《》]", " ", text).strip()
        results = self.relation_matcher.match(clean_text, top_k=1)
        if results:
            rid, uri, score = results[0]
            if score >= min_score:
                return (rid, uri)

        return None

    def parse(self, question: str) -> Dict[str, Any]:
        """
        Parses a natural language question (Chinese, English, etc.) into a BetaE logical query.
        """
        q = question.strip()
        q_lower = q.lower()

        # -------------------------------------------------------------
        # 1. Detect Negation Conjunction (2in)
        # e.g., "哪些电影由克里斯蒂安·贝尔主演且并非由克里斯托弗·诺兰导演？"
        # e.g., "Which movies star Christian Bale and were not directed by Christopher Nolan?"
        # -------------------------------------------------------------
        neg_split = re.split(NEG_CONNECTIVES_REGEX, q_lower, maxsplit=1)
        if len(neg_split) == 2:
            part1, part2 = neg_split
            ents1 = self.find_entities(part1)
            ents2 = self.find_entities(part2)

            if ents1 and ents2:
                e1, e1_name = max(ents1, key=lambda x: len(x[1]))[:2]
                e2, e2_name = max(ents2, key=lambda x: len(x[1]))[:2]
                
                # Strip entity mentions before relation extraction to eliminate semantic bias
                p1_rel_text = part1.replace(e1_name.lower(), " ")
                p2_rel_text = part2.replace(e2_name.lower(), " ")
                
                rel1 = self.find_relation(p1_rel_text)
                rel2 = self.find_relation(p2_rel_text)

                if rel1 and rel2:
                    r1, r1_uri = rel1
                    r2, r2_uri = rel2
                    # BetaE 2in: ((e1, (r1,)), (e2, (r2, -2)))
                    query_tuple = ((e1, (r1,)), (e2, (r2, -2)))
                    return {
                        "question": question,
                        "query_type": "2in",
                        "logical_query": query_tuple,
                        "entities": [(e1, e1_name), (e2, e2_name)],
                        "relations": [(r1, r1_uri), (r2, r2_uri)],
                        "explanation": f"Intersection with Negation: ?X satisfies {r1_uri}({e1_name}, ?X) AND NOT {r2_uri}({e2_name}, ?X)"
                    }

        # -------------------------------------------------------------
        # 2. Detect Conjunction / Intersection (2i)
        # e.g., "哪些电影由克里斯蒂安·贝尔主演并且由克里斯托弗·诺兰导演？"
        # e.g., "Which movies star Christian Bale and were directed by Christopher Nolan?"
        # -------------------------------------------------------------
        conj_split = re.split(CONJ_CONNECTIVES_REGEX, q_lower)
        if len(conj_split) >= 2:
            part1 = conj_split[0]
            part2 = " ".join(conj_split[1:])
            ents1 = self.find_entities(part1)
            ents2 = self.find_entities(part2)

            if ents1 and ents2:
                e1, e1_name = max(ents1, key=lambda x: len(x[1]))[:2]
                e2, e2_name = max(ents2, key=lambda x: len(x[1]))[:2]
                
                p1_rel_text = part1.replace(e1_name.lower(), " ")
                p2_rel_text = part2.replace(e2_name.lower(), " ")

                rel1 = self.find_relation(p1_rel_text)
                rel2 = self.find_relation(p2_rel_text)

                if rel1 and rel2:
                    r1, r1_uri = rel1
                    r2, r2_uri = rel2
                    # BetaE 2i: ((e1, (r1,)), (e2, (r2,)))
                    query_tuple = ((e1, (r1,)), (e2, (r2,)))
                    return {
                        "question": question,
                        "query_type": "2i",
                        "logical_query": query_tuple,
                        "entities": [(e1, e1_name), (e2, e2_name)],
                        "relations": [(r1, r1_uri), (r2, r2_uri)],
                        "explanation": f"Conjunction: ?X satisfies {r1_uri}({e1_name}, ?X) AND {r2_uri}({e2_name}, ?X)"
                    }

        # -------------------------------------------------------------
        # 3. Detect Disjunction / Union (2u)
        # e.g., "哪些电影由克里斯蒂安·贝尔主演或者由克里斯托弗·诺兰导演？"
        # e.g., "Which movies star Christian Bale or were directed by Christopher Nolan?"
        # -------------------------------------------------------------
        disj_split = re.split(DISJ_CONNECTIVES_REGEX, q_lower)
        if len(disj_split) >= 2:
            part1 = disj_split[0]
            part2 = " ".join(disj_split[1:])
            ents1 = self.find_entities(part1)
            ents2 = self.find_entities(part2)

            if ents1 and ents2:
                e1, e1_name = max(ents1, key=lambda x: len(x[1]))[:2]
                e2, e2_name = max(ents2, key=lambda x: len(x[1]))[:2]
                
                p1_rel_text = part1.replace(e1_name.lower(), " ")
                p2_rel_text = part2.replace(e2_name.lower(), " ")

                rel1 = self.find_relation(p1_rel_text)
                rel2 = self.find_relation(p2_rel_text)

                if rel1 and rel2:
                    r1, r1_uri = rel1
                    r2, r2_uri = rel2
                    # BetaE 2u: ((e1, (r1,)), (e2, (r2,)), (-1,))
                    query_tuple = ((e1, (r1,)), (e2, (r2,)), (-1,))
                    return {
                        "question": question,
                        "query_type": "2u",
                        "logical_query": query_tuple,
                        "entities": [(e1, e1_name), (e2, e2_name)],
                        "relations": [(r1, r1_uri), (r2, r2_uri)],
                        "explanation": f"Disjunction: ?X satisfies {r1_uri}({e1_name}, ?X) OR {r2_uri}({e2_name}, ?X)"
                    }

        # -------------------------------------------------------------
        # 4. Detect 2-hop Path (2p)
        # e.g., "盗梦空间的导演出生在哪里？"
        # e.g., "Where was the director of Inception born?"
        # -------------------------------------------------------------
        ents = self.find_entities(q)
        if ents:
            e, e_name, e_start, e_end = max(ents, key=lambda x: len(x[1]))

            is_2p = (
                re.search(r"\bdirector of\b", q_lower) or
                re.search(r"导演出生|导演的出生|导演.*在哪|导演.*国籍", q_lower)
            )
            if is_2p:
                if re.search(r"(?:born|birthplace|出生|出生地|出生在哪)", q_lower):
                    # r1: reverse director (-/film/director/film, id 205)
                    # r2: place of birth (+/people/person/place_of_birth, id 48)
                    r1 = self.rel2id.get("-/film/director/film", 205)
                    r2 = self.rel2id.get("+/people/person/place_of_birth", 48)
                    return {
                        "question": question,
                        "query_type": "2p",
                        "logical_query": (e, (r1, r2)),
                        "entities": [(e, e_name)],
                        "relations": [(r1, "-/film/director/film"), (r2, "+/people/person/place_of_birth")],
                        "explanation": f"2-hop Path: ?X is birthplace of director of {e_name}"
                    }
                elif re.search(r"(?:nationality|citizen|国籍)", q_lower):
                    r1 = self.rel2id.get("-/film/director/film", 205)
                    r2 = self.rel2id.get("+/people/person/nationality", 96)
                    return {
                        "question": question,
                        "query_type": "2p",
                        "logical_query": (e, (r1, r2)),
                        "entities": [(e, e_name)],
                        "relations": [(r1, "-/film/director/film"), (r2, "+/people/person/nationality")],
                        "explanation": f"2-hop Path: ?X is nationality of director of {e_name}"
                    }

            # -------------------------------------------------------------
            # 5. Default: 1-hop Projection (1p)
            # e.g., "克里斯托弗·诺兰导演了哪些电影？"
            # e.g., "Where was Christopher Nolan born?"
            # -------------------------------------------------------------
            clean_q = q.replace(e_name, " ")
            rel = self.find_relation(clean_q)
            if rel:
                r, r_uri = rel
                return {
                    "question": question,
                    "query_type": "1p",
                    "logical_query": (e, (r,)),
                    "entities": [(e, e_name)],
                    "relations": [(r, r_uri)],
                    "explanation": f"1-hop Path: ?X satisfies {r_uri}({e_name}, ?X)"
                }

        raise ValueError(
            f"Could not automatically parse question into a logical query: '{question}'. "
            f"Found entities: {[m[1] for m in ents]}. Ensure key entity and relation phrases are included."
        )
