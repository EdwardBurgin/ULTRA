import re
from typing import Any, Dict, List, Optional, Tuple, Union


# Common question words, relational verbs/nouns, and schema indicators that should not
# be casually linked as subject entities unless explicitly quoted.
SCHEMA_STOP_WORDS = {
    "star", "stars", "starring", "starred", "major", "majors", "majored", "director",
    "directors", "directed", "actor", "actors", "actress", "actresses", "film", "films",
    "movie", "movies", "study", "studies", "education", "field", "fields", "degree",
    "degrees", "student", "students", "born", "birth", "birthplace", "nationality",
    "citizen", "citizenship", "award", "awards", "genre", "genres", "music", "author",
    "authors", "written", "writer", "writers", "country", "countries", "city", "cities",
    "place", "places", "what", "which", "where", "who", "when", "how", "whose", "whom",
    "both", "either", "neither", "also", "well", "name", "names", "kind", "kinds", "type",
    "types", "one", "ones", "associated", "located", "contained", "people", "person",
}


class NLQueryParser:
    """
    Translates naturally phrased questions into formal First-Order Logical (FOL)
    query structures (BetaE nested tuples) for UltraQuery execution.
    
    Supports:
      - 1p: 1-hop path queries: (e, (r,))
      - 2p: 2-hop path queries: (e, (r1, r2))
      - 2i: 2-hop conjunctions / intersections: ((e1, (r1,)), (e2, (r2,)))
      - 2in: 2-hop intersections with negation: ((e1, (r1,)), (e2, (r2, -1)))
      - 2u: 2-hop disjunctions / unions: ((e1, (r1,)), (e2, (r2,)), -2)
    """

    def __init__(
        self,
        id2ent: Dict[int, str],
        id2rel: Dict[int, str],
        ent2text: Optional[Dict[str, str]] = None,
        rel2text: Optional[Dict[str, str]] = None,
    ):
        self.id2ent = id2ent
        self.id2rel = id2rel
        self.ent2id = {v: k for k, v in id2ent.items()}
        self.rel2id = {v: k for k, v in id2rel.items()}
        
        self.ent2text = ent2text or {}
        self.rel2text = rel2text or {}

        # Build name -> entity_id lookup
        self.name2id: Dict[str, int] = {}
        for mid, name in self.ent2text.items():
            if mid in self.ent2id:
                eid = self.ent2id[mid]
                self.name2id[name.lower().strip()] = eid

        # Add raw MIDs
        for mid, eid in self.ent2id.items():
            self.name2id[mid.lower().strip()] = eid

        # Add common aliases for high-frequency entities
        aliases = {
            "uc berkeley": "university of california, berkeley",
            "berkeley": "university of california, berkeley",
            "cal": "university of california, berkeley",
            "harvard": "harvard university",
            "mit": "massachusetts institute of technology",
            "ucl": "university college london",
            "nyu": "new york university",
            "nolan": "christopher nolan",
            "chris nolan": "christopher nolan",
            "bale": "christian bale",
            "obama": "barack obama",
            "barack": "barack obama",
            "steve jobs": "steve jobs",
            "alan turing": "alan turing",
            "the dark knight": "the dark knight",
            "dark knight": "the dark knight",
            "the dark knight rises": "the dark knight rises",
            "dark knight rises": "the dark knight rises",
            "inception": "inception",
            "memento": "memento",
            "the prestige": "the prestige",
            "prestige": "the prestige",
            "batman begins": "batman begins",
            "america": "united states of america",
            "usa": "united states of america",
            "us": "united states of america",
            "uk": "united kingdom",
            "britain": "united kingdom",
            "england": "england",
            "london": "london",
            "chicago cubs": "chicago cubs",
        }
        for alias, target_name in aliases.items():
            if target_name in self.name2id and alias not in self.name2id:
                self.name2id[alias] = self.name2id[target_name]

        # Sort candidate names by length descending
        self._sorted_names = sorted(self.name2id.keys(), key=lambda x: len(x), reverse=True)

        # High-precision relation patterns for FB15k-237
        self._relation_patterns: List[Tuple[str, str]] = [
            # Films / Media
            (r"\b(direct|directed|director|filmmaker|directed by)\b", "+/film/director/film"),
            (r"\b(act in|acted in|actor|actress|star|stars|starred|starring|performance|played in|featuring)\b", "+/film/actor/film./film/performance/film"),
            (r"\b(produced|produced by|producer)\b", "+/film/film/produced_by"),
            (r"\b(written by|screenplay|screenwriter|wrote)\b", "+/film/film/written_by"),
            (r"\b(film genre|movie genre|film.*genre)\b", "+/film/film/genre"),
            
            # People / Biography
            (r"\b(born in|born at|birthplace|place of birth|where was .* born)\b", "+/people/person/place_of_birth"),
            (r"\b(nationality|citizenship|citizen of|from which country)\b", "+/people/person/nationality"),
            (r"\b(gender|sex)\b", "+/people/person/gender"),
            (r"\b(profession|job|occupation|career)\b", "+/people/person/profession"),
            (r"\b(spouse|married to|wife|husband)\b", "+/people/person/spouse_s./people/marriage/spouse"),
            (r"\b(child|children|son|daughter)\b", "+/people/person/children"),
            (r"\b(parents|father|mother)\b", "+/people/person/parents"),
            (r"\b(influenced by|influences)\b", "+/influence/influence_node/influenced_by"),
            
            # Education
            (r"\b(major|field of study|fields of study|majored in|degree in|study|studies)\b", "+/education/educational_institution/students_graduates./education/education/major_field_of_study"),
            (r"\b(student|students|graduate|graduates|alumni|attended|graduated from|alma mater)\b", "+/education/educational_institution/students_graduates./education/education/student"),
            (r"\b(school type|type of school|institution type)\b", "+/education/educational_institution/school_type"),
            (r"\b(campuses|campus)\b", "+/education/educational_institution/campuses"),

            # Geography & Locations
            (r"\b(capital|capital of|capital city)\b", "+/location/country/capital"),
            (r"\b(language|languages|languages spoken|official language)\b", "+/location/country/languages_spoken"),
            (r"\b(located in|contained in|where is|in which city|in which country)\b", "+/location/location/contains"),

            # Awards & Honors
            (r"\b(award|awards|won|received|honor|awarded|nominated for)\b", "+/award/award_nominee/award_nominations./award/award_nomination/award"),

            # Music
            (r"\b(music genre|musical style|genre of music)\b", "+/music/artist/genre"),
            (r"\b(instrument|instruments played|role in track)\b", "+/music/artist/track_contributions./music/track_contribution/role"),
        ]

    def find_entities(self, text: str) -> List[Tuple[int, str, int, int]]:
        """
        Extracts entity mentions from natural text.
        Returns a list of (entity_id, matched_text, start_char, end_char).
        """
        text_lower = text.lower()
        matches = []
        occupied_spans = []

        # 1. Check quoted text first: "Christopher Nolan"
        quoted = re.finditer(r'["\']([^"\']+)["\']', text)
        for m in quoted:
            phrase = m.group(1).lower().strip()
            if phrase in self.name2id:
                eid = self.name2id[phrase]
                matches.append((eid, m.group(1), m.start(1), m.end(1)))
                occupied_spans.append((m.start(1), m.end(1)))

        # 2. Greedy longest string matching
        for candidate in self._sorted_names:
            # Skip schema stopwords unless quoted
            if candidate in SCHEMA_STOP_WORDS:
                continue
            if len(candidate) < 3:
                continue
            
            pattern = r'\b' + re.escape(candidate) + r'\b'
            for m in re.finditer(pattern, text_lower):
                s, e = m.start(), m.end()
                # Check overlap with already claimed longer spans
                if any(not (e <= os_ or s >= oe_) for os_, oe_ in occupied_spans):
                    continue
                
                eid = self.name2id[candidate]
                orig_text = text[s:e]
                matches.append((eid, orig_text, s, e))
                occupied_spans.append((s, e))

        # Sort by start position
        matches.sort(key=lambda x: x[2])
        return matches

    def find_relation(self, text: str) -> Optional[Tuple[int, str]]:
        """
        Matches relational phrases in text to a knowledge graph relation.
        Returns (relation_id, relation_uri) or None.
        """
        text_lower = text.lower()
        for pat, rel_uri in self._relation_patterns:
            if re.search(pat, text_lower):
                if rel_uri in self.rel2id:
                    return (self.rel2id[rel_uri], rel_uri)
                for prefix in ["+", "-"]:
                    alt = prefix + rel_uri.lstrip("+-")
                    if alt in self.rel2id:
                        return (self.rel2id[alt], alt)

        # Fallback: token overlap against rel2text
        tokens = set(re.findall(r"\w+", text_lower)) - SCHEMA_STOP_WORDS
        best_r = None
        best_score = 0.0

        for r_uri, desc in self.rel2text.items():
            desc_tokens = set(re.findall(r"\w+", desc.lower()))
            overlap = len(tokens & desc_tokens)
            if overlap > best_score and overlap >= 2:
                best_score = overlap
                best_r = r_uri

        if best_r and best_r in self.rel2id:
            return (self.rel2id[best_r], best_r)
        elif best_r:
            for prefix in ["+", "-"]:
                alt = prefix + best_r.lstrip("+-")
                if alt in self.rel2id:
                    return (self.rel2id[alt], alt)

        return None

    def parse(self, question: str) -> Dict[str, Any]:
        """
        Parses a natural language question into a BetaE logical query structure.
        """
        q = question.strip()
        q_lower = q.lower()

        # -------------------------------------------------------------
        # 1. Detect Negation Conjunction (2in)
        # e.g., "movies star Christian Bale and were not directed by Christopher Nolan"
        # -------------------------------------------------------------
        neg_regex = r"\b(?:and|but)\s+(?:were\s+|are\s+|is\s+|was\s+|did\s+)?not\b|\bwithout\b|\bexcluding\b"
        neg_split = re.split(neg_regex, q_lower, maxsplit=1)
        if len(neg_split) == 2:
            part1, part2 = neg_split
            ents1 = self.find_entities(part1)
            ents2 = self.find_entities(part2)
            rel1 = self.find_relation(part1)
            rel2 = self.find_relation(part2)

            if ents1 and ents2 and rel1 and rel2:
                # Pick longest entity if multiple
                e1, e1_name = max(ents1, key=lambda x: len(x[1]))[:2]
                e2, e2_name = max(ents2, key=lambda x: len(x[1]))[:2]
                r1, r1_uri = rel1
                r2, r2_uri = rel2
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
        # e.g., "movies star Christian Bale and were directed by Christopher Nolan"
        # -------------------------------------------------------------
        conj_split = re.split(r"\b(?:and|both)\b", q_lower)
        if len(conj_split) >= 2:
            part1 = conj_split[0]
            part2 = " ".join(conj_split[1:])
            ents1 = self.find_entities(part1)
            ents2 = self.find_entities(part2)
            rel1 = self.find_relation(part1)
            rel2 = self.find_relation(part2)

            if ents1 and ents2 and rel1 and rel2:
                e1, e1_name = max(ents1, key=lambda x: len(x[1]))[:2]
                e2, e2_name = max(ents2, key=lambda x: len(x[1]))[:2]
                r1, r1_uri = rel1
                r2, r2_uri = rel2
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
        # e.g., "movies star Christian Bale or were directed by Christopher Nolan"
        # -------------------------------------------------------------
        disj_split = re.split(r"\b(?:or|either)\b", q_lower)
        if len(disj_split) >= 2:
            part1 = disj_split[0]
            part2 = " ".join(disj_split[1:])
            ents1 = self.find_entities(part1)
            ents2 = self.find_entities(part2)
            rel1 = self.find_relation(part1)
            rel2 = self.find_relation(part2)

            if ents1 and ents2 and rel1 and rel2:
                e1, e1_name = max(ents1, key=lambda x: len(x[1]))[:2]
                e2, e2_name = max(ents2, key=lambda x: len(x[1]))[:2]
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
        # e.g., "Where was the director of Inception born?"
        # -------------------------------------------------------------
        ents = self.find_entities(q)
        if ents:
            # Pick longest entity
            e, e_name, e_start, e_end = max(ents, key=lambda x: len(x[1]))

            if re.search(r"\bdirector of\b", q_lower) and re.search(r"\b(born|birthplace)\b", q_lower):
                r1 = self.rel2id.get("-/film/director/film", 205)
                r2 = self.rel2id.get("+/people/person/place_of_birth", 48)
                query_tuple = (e, (r1, r2))
                return {
                    "question": question,
                    "query_type": "2p",
                    "logical_query": query_tuple,
                    "entities": [(e, e_name)],
                    "relations": [(r1, "-/film/director/film"), (r2, "+/people/person/place_of_birth")],
                    "explanation": f"2-hop Path: ?X is birthplace of director of {e_name}"
                }

            if re.search(r"\bdirector of\b", q_lower) and re.search(r"\b(nationality|citizen)\b", q_lower):
                r1 = self.rel2id.get("-/film/director/film", 205)
                r2 = self.rel2id.get("+/people/person/nationality", 96)
                query_tuple = (e, (r1, r2))
                return {
                    "question": question,
                    "query_type": "2p",
                    "logical_query": query_tuple,
                    "entities": [(e, e_name)],
                    "relations": [(r1, "-/film/director/film"), (r2, "+/people/person/nationality")],
                    "explanation": f"2-hop Path: ?X is nationality of director of {e_name}"
                }

            # -------------------------------------------------------------
            # 5. Default: 1-hop Projection (1p)
            # e.g., "What films did Christopher Nolan direct?"
            # -------------------------------------------------------------
            rel = self.find_relation(q)
            if rel:
                r, r_uri = rel
                query_tuple = (e, (r,))
                return {
                    "question": question,
                    "query_type": "1p",
                    "logical_query": query_tuple,
                    "entities": [(e, e_name)],
                    "relations": [(r, r_uri)],
                    "explanation": f"1-hop Path: ?X satisfies {r_uri}({e_name}, ?X)"
                }

        raise ValueError(
            f"Could not automatically parse question into a logical query: '{question}'. "
            f"Found entities: {[m[1] for m in ents]}. Ensure key entity and relation phrases are included."
        )
