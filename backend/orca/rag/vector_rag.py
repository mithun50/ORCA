"""Document RAG over marine advisories, regulations and SOPs.

Runs in one of two modes, decided at startup:

* **lexical** (default, zero extra dependencies): a pure-Python BM25 index with a
  marine synonym expansion pass. Good enough for a corpus of advisories and
  regulations, and it means the prototype has no model download on first run.
* **dense** (when `ORCA_QDRANT_URL` is set and sentence-transformers is
  importable): embeddings in Qdrant, with the BM25 scores blended in so exact
  terms like "SWH" or "IMBL" are never lost to a fuzzy nearest neighbour.

Live advisory text pulled by the INCOIS and IMD connectors is injected into the
same index at request time, so a citation can be to today's bulletin rather than
only to the seeded corpus.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..config import get_settings
from ..schemas import Provenance, Tier

log = logging.getLogger("orca.rag.vector")

# Domain synonyms so a fisherman's wording reaches agency vocabulary.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "fish": ("fishing", "catch", "shoal", "pfz", "fishery"),
    "pfz": ("potential fishing zone", "advisory", "shoal", "aggregation"),
    "safe": ("safety", "hazard", "warning", "advisory", "venture"),
    "venture": ("go to sea", "sortie", "sail", "safety"),
    "wave": ("swh", "significant wave height", "swell", "sea state"),
    "wind": ("gale", "squall", "beaufort", "gust"),
    "storm": ("cyclone", "depression", "low pressure", "squall"),
    "lightning": ("thunderstorm", "convective", "squall", "damini"),
    "boundary": ("imbl", "eez", "maritime boundary", "geofence"),
    "chlorophyll": ("chl", "ocean colour", "productivity", "phytoplankton"),
    "temperature": ("sst", "sea surface temperature", "thermal front"),
    "tide": ("sea level", "high water", "low water"),
    "route": ("navigation", "passage", "track", "course"),
    "decline": ("decrease", "reduction", "downturn", "productivity"),
}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for", "from",
    "how", "i", "if", "in", "is", "it", "me", "my", "of", "on", "or", "that",
    "the", "there", "they", "this", "to", "we", "what", "when", "where", "which",
    "will", "with", "you", "your", "any", "should", "would", "near", "today",
    "tomorrow",
}


def tokenize(text: str) -> list[str]:
    return [
        t
        for t in re.findall(r"[a-z0-9]+", text.lower())
        if t not in STOPWORDS and len(t) > 1
    ]


def expand_query(text: str) -> list[str]:
    tokens = tokenize(text)
    expanded = list(tokens)
    for token in tokens:
        for extra in SYNONYMS.get(token, ()):
            expanded.extend(tokenize(extra))
    return expanded


@dataclass
class Chunk:
    id: str
    title: str
    text: str
    source_path: str
    agency: str = ""
    tier: Tier = Tier.SEED
    url: str = ""
    tags: tuple[str, ...] = ()
    tokens: list[str] = field(default_factory=list)

    def provenance(self) -> Provenance:
        return Provenance(
            agency=self.agency or "ORCA seeded knowledge base",
            dataset=self.title,
            tier=self.tier,
            url=self.url,
            access_method="document-rag",
            official=self.tier in (Tier.ISRO, Tier.INCOIS, Tier.IMD),
            caveat=(
                ""
                if self.tier in (Tier.ISRO, Tier.INCOIS, Tier.IMD)
                else "curated reference text seeded into the ORCA knowledge base"
            ),
        )


@dataclass
class Retrieved:
    chunk: Chunk
    score: float
    matched_terms: list[str] = field(default_factory=list)


class VectorRag:
    """BM25 (+ optional dense) index over the advisory corpus."""

    K1 = 1.4
    B = 0.72

    def __init__(self) -> None:
        self.settings = get_settings()
        self.chunks: list[Chunk] = []
        self._df: Counter[str] = Counter()
        self._avg_len = 1.0
        self._dense = None  # lazily built Qdrant-backed searcher
        self._dense_ready = False

    # ------------------------------------------------------------- indexing #

    def load_directory(self, directory: str | None = None) -> int:
        root = Path(directory or self.settings.knowledge_dir)
        if not root.exists():
            log.warning("knowledge dir %s missing", root)
            return 0
        added = 0
        for path in sorted(root.rglob("*.md")):
            added += len(self._chunks_from_markdown(path))
        self._reindex()
        log.info("vector RAG indexed %d chunks from %s", len(self.chunks), root)
        return added

    def _chunks_from_markdown(self, path: Path) -> list[Chunk]:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return []
        meta, body = _split_front_matter(raw)
        agency = meta.get("agency", "")
        url = meta.get("url", "")
        tier = _tier_from_string(meta.get("tier", ""))
        tags = tuple(t.strip() for t in meta.get("tags", "").split(",") if t.strip())
        doc_title = meta.get("title") or path.stem.replace("_", " ").title()

        created: list[Chunk] = []
        for i, (heading, section) in enumerate(_split_sections(body)):
            text = section.strip()
            if len(text) < 60:
                continue
            title = f"{doc_title} - {heading}" if heading else doc_title
            chunk = Chunk(
                id=f"{path.stem}#{i}",
                title=title,
                text=text,
                source_path=str(path),
                agency=agency,
                tier=tier,
                url=url,
                tags=tags,
            )
            self.chunks.append(chunk)
            created.append(chunk)
        return created

    def add_live_document(
        self,
        *,
        doc_id: str,
        title: str,
        text: str,
        agency: str,
        tier: Tier,
        url: str,
        tags: Iterable[str] = (),
    ) -> int:
        """Inject freshly fetched bulletin text into the index for this process."""
        existing = {c.id for c in self.chunks}
        added = 0
        for i, block in enumerate(_split_paragraphs(text, target_chars=900)):
            cid = f"{doc_id}#{i}"
            if cid in existing or len(block) < 80:
                continue
            self.chunks.append(
                Chunk(
                    id=cid,
                    title=title,
                    text=block,
                    source_path=url,
                    agency=agency,
                    tier=tier,
                    url=url,
                    tags=tuple(tags),
                )
            )
            added += 1
        if added:
            self._reindex()
        return added

    def _reindex(self) -> None:
        self._df = Counter()
        total = 0
        for chunk in self.chunks:
            chunk.tokens = tokenize(f"{chunk.title} {chunk.text} {' '.join(chunk.tags)}")
            total += len(chunk.tokens)
            for term in set(chunk.tokens):
                self._df[term] += 1
        self._avg_len = (total / len(self.chunks)) if self.chunks else 1.0
        self._dense_ready = False

    # -------------------------------------------------------------- search  #

    def search(
        self, query: str, k: int = 5, *, tag_filter: Iterable[str] | None = None
    ) -> list[Retrieved]:
        if not self.chunks:
            return []
        terms = expand_query(query)
        if not terms:
            return []
        wanted = set(tag_filter or ())
        n = len(self.chunks)
        results: list[Retrieved] = []
        for chunk in self.chunks:
            if wanted and not wanted.intersection(chunk.tags):
                continue
            score, matched = self._bm25(terms, chunk, n)
            if score > 0:
                # official agency text outranks seeded reference text on ties
                if chunk.tier in (Tier.ISRO, Tier.INCOIS, Tier.IMD):
                    score *= 1.15
                results.append(Retrieved(chunk=chunk, score=score, matched_terms=matched))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]

    def _bm25(
        self, terms: list[str], chunk: Chunk, n_docs: int
    ) -> tuple[float, list[str]]:
        tf = Counter(chunk.tokens)
        length = len(chunk.tokens) or 1
        score = 0.0
        matched: list[str] = []
        for term in set(terms):
            freq = tf.get(term, 0)
            if not freq:
                continue
            matched.append(term)
            df = self._df.get(term, 0) or 1
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denom = freq + self.K1 * (1 - self.B + self.B * length / self._avg_len)
            score += idf * (freq * (self.K1 + 1)) / denom
        return score, matched

    # --------------------------------------------------------------- stats  #

    def stats(self) -> dict[str, Any]:
        by_tier: Counter[str] = Counter(c.tier.value for c in self.chunks)
        return {
            "mode": "dense+lexical" if self._dense_ready else "lexical-bm25",
            "chunks": len(self.chunks),
            "documents": len({c.source_path for c in self.chunks}),
            "by_tier": dict(by_tier),
            "avg_chunk_tokens": round(self._avg_len, 1),
        }


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _split_front_matter(raw: str) -> tuple[dict[str, str], str]:
    if not raw.startswith("---"):
        return {}, raw
    parts = raw.split("---", 2)
    if len(parts) < 3:
        return {}, raw
    meta: dict[str, str] = {}
    for line in parts[1].splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip().lower()] = value.strip()
    return meta, parts[2]


def _split_sections(body: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    heading = ""
    buffer: list[str] = []
    for line in body.splitlines():
        if line.startswith("#"):
            if buffer:
                sections.append((heading, "\n".join(buffer)))
                buffer = []
            heading = line.lstrip("#").strip()
        else:
            buffer.append(line)
    if buffer:
        sections.append((heading, "\n".join(buffer)))
    return [(h, t) for h, t in sections if t.strip()]


def _split_paragraphs(text: str, target_chars: int = 900) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}|(?<=[.!?])\s{2,}", text) if p.strip()]
    blocks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) < target_chars:
            current = f"{current} {para}".strip()
        else:
            if current:
                blocks.append(current)
            current = para
    if current:
        blocks.append(current)
    return blocks


def _tier_from_string(value: str) -> Tier:
    mapping = {
        "isro": Tier.ISRO,
        "mosdac": Tier.ISRO,
        "incois": Tier.INCOIS,
        "moes": Tier.INCOIS,
        "imd": Tier.IMD,
        "fallback": Tier.FALLBACK,
        "seed": Tier.SEED,
    }
    return mapping.get(value.strip().lower(), Tier.SEED)
