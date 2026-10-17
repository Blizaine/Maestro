"""Conservative source-grounded grouping for H3 object reference views."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


_MAX_SOURCE_CHARS = 30_000
_MAX_CONTEXT_CHARS = 18_000
_MAX_GROUPS = 8
_MAX_PICTURES_PER_GROUP = 9
_MAX_EVIDENCE_CHARS = 600

_OBJECT_ROLE_RE = re.compile(
    r"<Picture\s+(?P<index>[1-9]\d*)>\s+is\s+an\s+object\s*/\s*prop\s+"
    r"reference\s+for\s+(?P<role>[^;\r\n]+)",
    re.IGNORECASE,
)
_PICTURE_MENTION_RE = re.compile(
    r"(?<!\w)(?:<\s*Picture\s*(?P<tag>[1-9]\d*)\s*>|Picture\s*(?P<plain>[1-9]\d*))(?!\w)",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"[^.!?\r\n]+(?:[.!?]+|$)")
_ENTITY_STOP = set("""
    a an the and or of for in on at to with from by as is are its it their this that
    same one single object prop reference picture image view views shot photo
    exterior outside outer interior inside inner feature features detail details
    close up front rear back side top bottom left right full body profile cutaway
    show shows shown showing featuring equipped fitted carrying bearing small large
    big tiny red blue green black white silver gold orange yellow brown gray grey
    old new extra added another different camera lens scene frame using like seen
    based modeled switch switches then briefly tight wide medium macro overhead low
    high pov angle closeup closeups detailed
""".split())
_INTERIOR_TERMS = {"inside", "interior", "interiors", "cabin", "cockpit", "inner"}
_EXTERIOR_TERMS = {"exterior", "outside", "outer", "profile", "silhouette"}
_FEATURE_TERMS = {"feature", "features", "detail", "details", "closeup", "closeups"}
_FEATURE_RELATION_RE = re.compile(
    r"\b(?:has|have|with|featuring|equipped|fitted|carrying|bearing|including|detail|details|feature|features)\b",
    re.IGNORECASE,
)
_COREFERENCE_RE = re.compile(
    r"\b(?:it|its|they|their|this|that|these|those)\b", re.IGNORECASE
)
_VIEW_LINK_RE = re.compile(
    r"\b(?:like|as\s+(?:in|shown|seen)|use|using|based\s+on|modeled\s+on|has|have|with|"
    r"inside|interior|exterior|feature|detail|view|angle|shot|shown|seen)\b",
    re.IGNORECASE,
)
_CLAUSE_LINK_RE = re.compile(
    r"\b(?:has|have|features?|includes?|shows?|shown|using|like|with|carrying|bearing|"
    r"equipped|fitted|is|are|was|were|appears?|arrives?|enters?|follows?|moves?|"
    r"flies|launches?|blasts?|turns?|switch(?:es|ed)?|sits?|stands?|rests?|waits?)\b",
    re.IGNORECASE,
)
_DISTINCT_ENTITY_RE = re.compile(
    r"\b(?:not\s+the\s+same|different|separate|distinct|another|additional|other)\s+"
    r"(?:(?:[a-z0-9-]+)\s+){0,2}{entity}\b",
    re.IGNORECASE,
)
_MULTIPLE_COUNT_RE = re.compile(
    r"\b(?:two|three|four|five|several|multiple|many|both|a\s+pair\s+of)\s+"
    r"(?:(?:[a-z0-9-]+)\s+){0,2}{entity}\b",
    re.IGNORECASE,
)


def _words(text: str) -> list[str]:
    words = [match.group(0).casefold() for match in _WORD_RE.finditer(text)]
    normalized = []
    for word in words:
        if word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"
        elif word.endswith("s") and len(word) > 4 and not word.endswith("ss"):
            word = word[:-1]
        normalized.append(word)
    return normalized


def _role_views(role: str) -> set[str]:
    words = set(_words(role))
    views = set()
    if words & _INTERIOR_TERMS:
        views.add("interior")
    if words & _EXTERIOR_TERMS:
        views.add("exterior")
    if words & _FEATURE_TERMS or _FEATURE_RELATION_RE.search(role):
        views.add("feature")
    if not views:
        views.add("general")
    return views


def _object_roles(reference_context: str) -> dict[int, dict[str, Any]]:
    roles: dict[int, dict[str, Any]] = {}
    duplicate_indices: set[int] = set()
    for match in _OBJECT_ROLE_RE.finditer(reference_context):
        index = int(match.group("index"))
        role = " ".join(match.group("role").split()).strip(" .")[:240]
        if not role:
            continue
        if index in roles:
            duplicate_indices.add(index)
            continue
        roles[index] = {"role": role, "tokens": set(_words(role)), "views": _role_views(role)}
    for index in duplicate_indices:
        roles.pop(index, None)
    return roles


def _source_mentions(text: str, offset: int = 0) -> list[tuple[int, int, int]]:
    mentions = []
    for match in _PICTURE_MENTION_RE.finditer(text):
        number = int(match.group("tag") or match.group("plain"))
        mentions.append((number, offset + match.start(), offset + match.end()))
    return mentions


def _singular_entity(entity: str) -> str:
    return entity[:-1] if entity.endswith("s") and len(entity) > 3 and not entity.endswith("ss") else entity


def _plural_entity(entity: str) -> str:
    if entity.endswith("y") and len(entity) > 2:
        return entity[:-1] + "ies"
    if entity.endswith(("s", "x", "ch", "sh")):
        return entity + "es"
    return entity + "s"


def _distinct_instances(sentence: str, entity: str) -> bool:
    normalized_entity = _singular_entity(entity)
    escaped_entity = re.escape(normalized_entity) + r"(?:s|es)?"
    plural = re.escape(_plural_entity(normalized_entity))
    distinct_pattern = _DISTINCT_ENTITY_RE.pattern.replace("{entity}", escaped_entity)
    count_pattern = _MULTIPLE_COUNT_RE.pattern.replace("{entity}", plural)
    if re.search(distinct_pattern, sentence, re.I):
        return True
    if re.search(count_pattern, sentence, re.I):
        return True
    if re.search(r"\b(?:two|three|four|several|multiple|many)\s+(?:separate\s+)?instances?\b", sentence, re.I):
        return True
    # Two independently introduced instances can otherwise look like a shared noun chain.
    starts = re.compile(rf"\b(?:a|an|one)\s+(?:(?:[a-z0-9-]+)\s+){{0,2}}{escaped_entity}\b", re.I)
    return len(list(starts.finditer(sentence))) > 1


def _candidate_entity(sentence: str, picture_roles: list[dict[str, Any]]) -> str | None:
    counts: Counter[str] = Counter()
    for item in picture_roles:
        counts.update(item["tokens"])
    source_tokens = set(_words(sentence))
    candidates = [
        word for word, count in counts.items()
        if count >= 2 and word in source_tokens and word not in _ENTITY_STOP and len(word) >= 3
    ]
    if not candidates:
        return None
    # Prefer role terms shared by the most declarations, then the most specific word.
    candidates.sort(key=lambda word: (counts[word], len(word), -sentence.casefold().find(word)), reverse=True)
    selected = candidates[0]
    source_match = re.search(rf"(?<!\w){re.escape(selected)}(?:s|es)?(?!\w)", sentence, re.I)
    return source_match.group(0) if source_match else None


def _starts_with_competing_subject(
    clause: str,
    entity: str,
    role_tokens: set[str],
) -> bool:
    """Reject a new article-led subject before a picture label, not scene nouns in the gap."""
    match = re.match(
        r"\s*,?\s*(?:(?:and|then|next|afterward|afterwards)\b[\s,]+)*"
        r"(?P<article>a|an|the|another|different|separate)\s+"
        r"(?P<phrase>[^,.!?;]{1,70})",
        clause,
        re.IGNORECASE,
    )
    if not match:
        return False
    phrase = _CLAUSE_LINK_RE.split(match.group("phrase"), maxsplit=1)[0]
    words = set(_words(phrase))
    if not words:
        return False

    article = match.group("article").casefold()
    words = {word for word in words if not word.isdigit()}
    # An indefinite instance of the canonical object is a distinct introduction;
    # scene descriptions such as "an inside shot" contain only view words here.
    if article in {"a", "an"} and entity in words:
        return True
    return bool(words - _ENTITY_STOP - role_tokens - {entity})


def _has_source_chain(
    sentence: str,
    mentions: list[tuple[int, int, int]],
    entity: str,
    roles: dict[int, dict[str, Any]],
) -> bool:
    lowered = sentence.casefold()
    entity_token = _singular_entity(entity.casefold())
    if _distinct_instances(sentence, entity_token):
        return False
    first = mentions[0]
    if entity_token not in _words(sentence[max(0, first[1] - 100):min(len(sentence), first[2] + 100)]):
        return False

    escaped_entity = re.escape(entity_token)
    explicit_same = bool(re.search(
        rf"\b(?:same|single)\s+(?:(?:[a-z0-9-]+)\s+){{0,2}}{escaped_entity}(?:s|es)?\b"
        rf"|\b(?:same|single|one)\s+object\b",
        lowered,
    ))
    previous_end = first[2]
    role_tokens = set().union(*(roles[number]["tokens"] for number, _, _ in mentions))
    chain_established = explicit_same
    for number, start, end in mentions[1:]:
        gap = sentence[previous_end:start]
        link_text = gap + sentence[end:min(len(sentence), end + 60)]
        role = roles[number]
        clause_start = max(
            sentence.rfind(".", previous_end, start),
            sentence.rfind("!", previous_end, start),
            sentence.rfind("?", previous_end, start),
            sentence.rfind(";", previous_end, start),
        ) + 1
        current_clause = sentence[max(clause_start, previous_end):start]
        linked = (
            explicit_same
            or bool(_COREFERENCE_RE.search(gap + " " + sentence[end:min(len(sentence), end + 60)]))
            or (
                bool(role["views"] & {"feature"})
                and bool(_FEATURE_RELATION_RE.search(gap))
                and bool((role["tokens"] - _ENTITY_STOP - {entity_token}) & set(_words(gap)))
            )
            or (
                chain_established
                and bool(role["views"] & {"interior"})
                and bool(_VIEW_LINK_RE.search(current_clause))
            )
        )
        if not linked:
            return False
        if _starts_with_competing_subject(current_clause, entity_token, role_tokens):
            return False
        chain_established = True
        previous_end = end
    return True


def extract_h3_reference_view_groups(
    source_prompt: str,
    reference_context: str,
) -> list[dict[str, Any]]:
    """Return only source-supported groups of complementary object picture views.

    The result uses canonical one-based Picture slot indices and contains no
    generated subject identifiers. Ambiguous, malformed, oversized, or weakly
    connected inputs fail closed as an empty list.
    """

    source = str(source_prompt or "")
    context = str(reference_context or "")
    if not source.strip() or not context.strip() or len(source) > _MAX_SOURCE_CHARS or len(context) > _MAX_CONTEXT_CHARS:
        return []
    roles = _object_roles(context)
    if len(roles) < 2:
        return []

    sentences = list(_SENTENCE_RE.finditer(source))
    candidates = []
    for start_index in range(len(sentences)):
        for width in range(1, min(3, len(sentences) - start_index) + 1):
            start = sentences[start_index].start()
            end = sentences[start_index + width - 1].end()
            evidence = source[start:end]
            if len(evidence) > _MAX_EVIDENCE_CHARS:
                continue
            raw_mentions = [item for item in _source_mentions(evidence) if item[0] in roles]
            seen_mentions: set[int] = set()
            mentions = []
            for item in raw_mentions:
                if item[0] not in seen_mentions:
                    mentions.append(item)
                    seen_mentions.add(item[0])
            indices = list(dict.fromkeys(item[0] for item in mentions))
            if 2 <= len(indices) <= _MAX_PICTURES_PER_GROUP:
                candidates.append((len(indices), len(evidence), start, evidence, mentions, indices))

    groups: list[dict[str, Any]] = []
    used_indices: set[int] = set()
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    for _, _, _, evidence, mentions, indices in candidates:
        if used_indices.intersection(indices):
            continue
        picture_roles = [roles[index] for index in indices]
        entity = _candidate_entity(evidence, picture_roles)
        if not entity or not _has_source_chain(evidence, mentions, entity, roles):
            continue
        entity_token = _singular_entity(_words(entity)[0])
        if any(
            entity_token not in item["tokens"]
            and (item["tokens"] - _ENTITY_STOP)
            for item in picture_roles
        ):
            continue
        views = set().union(*(item["views"] for item in picture_roles))
        if len(views) < 2 or not views.intersection({"exterior", "general"}) or not views.intersection({"feature", "interior"}):
            continue
        if len(groups) >= _MAX_GROUPS:
            break

        label_indices = sorted(indices)
        view_facts = [
            f"<Picture {index}> has the declared {', '.join(sorted(roles[index]['views']))} view role: {roles[index]['role']}."
            for index in label_indices
        ]
        groups.append({
            "picture_indices": label_indices,
            "entity": entity,
            "relationship": f"The source connects these complementary views to one persistent {entity}.",
            "source_evidence": evidence,
            "view_facts": view_facts,
        })
        used_indices.update(indices)
    return groups


__all__ = ["extract_h3_reference_view_groups"]
