"""Evidence-honesty gates for the manuscript draft."""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_MAIN = _ROOT / "paper" / "main.tex"
_REFERENCES = _ROOT / "paper" / "references.bib"
_EVIDENCE = _ROOT / "paper" / "claim_evidence.md"
_MARKER_RE = re.compile(r"\b(?:TODO|TBD|FIXME)\b", re.IGNORECASE)
_CITATION_RE = re.compile(r"\\cite[pt]?\*?\{([^}]+)\}")
_BIB_ENTRY_RE = re.compile(r"(?ms)^@\w+\{([^,]+),(.*?)(?=^@\w+\{|\Z)")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _active_tex(tex: str) -> str:
    return re.sub(r"(?m)(?<!\\)%.*$", "", tex)


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _section(tex: str, name: str) -> str:
    tex = _active_tex(tex)
    marker = rf"\section{{{name}}}"
    assert tex.count(marker) == 1, f"{name} must appear exactly once"
    start = tex.index(marker) + len(marker)
    remainder = tex[start:]
    next_section = re.search(r"\\section\{", remainder)
    return remainder[: next_section.start()] if next_section else remainder


def _abstract(tex: str) -> str:
    tex = _active_tex(tex)
    match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        tex,
        re.DOTALL,
    )
    assert match is not None, "main.tex must contain an abstract"
    return match.group(1)


def test_manuscript_files_have_no_unresolved_markers() -> None:
    for path in (_MAIN, _REFERENCES, _EVIDENCE):
        assert not _MARKER_RE.search(_read(path)), f"unresolved marker in {path}"


def test_required_sections_are_complete_unique_and_ordered() -> None:
    tex = _read(_MAIN)
    active = _active_tex(tex)
    sections = (
        "Introduction",
        "Related Work",
        "System and Method",
        "Experimental Setup",
        "Results",
        "Limitations",
        "Conclusion",
    )
    for name in sections:
        assert active.count(rf"\section{{{name}}}") == 1
    positions = [active.index(rf"\section{{{name}}}") for name in sections]
    assert positions == sorted(positions)
    for name in sections:
        assert _section(tex, name).strip(), f"{name} must contain active prose"


def test_pending_evaluation_is_explicit_without_result_overclaims() -> None:
    tex = _read(_MAIN)
    abstract = _abstract(tex).lower()
    results = _section(tex, "Results").lower()
    conclusion = _section(tex, "Conclusion").lower()

    for body, label in ((abstract, "abstract"), (results, "results")):
        assert "54-cell" in body and "pending" in body, label
        assert "calibration evaluation" in body and "pending" in body, label

    assert "0/54" in results
    assert "implemented, not empirically evaluated" in tex.lower()

    claim_sections = "\n".join((abstract, results, conclusion))
    forbidden = (
        r"\b(?:we|our results?)\s+"
        r"(?:show|demonstrate|find|establish|confirm|measure)\b",
        r"\boutperform(?:s|ed|ing)?\b",
        r"\brankings?\s+invert(?:s|ed)?\b",
        r"\bcalibrated\s+(?:deep[- ]ensemble|world model)\b",
        r"\b(?:ensemble|model|system|method|hagent)\s+"
        r"(?:is|are)\s+calibrated\b",
        r"\b(?:improves?|improved)\s+(?:performance|accuracy|score|cost)\b",
        r"\breduces?\s+(?:cost|latency|training jobs?)\b",
        r"\b(?:hagent|our (?:system|method|approach|model)|"
        r"the proposed (?:system|method|approach|model))\s+"
        r"(?:is|are)\s+"
        r"(?:effective|superior|significant|robust|calibrated)\b",
        r"\b(?:hagent|our (?:system|method|approach|model)|"
        r"the proposed (?:system|method|approach|model))\s+"
        r"increases?\s+sample efficiency\b",
        r"\bstatistically significant\b",
    )
    for pattern in forbidden:
        assert not re.search(pattern, claim_sections), pattern


def test_draft_labels_development_evidence_and_orchestration_honestly() -> None:
    tex = _read(_MAIN)
    setup = _normalized(_section(tex, "Experimental Setup"))
    introduction = _section(tex, "Introduction").lower()

    assert "120 synthetic trajectories" in setup
    assert (
        "they are development data for fitting and checking the outcome model, "
        "not a test set, publication benchmark, or estimate of performance" in setup
    )
    assert "infrastructure, not a scientific contribution" in introduction


def test_pending_draft_has_no_active_generated_tables_or_includes() -> None:
    active = _active_tex(_read(_MAIN))
    forbidden = (
        r"\input{",
        r"\include{",
        r"\begin{table",
        r"\begin{tabular",
    )
    for command in forbidden:
        assert command not in active


def test_claim_evidence_has_required_taxonomy_and_boundaries() -> None:
    evidence = _read(_EVIDENCE)
    required = (
        "## 2. Implemented mechanisms",
        "## 3. Development-only evidence",
        "## 4. Pending empirical claims and required evidence",
        "## 5. Bibliography verification ledger",
        "Only primary sources were accepted",
        "0/54",
        "120",
        "implemented, not empirically evaluated",
    )
    for phrase in required:
        assert phrase in evidence


def test_claim_evidence_freeze_record_remains_pending() -> None:
    evidence = _read(_EVIDENCE)
    required = (
        'git_sha: "<PENDING_AFTER_CODE_AND_DATA_FREEZE>"',
        'matrix_design_sha256: "<PENDING_FROM_VALIDATED_MATRIX_DESIGN>"',
        'checkpoint_sha256: "<PENDING_FROM_VALIDATED_ENSEMBLE_MANIFEST>"',
        'matrix_results: "src/backend/benchmarks/agent_matrix_results.jsonl"',
        'paired_advice: "src/backend/benchmarks/agent_matrix_advice.jsonl"',
        'calibration_report: "<PENDING_GROUPED_HOLDOUT_ARTIFACT_PATH>"',
        'statistical_report: "<PENDING_PAIRED_ANALYSIS_ARTIFACT_PATH>"',
        'mpc_results: "<PENDING_18_CELL_ARTIFACT_PATH_OR_EXPLICITLY_NOT_RUN>"',
    )
    for phrase in required:
        assert phrase in evidence


def test_citations_resolve_to_verified_bibliography_entries() -> None:
    tex = _read(_MAIN)
    bibliography = _read(_REFERENCES)
    entries = _BIB_ENTRY_RE.findall(bibliography)
    keys = [key.strip() for key, _ in entries]

    assert len(entries) == 13
    assert len(keys) == len(set(keys))
    assert "and others" not in bibliography.lower()

    citations = {
        key.strip() for group in _CITATION_RE.findall(tex) for key in group.split(",")
    }
    assert citations, "the draft must cite its related work"
    assert citations == set(keys)

    for key, body in entries:
        assert re.search(r"(?m)^\s*url\s*=", body), (
            f"{key.strip()} lacks a primary-source URL"
        )


def test_corrected_primary_source_metadata_is_locked() -> None:
    bibliography = _read(_REFERENCES)
    required = (
        "Proceedings of the 42nd International Conference on Machine Learning",
        "volume    = {267}",
        "pages     = {60099--60146}",
        "volume    = {280}",
        "pages     = {1146--1169}",
        "@misc{lecun2022jepa",
        "Lecture Notes in Networks and Systems",
        "doi       = {10.1007/978-981-95-1746-6_46}",
    )
    for phrase in required:
        assert phrase in bibliography
