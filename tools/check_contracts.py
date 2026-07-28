#!/usr/bin/env python3
"""Validate coverage matrices and skipped-test issue links without network access."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 and older
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
COVERAGE = ROOT / "spec" / "coverage"
UPSTREAM = ROOT / "spec" / "upstream"
NATIVE_LANGUAGE_CONTRACT = (
    ROOT / "spec" / "api" / "thouless-native-languages.toml"
)
NATIVE_LANGUAGE_DESIGN = ROOT / "docs" / "native-language-api-design.md"
DOMAIN_API_PROPOSAL = ROOT / "spec" / "api" / "thouless-domain-100.toml"
AGENT_ENTRYPOINT = ROOT / "AGENTS.md"
AGENT_INSTRUCTION = ROOT / "instructions" / "scientific-software-reimplementation.md"
AGENT_INSTRUCTION_REFERENCE = "instructions/scientific-software-reimplementation.md"
ISSUE_PATTERN = re.compile(
    r"^https://github\.com/matrixlab-research/thouless/issues/[1-9][0-9]*$"
)
ISSUE_IN_TEXT_PATTERN = re.compile(
    r"https://github\.com/matrixlab-research/thouless/issues/[1-9][0-9]*"
)
VALID_STATUSES = {"implemented", "partial", "missing", "blocked"}
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
TBQ_PATTERN = re.compile(r"^TBQ-([0-9]{3})$")
EXPECTED_DOMAIN_SUITES = (
    "01-model-construction",
    "02-bands-dos-fermiology",
    "03-magnetic-flux-hofstadter",
    "04-bulk-topology",
    "05-boundaries-bulk-boundary",
    "06-quantum-geometry-response",
    "07-disorder-localization",
    "08-open-transport",
    "09-superconducting-bdg",
    "10-non-hermitian",
    "11-floquet-dynamics",
    "12-interactions-self-consistency",
    "13-moire-strain-supercells",
    "14-magnetism-spin-orbital",
    "15-optical-thermoelectric",
    "16-aperiodic-amorphous-fractal",
    "17-defects-interfaces",
    "18-multiscale-validation",
    "19-scientific-scale-numerics",
    "20-inference-inverse-design",
)


def fail(message: str) -> None:
    print(f"contract check failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for key in ("capability", "interface"):
        result.extend(document.get(key, []))
    return result


def check_matrix(path: Path) -> None:
    with path.open("rb") as source:
        document = tomllib.load(source)
    matrix_rows = rows(document)
    if not matrix_rows:
        fail(f"{path.relative_to(ROOT)} has no coverage rows")

    for index, row in enumerate(matrix_rows, start=1):
        status = row.get("status")
        if status not in VALID_STATUSES:
            fail(f"{path.name} row {index} has invalid status {status!r}")
        if status != "implemented" and not ISSUE_PATTERN.match(row.get("issue", "")):
            fail(f"{path.name} row {index} has no valid gap issue")

    if "source_package" in document:
        inventory_status = document.get("inventory_status")
        if inventory_status == "incomplete":
            if not ISSUE_PATTERN.match(document.get("inventory_issue", "")):
                fail(
                    f"{path.name} has an incomplete inventory without an issue"
                )
        elif inventory_status == "complete":
            api_manifest = document.get("api_manifest", "")
            if not api_manifest.startswith("spec/api/"):
                fail(f"{path.name} has no repository API manifest")
            if not (ROOT / api_manifest).is_file():
                fail(f"{path.name} references a missing API manifest")
        else:
            fail(
                f"{path.name} has invalid inventory status "
                f"{inventory_status!r}"
            )


def check_compatibility_test(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    if not ISSUE_IN_TEXT_PATTERN.search(source):
        fail(f"{path.relative_to(ROOT)} does not link its gap issue")
    if "require_compat_module" not in source:
        fail(f"{path.relative_to(ROOT)} can bypass the compatibility import guard")


def check_upstream_manifest(path: Path) -> None:
    with path.open("rb") as source:
        document = tomllib.load(source)
    if not COMMIT_PATTERN.match(document.get("commit", "")):
        fail(f"{path.name} does not pin an exact source commit")
    if not ISSUE_PATTERN.match(document.get("gap_issue", "")):
        fail(f"{path.name} does not link its compatibility gap issue")
    expected_skips = document.get("expected_skip", [])
    declared_skips = document.get("strict_skips", 0)
    if declared_skips != len(expected_skips):
        fail(
            f"{path.name} declares {declared_skips} skips but "
            f"documents {len(expected_skips)}"
        )
    if "strict_passes" in document:
        if document["strict_passes"] + declared_skips != document["strict_tests"]:
            fail(f"{path.name} pass and skip counts do not equal strict_tests")
    for skip in expected_skips:
        if not skip.get("node") or not skip.get("reason"):
            fail(f"{path.name} has an under-specified expected skip")
        if not ISSUE_PATTERN.match(skip.get("issue", "")):
            fail(f"{path.name} has an expected skip without a valid issue")
    status = document.get("collection_status")
    if status == "complete":
        if document.get("collected_tests", 0) < 1:
            fail(f"{path.name} has no collected test count")
    elif status == "blocked":
        if not document.get("collection_blocker"):
            fail(f"{path.name} has a blocked collection without a reason")
    else:
        fail(f"{path.name} has invalid collection status {status!r}")


def check_agent_instructions() -> None:
    if not AGENT_ENTRYPOINT.is_file():
        fail("AGENTS.md is missing")
    if not AGENT_INSTRUCTION.is_file():
        fail(f"{AGENT_INSTRUCTION.relative_to(ROOT)} is missing")

    entrypoint = AGENT_ENTRYPOINT.read_text(encoding="utf-8")
    if AGENT_INSTRUCTION_REFERENCE not in entrypoint:
        fail("AGENTS.md does not point to the repository reimplementation instruction")

    instruction = AGENT_INSTRUCTION.read_text(encoding="utf-8")
    required_terms = (
        "Rust-native API",
        "PythTB 2.0",
        "Kwant 1.5",
        "GitHub Issues",
        "held-out",
        "Prohibit Fitting to Known Tests",
    )
    missing_terms = [term for term in required_terms if term not in instruction]
    if missing_terms:
        fail(f"agent instruction is missing required terms: {missing_terms}")


def check_native_language_contract() -> None:
    if not NATIVE_LANGUAGE_DESIGN.is_file():
        fail(f"{NATIVE_LANGUAGE_DESIGN.relative_to(ROOT)} is missing")
    if not NATIVE_LANGUAGE_CONTRACT.is_file():
        fail(f"{NATIVE_LANGUAGE_CONTRACT.relative_to(ROOT)} is missing")

    instruction = AGENT_INSTRUCTION.read_text(encoding="utf-8")
    for reference in (
        "../docs/native-language-api-design.md",
        "../spec/api/thouless-native-languages.toml",
    ):
        if reference not in instruction:
            fail(f"agent instruction does not reference {reference}")

    with NATIVE_LANGUAGE_CONTRACT.open("rb") as source:
        contract = tomllib.load(source)
    if contract.get("status") not in {"design", "stable"}:
        fail("native language contract status must be design or stable")
    if not contract.get("contract_version"):
        fail("native language contract has no version")
    if set(contract.get("languages", {})) != {"rust", "python", "julia"}:
        fail("native language contract must name Rust, Python, and Julia")
    tracking = contract.get("tracking", {})
    expected_tracking = {
        "rust_contract",
        "python_native",
        "julia_native",
        "cross_language_ci",
    }
    if set(tracking) != expected_tracking:
        fail("native language contract has incomplete issue tracking")
    for issue in tracking.values():
        if not ISSUE_PATTERN.match(issue):
            fail("native language contract has an invalid gap issue")

    workflows = contract.get("workflow", [])
    identifiers = [workflow.get("id") for workflow in workflows]
    if len(identifiers) != len(set(identifiers)):
        fail("native language contract contains duplicate workflow identifiers")
    for index, workflow in enumerate(workflows, start=1):
        for language in ("rust", "python", "julia"):
            if not workflow.get(language):
                fail(
                    "native language contract workflow "
                    f"{index} has no {language} namespace"
                )

    with (COVERAGE / "native.toml").open("rb") as source:
        native = tomllib.load(source)
    expected = {
        capability["id"]
        for capability in native.get("capability", [])
        if capability["id"] != "validation.held_out"
    }
    actual = set(identifiers)
    if expected != actual:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        fail(
            "native language contract does not match native capabilities: "
            f"missing={missing}, unexpected={unexpected}"
        )


def check_domain_api_proposal() -> None:
    if not DOMAIN_API_PROPOSAL.is_file():
        fail(f"{DOMAIN_API_PROPOSAL.relative_to(ROOT)} is missing")
    with DOMAIN_API_PROPOSAL.open("rb") as source:
        proposal = tomllib.load(source)

    if proposal.get("schema_version") != 1:
        fail("domain API proposal has an unsupported schema")
    if proposal.get("status") != "proposal":
        fail("domain API design must remain a proposal until implemented")
    if proposal.get("question_count") != 100:
        fail("domain API proposal must declare 100 questions")
    if proposal.get("suite_count") != 20:
        fail("domain API proposal must declare 20 suites")
    if not COMMIT_PATTERN.match(proposal.get("benchmark_commit", "")):
        fail("domain API proposal does not pin the benchmark commit")
    if not ISSUE_PATTERN.match(proposal.get("tracking_issue", "")):
        fail("domain API proposal has no valid tracking issue")
    for field in ("design_document", "stable_contract"):
        reference = proposal.get(field, "")
        if not reference or not (ROOT / reference).is_file():
            fail(f"domain API proposal references missing {field}")

    concepts = proposal.get("concept", [])
    concept_ids = [row.get("id") for row in concepts]
    if not concept_ids or len(concept_ids) != len(set(concept_ids)):
        fail("domain API proposal has missing or duplicate concepts")
    for row in concepts:
        if row.get("status") not in {"implemented", "partial", "missing"}:
            fail(f"domain API concept {row.get('id')} has invalid status")
        if row.get("status") != "implemented":
            issue = row.get("issue", "")
            if not ISSUE_PATTERN.match(issue):
                fail(f"domain API concept {row.get('id')} has no gap issue")
        if not row.get("summary"):
            fail(f"domain API concept {row.get('id')} has no summary")

    request_families = proposal.get("request_family", [])
    request_ids = [row.get("id") for row in request_families]
    if not request_ids or len(request_ids) != len(set(request_ids)):
        fail("domain API proposal has missing or duplicate request families")
    for row in request_families:
        if not row.get("module") or not row.get("examples"):
            fail(f"request family {row.get('id')} is under-specified")

    coverage = proposal.get("coverage", [])
    suites = [row.get("suite") for row in coverage]
    if len(coverage) != 20 or len(suites) != len(set(suites)):
        fail("domain API proposal must contain 20 unique suite rows")
    if tuple(suites) != EXPECTED_DOMAIN_SUITES:
        fail("domain API proposal does not preserve the pinned suite catalog")

    seen_questions: list[str] = []
    known_concepts = set(concept_ids)
    known_requests = set(request_ids)
    for suite_index, row in enumerate(coverage):
        questions = row.get("questions", [])
        if len(questions) != 5:
            fail(f"suite {row.get('suite')} must map exactly five questions")
        first = suite_index * 5 + 1
        expected_questions = {
            f"TBQ-{number:03d}" for number in range(first, first + 5)
        }
        if set(questions) != expected_questions:
            fail(
                f"suite {row.get('suite')} does not map its pinned questions"
            )
        seen_questions.extend(questions)
        if not row.get("concepts") or not row.get("requests"):
            fail(f"suite {row.get('suite')} has an empty API composition")
        unknown_concepts = set(row["concepts"]) - known_concepts
        unknown_requests = set(row["requests"]) - known_requests
        if unknown_concepts:
            fail(
                f"suite {row.get('suite')} references unknown concepts "
                f"{sorted(unknown_concepts)}"
            )
        if unknown_requests:
            fail(
                f"suite {row.get('suite')} references unknown requests "
                f"{sorted(unknown_requests)}"
            )

    if len(seen_questions) != len(set(seen_questions)):
        fail("domain API proposal contains duplicate TBQ identifiers")
    actual_numbers: set[int] = set()
    for question in seen_questions:
        match = TBQ_PATTERN.match(question)
        if match is None:
            fail(f"invalid domain question identifier {question!r}")
        actual_numbers.add(int(match.group(1)))
    expected_numbers = set(range(1, 101))
    if actual_numbers != expected_numbers:
        fail(
            "domain API proposal is not a complete TBQ-001..TBQ-100 mapping: "
            f"missing={sorted(expected_numbers - actual_numbers)}, "
            f"extra={sorted(actual_numbers - expected_numbers)}"
        )


def main() -> None:
    check_agent_instructions()
    check_native_language_contract()
    check_domain_api_proposal()
    matrices = sorted(COVERAGE.glob("*.toml"))
    if not matrices:
        fail("no coverage matrices found")
    for matrix in matrices:
        check_matrix(matrix)

    manifests = sorted(UPSTREAM.glob("*.toml"))
    if not manifests:
        fail("no upstream test manifests found")
    for manifest in manifests:
        check_upstream_manifest(manifest)

    tests = sorted((ROOT / "compat-tests").glob("test_*.py"))
    if not tests:
        fail("no compatibility smoke tests found")
    for test in tests:
        check_compatibility_test(test)

    print(
        "validated agent instructions, native language design, "
        "100-question domain API proposal, "
        f"{len(matrices)} coverage matrices, {len(manifests)} upstream "
        f"manifests, and {len(tests)} test modules"
    )


if __name__ == "__main__":
    main()
