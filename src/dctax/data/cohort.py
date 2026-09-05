"""Assembling the case cohort from its source datasets.

Sources are read in the order the cohort lists them, each one deduplicated by case id on its
own, and the results concatenated. The case id is the md5 of the case text with every non-letter removed.

Source files are read from `DCTAX_DATASET_DIR`. `expected_count` and `expected_size` from the
config are asserted during the build.

    dctax cohort --cohort main2073

    cfg = config.load_cohort("main2073")
    cases = cohort.build(cfg)
    prompts = cohort.questions(cases, cfg)
    cohort.write(cfg, cases)
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from dctax.config import CohortConfig, SourceConfig, data_root, dataset_dir, load_prompt

# nejm_cpc_dataset.csv holds case presentations far longer than the default field limit.
csv.field_size_limit(sys.maxsize)

LETTERS = re.compile(r"[^a-zA-Z]")


@dataclass(frozen=True)
class Case:
    """One case presentation and its gold diagnosis.

    case_id  md5 of the case text reduced to letters
    source   the cohort source it was read from
    case     the case presentation shown to the model
    answer   the gold diagnosis
    """

    case_id: str
    source: str
    case: str
    answer: str


def case_id(text: str) -> str:
    """Canonical case id: md5 of the case text with non-letters removed."""
    return hashlib.md5(LETTERS.sub("", text).encode("utf-8")).hexdigest()


def _rows(source: SourceConfig, directory: Path) -> list[dict]:
    """Raw records from one source, before any field is read."""
    if source.kind == "csv":
        with (directory / source.path).open(encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    if source.kind == "hf":
        from datasets import load_dataset

        # A bundled copy under the dataset directory is preferred over the Hub, so a
        # source that ships with the repository resolves without a network call.
        local = directory / source.path
        target = str(local) if local.exists() else source.path
        return list(load_dataset(target, split=source.split or "train"))
    raise ValueError(f"{source.name}: unknown kind {source.kind!r}")


def read_source(source: SourceConfig, directory: Path | None = None) -> list[Case]:
    """Cases from one source, blank ones dropped and the rest deduplicated by case id."""
    directory = directory or dataset_dir()
    seen: set[str] = set()
    cases = []
    for row in _rows(source, directory):
        # Case text is kept exactly as the source holds it, trailing whitespace included, since
        # it goes into the prompt verbatim. Only the blank check looks past that whitespace.
        text = str(row.get(source.case_field) or "")
        if not text.strip():
            continue
        identifier = case_id(text)
        if identifier in seen:
            continue
        seen.add(identifier)
        cases.append(
            Case(
                case_id=identifier,
                source=source.name,
                case=text,
                answer=str(row.get(source.answer_field) or "").strip(),
            )
        )
    return cases


def build(cohort: CohortConfig, directory: Path | None = None, *, verbose: bool = True) -> list[Case]:
    """Every case in the cohort, checked against the counts the config expects."""
    cases: list[Case] = []
    for source in cohort.sources:
        found = read_source(source, directory)
        if verbose:
            print(f"  {source.name}: {len(found)} cases", flush=True)
        if source.expected_count is not None and len(found) != source.expected_count:
            raise ValueError(
                f"{source.name}: {len(found)} cases, expected {source.expected_count}"
            )
        cases.extend(found)

    if cohort.expected_size is not None and len(cases) != cohort.expected_size:
        raise ValueError(f"{cohort.name}: {len(cases)} cases, expected {cohort.expected_size}")
    return cases


def questions(cases: list[Case], cohort: CohortConfig) -> list[str]:
    """The prompt shown to a model for each case."""
    template = load_prompt(cohort.prompt)
    return [template.format(case_prompt=case.case) for case in cases]


def path_for(name: str) -> Path:
    return data_root() / "cohorts" / f"{name}.json"


def write(cohort: CohortConfig, cases: list[Case], path: Path | None = None) -> Path:
    """Write the cohort as JSON, in the order the cases were assembled."""
    path = path or path_for(cohort.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump([asdict(case) for case in cases], handle, indent=1)
    return path


def load(name: str, path: Path | None = None) -> list[Case]:
    """Read a cohort written by `write`."""
    path = path or path_for(name)
    with path.open() as handle:
        return [Case(**record) for record in json.load(handle)]
