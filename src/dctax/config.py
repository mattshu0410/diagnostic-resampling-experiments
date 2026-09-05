"""Configuration loading and artifact locations.

Model and cohort definitions are YAML files under `configs/`. Layer count and hidden size are
read from the model checkpoint at runtime; the YAML holds experimental choices and, under
`expected`, values asserted against the checkpoint.

Artifact and dataset roots are overridable with DCTAX_DATA_ROOT and DCTAX_DATASET_DIR.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "configs"


def data_root() -> Path:
    """Root directory for generated artifacts."""
    return Path(os.environ.get("DCTAX_DATA_ROOT", REPO_ROOT / "artifacts")).expanduser()


def dataset_dir() -> Path:
    """Directory holding the source dataset files."""
    return Path(os.environ.get("DCTAX_DATASET_DIR", REPO_ROOT / "datasets")).expanduser()


@dataclass(frozen=True)
class DecodingConfig:
    """Sampling settings for generation.

    A temperature of 0 selects greedy decoding, leaving top_p, top_k and repetition_penalty
    unused. chat_template_kwargs is passed to apply_chat_template, for templates that take a
    flag of their own: gpt-oss reads reasoning_effort, gemma needs enable_thinking since its
    template defaults thinking off. seed fixes the sampler for reproducible generation; None
    leaves it unseeded.
    """

    max_new_tokens: int
    do_sample: bool
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    seed: int | None = None
    chat_template_kwargs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfig:
    """One model's identity, decoding settings, and collection settings.

    slug                   short name used in artifact filenames
    hf_id                  HuggingFace checkpoint id
    dtype                  forward-pass dtype; None keeps the checkpoint's native
                           quantization. Must match the dtype used for generation.
    layers                 layer indices collected for this model
    decoding               sampling settings for generation
    expected_trace_format  envelope collection reads this model's reasoning from. It is tried
                           first and the ordered chain still runs when it matches nothing, so
                           naming it steers extraction without overriding it
    answer_fallbacks       ways of recovering the final diagnosis when this model skips or
                           misuses the <answer> tags, named from utils.answers.FALLBACKS
    sentence_rules         PyRuSH ruleset its reasoning is split under, from traces.RULESETS
    tokenizer_mode         vLLM tokenizer mode; "mistral" builds prompts from messages with
                           the checkpoint's own tokenizer instead of a rendered chat template
    env                    environment under envs/ required to run this model
    expected               values asserted against the loaded checkpoint config
    """

    slug: str
    hf_id: str
    dtype: str | None
    layers: tuple[int, ...]
    decoding: DecodingConfig
    expected_trace_format: str
    answer_fallbacks: tuple[str, ...] = ()
    sentence_rules: str = "clinical"
    tokenizer_mode: str | None = None
    env: str | None = None
    expected: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SourceConfig:
    """One dataset feeding a cohort.

    kind            "csv" for a file under the dataset directory, "hf" for a HuggingFace dataset
    path            filename or dataset id
    case_field      column holding the case presentation text
    answer_field    column holding the gold diagnosis
    expected_count  number of usable cases this source is expected to yield
    """

    name: str
    kind: str
    path: str
    case_field: str
    answer_field: str
    split: str | None = None
    expected_count: int | None = None


@dataclass(frozen=True)
class CohortConfig:
    """A set of cases assembled from one or more sources.

    Cases are deduplicated by case id, keeping the first occurrence.
    """

    name: str
    sources: tuple[SourceConfig, ...]
    prompt: str
    expected_size: int | None = None


@dataclass(frozen=True)
class TaxonomyConfig:
    """One taxonomy run.

    roster            model slugs contributing to the barycenter
    cohort_size       the n_examples component of the activation filenames to read
    layers            layer per slug; omit when layer_index is given
    layer_index       position in each model's collected layers, 0-based
    n_cases           cases sampled from those shared by the roster
    alpha             structure weight in the fused objective, 1 structure only, unused by kmeans
    template_size     support points in the barycenter, and groups when grouping is direct
    marginals         uniform | kmeans
    mass_clusters     k-means clusters used to equalise sentence mass under kmeans marginals
    variant           fgw | srfgw | kmeans
    dtype             precision for the distance matrices and the solve
    degenerate_filter drop cases where any model repeats a sentence
    """

    name: str
    roster: tuple[str, ...]
    cohort_size: int = 2073
    layers: dict[str, int] | None = None
    layer_index: int | None = None
    n_cases: int = 1000
    seed: int = 0
    alpha: float = 0.8
    template_size: int = 8
    marginals: str = "uniform"
    mass_clusters: int = 8
    variant: str = "fgw"
    max_iter: int = 30
    dtype: str = "float32"
    embed_model: str = "text-embedding-3-large"
    degenerate_filter: bool = True
    degen_min_words: int = 6
    degen_min_rep: int = 3

    def resolve_layers(self) -> dict[str, int]:
        """Layer to read for each model in the roster."""
        if self.layers is not None:
            missing = sorted(set(self.roster) - set(self.layers))
            if missing:
                raise ValueError(f"{self.name}: no layer given for {missing}")
            return {slug: self.layers[slug] for slug in self.roster}
        if self.layer_index is None:
            raise ValueError(f"{self.name}: set either layers or layer_index")
        resolved = {}
        for slug in self.roster:
            collected = load_model(slug).layers
            if not -len(collected) <= self.layer_index < len(collected):
                raise ValueError(f"{slug}: layer_index {self.layer_index} outside {list(collected)}")
            resolved[slug] = collected[self.layer_index]
        return resolved


def load_experiment(name: str) -> TaxonomyConfig:
    """Load configs/experiments/<name>.yaml."""
    raw = _read_yaml(CONFIG_ROOT / "experiments" / f"{name}.yaml")
    known = {f.name for f in fields(TaxonomyConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"{name}: unrecognised keys {unknown}")
    raw["roster"] = tuple(raw["roster"])
    return TaxonomyConfig(**raw)


@dataclass(frozen=True)
class ResampleConfig:
    """One resampling experiment, from case selection through to which solves to assess.

    name              names the pool, the runs and the outputs
    models            slugs resampled, all sharing one pool so results stay comparable
    solve             taxonomy run the sentences are joined to
    strata            unused by the pool draw; kept so existing experiment files still load
    cohort            cohort the pool is drawn from
    pool_size         cases to screen, spread as evenly over the cohort's sources as it allows
    core_min          models a case must qualify for to enter the shared set every model sweeps
    floor             cases a model is topped up to when the shared set leaves it short
    min_each          correct and wrong screening samples a case needs to give a sweepable pair
    max_sentences     longest trace to resample, since sweep cost scales with boundaries
    seed              fixes the pool draw, so any machine reproduces the same cases
    screen_rollouts   rollouts at boundary 0 when measuring how far a model's answer moves
    sweep_rollouts    rollouts at every boundary, for the cases selected from the screen
    max_new_tokens    per-rollout ceiling; a case whose screen reaches it is excluded
    n_variable        most variable cases to sweep
    n_control         least variable cases, which read as the metric's floor
    threshold         cosine similarity below which a regenerated sentence counts as different
    sizes             group counts to assess, each the named solve with its K swapped
    """

    name: str
    models: tuple[str, ...]
    solve: str
    strata: dict[int, int]
    cohort: str = "main2073"
    pool_size: int = 1000
    core_min: int = 6
    floor: int = 40
    min_each: int = 1
    max_sentences: int = 60
    seed: int = 0
    screen_rollouts: int = 20
    sweep_rollouts: int = 100
    max_new_tokens: int = 8192
    n_variable: int = 60
    n_control: int = 10
    threshold: float = 0.8
    sizes: tuple[int, ...] = (4, 6, 8)

    def screen_run(self, slug: str) -> str:
        return f"{self.name}-{slug}-screen"

    def sweep_run(self, slug: str) -> str:
        return f"{self.name}-{slug}"

    def solve_at(self, size: int) -> str:
        """The named solve with its group count swapped for `size`."""
        return re.sub(r"_K\d+_", f"_K{size}_", self.solve)


def load_resample(name: str) -> ResampleConfig:
    """Load configs/resampling/<name>.yaml."""
    raw = _read_yaml(CONFIG_ROOT / "resampling" / f"{name}.yaml")
    known = {f.name for f in fields(ResampleConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"{name}: unrecognised keys {unknown}")
    raw["models"] = tuple(raw["models"])
    raw["strata"] = {int(k): int(v) for k, v in raw["strata"].items()}
    if "sizes" in raw:
        raw["sizes"] = tuple(raw["sizes"])
    return ResampleConfig(**raw)


def _read_yaml(path: Path) -> dict:
    with path.open() as handle:
        return yaml.safe_load(handle)


def load_model(slug: str) -> ModelConfig:
    """Load configs/models/<slug>.yaml."""
    raw = _read_yaml(CONFIG_ROOT / "models" / f"{slug}.yaml")
    dec = raw["decoding"]
    return ModelConfig(
        slug=raw["slug"],
        hf_id=raw["hf_id"],
        dtype=raw.get("dtype"),
        layers=tuple(raw["layers"]),
        decoding=DecodingConfig(
            max_new_tokens=dec["max_new_tokens"],
            do_sample=dec["do_sample"],
            temperature=dec.get("temperature"),
            top_p=dec.get("top_p"),
            top_k=dec.get("top_k"),
            repetition_penalty=dec.get("repetition_penalty"),
            seed=dec.get("seed"),
            chat_template_kwargs=dict(dec.get("chat_template_kwargs", {})),
        ),
        expected_trace_format=raw["expected_trace_format"],
        answer_fallbacks=tuple(raw.get("answer_fallbacks", ())),
        sentence_rules=raw.get("sentence_rules", "clinical"),
        tokenizer_mode=raw.get("tokenizer_mode"),
        env=raw.get("env"),
        expected=raw.get("expected", {}),
    )


def all_model_slugs() -> list[str]:
    """Slugs of every model config present, sorted."""
    return sorted(p.stem for p in (CONFIG_ROOT / "models").glob("*.yaml"))


def load_cohort(name: str) -> CohortConfig:
    """Load configs/cohorts/<name>.yaml."""
    raw = _read_yaml(CONFIG_ROOT / "cohorts" / f"{name}.yaml")
    sources = tuple(
        SourceConfig(
            name=s["name"],
            kind=s["kind"],
            path=s["path"],
            case_field=s["case_field"],
            answer_field=s["answer_field"],
            split=s.get("split"),
            expected_count=s.get("expected_count"),
        )
        for s in raw["sources"]
    )
    return CohortConfig(
        name=raw["name"],
        sources=sources,
        prompt=raw["prompt"],
        expected_size=raw.get("expected_size"),
    )


def load_prompt(name: str) -> str:
    """Load configs/prompts/<name>.txt, stripping the trailing newline."""
    return (CONFIG_ROOT / "prompts" / f"{name}.txt").read_text().rstrip("\n")
