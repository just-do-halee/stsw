stsw v1.0 – The Last-Word Safe-Tensor Stream Suite

Perfectionist-grade Stream Writer & Stream Reader, designed once so no-one ever has to rewrite them.

⸻

0 Executive one-pager

Goal	Result
Problem	Safely write and lazily read tens-of-GB tensor collections on any hardware, with < 100 MB RAM and zero post-processing.
Solution	stsw – a single, tiny, pure-Python package exposing • StreamWriter → spec-perfect *.safetensors in forward-only streams. • StreamReader → zero-copy, constant-memory iterator over those files.
Compatibility	Bit-level identical to the latest official safetensors spec & reference loader.
DX mantra	“import stsw → do work → close() → done” – no hidden states, no foot-guns, 100 % type-hinted, pyright-clean.


⸻

1 Repository blueprint

stsw/
├─ __init__.py              # public re-exports
├─ _core/
│  ├─ dtype.py              # dtype <-> bytes / torch / numpy maps
│  ├─ header.py             # build/parse/validate header JSON
│  ├─ meta.py               # TensorMeta dataclass (+marshal helpers)
│  └─ crc32.py              # streaming CRC (xxhash fallback)
├─ writer/
│  └─ writer.py             # StreamWriter (sync, thread-safe)
├─ reader/
│  └─ reader.py             # StreamReader (sync)
├─ cli/
│  └─ __main__.py           # “python -m stsw …”
├─ io/
│  ├─ fileio.py             # cross-OS seek/write/fsync helpers
│  └─ mmapwrap.py           # safe mmap with Win fallback
├─ tests/                   # pytest + hypothesis (≥ 1 k cases)
├─ docs/                    # mkdocs-material live site
├─ benchmarks/
├─ pyproject.toml           # PEP 621, wheels via cibuildwheel
└─ CHANGELOG.md

Public import surface is only what stsw/__init__.py re-exports – iron-clad semantic versioning.

⸻

2 On-disk format (immutable)

Exactly the official safetensors layout:

┌─ 8 B  little-endian uint64  = N (header length in bytes)
├─ N B  UTF-8 JSON  (padded to align=64 by spaces, legal per spec)
└─ Data region:   tensor0 | pad | tensor1 | pad | …  (raw little-endian bytes)

data_offsets=[begin,end] are relative to the first byte of the data region, hence independent of header length.

⸻

3 Core abstractions

@dataclass(slots=True, frozen=True)
class TensorMeta:
    name: str
    dtype: Literal['F16','F32','I8', ...]
    shape: tuple[int, ...]
    offset_begin: int   # relative to data region
    offset_end:   int
    crc32: int | None = None   # optional, filled by writer


⸻

4 API spec – 100 % frozen at v1.0

4.1 Public constants

stsw.__version__           # '1.0.0'
stsw.DEFAULT_ALIGN         # 64

4.2 Writer (sync)

class StreamWriter:
    @classmethod
    def open(cls,
             path: os.PathLike | str,
             tensors: Sequence[TensorMeta],
             *,
             align: int = 64,
             buffer_size: int = 4 << 20,         # 4 MiB I/O buffer
             crc32: bool = False                 # per-tensor checksum
             ) -> "StreamWriter": ...

    def write_block(self, name: str, data: bytes | memoryview) -> None: ...
    def finalize_tensor(self, name: str) -> None: ...
    def close(self) -> None: ...
    def abort(self) -> None: ...
    def stats(self) -> "WriterStats": ...        # live telemetry

WriterStats = (written:int, total:int, mb_per_s:float, eta_s:float, rss_mb:float).

4.3 Reader (sync)

class StreamReader:
    def __init__(self,
                 path: os.PathLike | str,
                 *,
                 mmap: bool = True,
                 verify_crc: bool = False) -> None: ...

    def keys(self) -> list[str]: ...
    def meta(self, name: str) -> TensorMeta: ...
    def get_slice(self, name: str) -> memoryview: ...
    def to_torch(self, name: str, *, device="cpu") -> "torch.Tensor": ...
    def to_numpy(self, name: str) -> "numpy.ndarray": ...
    def __iter__(self) -> Iterator[str]: ...      # yields keys in file order
    def close(self) -> None: ...

Both classes are context-manager aware (with … as …:).

⸻

5 Algorithmic contract

5.1 StreamWriter
	1.	Pre−flight
Validate metas: unique names, monotonic non-overlapping offsets, dtype whitelist, offset_end ≡ offset_begin (mod align).
Build padded JSON → write 8 B length + header.
	2.	Data phase (user calls)
For each tensor:
	•	bytes must be written in order; any deviation → TensorOrderError.
	•	cumulative length equals declared size else LengthMismatchError.
	•	optional CRC32 incrementally updated.
After finalize_tensor, writer inserts alignment pad (zeros).
	3.	Close
If CRC enabled → back-patch CRC in header via os.pwrite.
Flush, fsync, atomic rename (path+".tmp" → path).
	4.	Abort / crash
.tmp file auto-deleted in atexit hook; partially written spec-valid file (with "__incomplete__":true) is left on disk if process hard-kills – resumable by StreamWriter.open(..., resume=True) (CLI helper).

5.2 StreamReader

Reads 8 B length → streams header in 256 KiB chunks → validates grammar.
When get_slice/to_torch is invoked:

abs = data_start + meta.offset_begin
return mmap_obj[abs : abs + (meta.offset_end-meta.offset_begin)]

No copy, constant memory; if verify_crc=True, a one-shot CRC32 is computed on first access and cached.

⸻

6 Developer-experience gold plating

DX feature	Implementation
Full type hints	pyright strict in CI, 0 errors.
Rich logging	stsw logger with rich formatting, opt-in environment variable.
Progress bars	Helper stsw.tqdm.wrap(writer) returns tqdm-aware object.
Self-test	python -m stsw.selftest writes 1 GB dummy, reads back, CRC-verifies – for user trust.
One-liner	stsw.dump(state_dict, "model.st", workers=4); internally builds metas, splits tensors into blocks, streams + closes.
CLI	stsw inspect file.st (schema table), stsw verify file.st (CRC check), stsw convert ckpt.pt file.st.


⸻

7 Performance & memory budget

Scenario	File	MB/s write	MB/s read	Peak RSS
NVMe, writer, 4× threads	16 GB	1 .8 GB/s	–	80 MB
Reader mmap → PyTorch GPU	16 GB	–	6.2 GB/s†	50 MB
HDD, adaptive blocks	16 GB	250 MB/s	230 MB/s	80 MB

† limited by cudaMemcpy, not by stsw.

⸻

8 CI / CD pipeline (GitHub Actions)
	1.	Unit + fuzz (pytest -n auto, hypothesis 50 k cases)
	2.	Static – pyright strict, ruff, black.
	3.	Bench regression – ASV, fails if > 5 % slower.
	4.	Wheel build – cibuildwheel (manylinux, musllinux, macOS x86/arm, Win).
	5.	Audit – python -m pip_audit.
	6.	Docs – mkdocs-material, deploy to gh-pages.
	7.	Publish – PyPI trusted-publisher on tag v1.0.*.

All artefacts SHA256-recorded; wheels reproducible (SOURCE_DATE_EPOCH).

⸻

9 Security & robustness extras
	•	Header size cap 100 MB.
	•	JSON depth limit 64; names validated against ^[A-Za-z0-9_.-]{1,300}$.
	•	Only whitelisted dtypes accepted (F16/F32/F64/I8/I16/I32/I64/BF16).
	•	CRC32 poly = 0x1EDC6F41 (Castagnoli), accelerated via zlib.crc32 or xxhash.
	•	Reader refuses files with "__incomplete__": true unless allow_partial=True.

⸻

10 Quick-start cheatsheet

# ---------- write ----------
import torch, stsw, itertools

state = torch.load("ckpt.pt", map_location="cpu")

metas, cursor = [], 0
for k, t in state.items():
    nbytes = t.numel() * t.element_size()
    metas.append(stsw.TensorMeta(
        name=k,
        dtype=stsw.dtype.normalize(t.dtype),
        shape=tuple(t.shape),
        offset_begin=cursor,
        offset_end=cursor+nbytes))
    cursor = (cursor+nbytes + 63)//64*64    # align 64

w = stsw.StreamWriter.open("model.st", metas, crc32=True, buffer_size=8<<20)
for k, t in state.items():
    for chunk in torch.tensor_split(t.view(-1), 4096):      # 4096 ≈ 4 MB
        w.write_block(k, chunk.numpy().tobytes())
    w.finalize_tensor(k)
w.close()

# ---------- read ----------
with stsw.StreamReader("model.st", mmap=True) as r:
    w0 = r.to_torch("embedding.weight", device="cuda")


⸻

11 Licence & governance
	•	Apache-2.0, DCO enforced.
	•	Public RFC process for breaking changes; ≥ 6 month deprecation notice.
	•	Two-maintainer merge rule; both must pass selftest locally.

⸻

Your last proof to the universe

pip install stsw → you possess a tool that cannot be out-engineered
for its purpose within the constraints of physics and CPython.
Nothing left to streamline – only data to move.

---

stsw — Paranoid TDD Protocol

Below is the non-negotiable workflow for every line of code in stsw.
If followed, the repo can never merge a regression in type-safety, behaviour, or performance.

⸻

1 Toolchain (lockfile-pinned)

Category	Tool	Version	Role
Interpreter	CPython	3.12.*	primary target
Type checker	Pyright	1.*	strict mode
Static linter	Ruff	≥ 0.4	style + simple bugs
Formatter	Black	24.*	line length 88
Test runner	Pytest	8.*	unit & integration
Prop/fuzz	Hypothesis	6.*	randomised contracts
Coverage	Coverage.py	7.*	branch + statement
Mutation	Mutmut	latest	spot missing asserts
Perf regression	ASV	master	microbenchmarks
CI	GitHub Actions	n/a	matrix: OS×Py

All versions are exact-pinned in pyproject.toml + requirements.lock (hash-verified).

⸻

2 Repository guards

Layer	Enforcement
Pre-commit	pre-commit hooks run: black –check → ruff → pyright. Commit refuses if any fail.
Pre-push	Full test suite (+ hypothesis health check) runs locally; push aborts on red.
CI gate	Same matrix as local, plus coverage ≥ 95 %, ASV delta ≤ 5 %. A PR cannot merge without all green checks.
Branch protection	main protected; squash-merge only; status checks + 1 maintainer review required.


⸻

3 The red-green-refactor ritual

1. RED   – Write a failing test *first* (unit or property).
2. GREEN – Implement minimal code until test passes.
3. BLUE  – Run: pyright → ruff → black –check → pytest –q.
           · ZERO pyright errors/warnings.
           · ZERO ruff E/F codes; max 3 W per module.
4. REFACTOR – Improve internals, re-run full check suite.
5. PERF – If public API pathway touched, run `asv dev`.

A commit is allowed only if all five stages satisfied.

⸻

4 Test-suite structure

tests/
├─ unit/
│  ├─ test_dtype.py         # pure functions
│  ├─ test_header.py
│  ├─ test_crc32.py
│  └─ …
├─ writer/
│  ├─ test_writer_happy.py
│  ├─ test_writer_errors.py
│  ├─ test_writer_resume.py
│  └─ property_writer.py    # Hypothesis
├─ reader/
│  ├─ test_reader_happy.py
│  ├─ test_reader_errors.py
│  └─ property_reader.py
└─ integ/
   ├─ test_writer_reader_roundtrip.py
   ├─ test_crc_verify_cli.py
   └─ perf_smoke.py

	•	Every public function has at least one unit test.
	•	Every exception branch has an explicit test that asserts the message and context values.
	•	Hypothesis strategies cover: dtype, shape (0-4 D), align powers, partial writes, SIGINT injection.
	•	pytest --cov=stsw --cov-branch fails if coverage < 95 %.

⸻

5 Mutation testing (monthly cron)

mutmut run --paths-to-mutate stsw --runner "pytest -q"
mutmut html

Threshold: surviving mutants ≤ 2 %. Otherwise: mandate new tests.

⸻

6 Performance contracts

ASV benchmarks live in benchmarks/.
Key metrics:

Name	Target
write_throughput_nvme	≥ 1.7 GB/s
write_rss_peak	≤ 100 MB
read_cuda_feed	≥ 6 GB/s

The CI job compares current PR against main. If any metric regresses > 5 %, the job fails.

⸻

7 Fail-fast runtime invariants (prod code)
	•	assert is never stripped – PYTHONOPTIMIZE is banned in runtime env.
	•	typing.assert_never used in all match exhaustiveness checks.
	•	__post_init__ of TensorMeta validates field ranges and patterns.
	•	logging.getLogger("stsw") emits WARNING if any non-fast-path copies detected.

⸻

8 Documentation-driven tests

All code snippets in docs/ are executed via mkdocs-exec during CI. A broken snippet = red build.

⸻

9 Local developer command palette

make type       # pyright strict
make lint       # ruff + black --check
make test       # pytest -n auto --cov
make fuzz       # pytest -m property
make bench-dev  # asv dev -q
make all        # the whole suite above

These make aliases are enforced in onboarding docs; new contributors must run make all before PR.

⸻

10 CI YAML (excerpt)

jobs:
  matrix:
    strategy:
      fail-fast: false
      matrix:
        os:   [ubuntu-latest, macos-latest, windows-latest]
        py:   ['3.9', '3.10', '3.11', '3.12']
    name: ${{ matrix.os }}-py${{ matrix.py }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: ${{ matrix.py }}}
      - run: pip install -e .[dev]
      - run: make type lint test
      - run: pytest -q --hypothesis-show-statistics
  coverage:
    needs: matrix
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: coverage html && coverage xml && coverage report --fail-under=98
  benchmarks:
    needs: matrix
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e .[dev,bench]
      - run: asv continuous main HEAD --factor 1.05


⸻

11 Golden rule (commit message footer)

Every commit must end with

Co-authored-by: test-suite <automation@ci>
Verified-by: stsw-matrix ✔

added automatically by a commit hook only when all local checks are green.

⸻

12 Outcome

Following this obsessive TDD regimen:
	•	Zero unchecked types: pyright blocks merge.
	•	Zero silent regressions: mutant, fuzz, and ASV catch them.
	•	Zero bit-rot: docs run as tests.
	•	Zero surprise behaviour: exhaustive branch coverage + property tests.

stsw thus remains eternally stable, trustworthy, and lightning-fast — the final proof you leave to the universe.

⸻

Your last proof to the universe

pip install stsw → you possess a tool that cannot be out-engineered
for its purpose within the constraints of physics and CPython.
Nothing left to streamline – only data to move.

⸻

stsw v1.0 – The Last-Word Safe-Tensor Stream Suite

Perfectionist-grade Stream Writer & Stream Reader, designed once so no-one ever has to rewrite them.

⸻

0 Executive one-pager

Goal	Result
Problem	Safely write and lazily read tens-of-GB tensor collections on any hardware, with < 100 MB RAM and zero post-processing.
Solution	stsw – a single, tiny, pure-Python package exposing • StreamWriter → spec-perfect *.safetensors in forward-only streams. • StreamReader → zero-copy, constant-memory iterator over those files.
Compatibility	Bit-level identical to the latest official safetensors spec & reference loader.
DX mantra	“import stsw → do work → close() → done” – no hidden states, no foot-guns, 100 % type-hinted, pyright-clean.

# Commit and push to this github repo

https://github.com/just-do-halee/stsw