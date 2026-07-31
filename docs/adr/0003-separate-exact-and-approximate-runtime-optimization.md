# Separate exact and approximate runtime optimization

The runtime improvement harness will evaluate two explicit lanes. The exact
lane accepts only behavior-preserving changes and can promote a measured
speedup automatically; the approximate lane permits bounded score drift but
requires near-miss separation and score-drift safety checks before human
promotion. Keeping the lanes separate lets routine cache and data-flow
refactors move quickly without allowing approximate computation changes to
inherit an equivalence claim they have not earned.

For the initial approximate lane, characterize uniform score drift by fitting
one global baseline-to-candidate score mapping. At least 95% of scored pairs
must fall within 0.01 score of that mapping, and the mapped decrease may not
exceed 5% relative. This is a provisional review-queue heuristic, not authority
for autonomous promotion. Pairs outside that envelope are surfaced rather than
silently discarded as outliers.

The autonomous loop has three broad dispositions: exact behavior-preserving
speedups may be applied automatically; candidates needing approximate,
target-hardware, or deployment review are retained with their evidence but are
not applied to the accepted line; candidates that violate a hard guard or lack
a reliable speedup are rejected. Review candidates are stored as explicit
experiment bundles or immutable candidate commits rather than long-lived Git
stashes so their code diff, parent revision, corpus, measurements, and reason
for review remain inspectable.

Automatic application requires an identical full-precision behavioral ledger,
pixel-identical corpus masks whenever preparation code is touched, and passing
focused tests. There is no numerical tolerance in this exact lane. A change
such as reducing locked rotation-search resolution from 256 to 128 belongs in
the approximate lane: retain its patch and report its live and diagnostic
speedups, score mapping and outliers, near-miss margins, threshold and ranking
changes, fragile-case effects, and a promotion recommendation with confidence.

Exact parity also covers the live workflow's required side effects: the
decision export and production claim-QC artifact set must remain complete, and
decoded QC pixels must match. This prevents fake speedups that merely skip
durable output work. Diagnostic baseline-versus-candidate panels are excluded;
they exist only for approximate review and are expected to differ.

The current `PASS_THRESHOLD` is uncalibrated, so a new threshold crossing is
reported but is not a hard rejection rule for a review candidate. Likewise,
the sign of a near-zero near-miss margin is not treated as ground truth: a
small change such as +0.01 to 0.00 contributes to the recommendation but does
not veto retention. Approximate candidates are judged by the size and breadth
of their near-miss margin changes, with special attention to material losses,
rather than by raw sign-flip counts alone.

Initially, a per-block near-miss margin loss of at most 0.02 is negligible, a
loss greater than 0.02 through 0.05 is material and requires inspection, and a
loss greater than 0.05 is severe. These are recommendation bands rather than
rejection gates. Before trusting them, repeated unchanged baseline collections
must show that normal measurement variance is comfortably smaller than 0.02.

Review candidates receive one of `STRONG_PROMOTE`, `PROMOTE`,
`REVIEW_EXPERIMENT`, or `DO_NOT_PROMOTE`, plus an explicit confidence level.
The label is advisory only: the user's decision on every approximate candidate
comes from the attached evidence. Exact candidates may advance autonomously
because their behavior-equivalence checks, rather than a qualitative rating,
provide the authority.

Each autonomous campaign runs in a newly created Git worktree on its own
optimization branch. Exact accepted changes commit there by default so the
surviving sequence is durable and reviewable. Approximate candidates are
exported as patch-and-report bundles and removed from the active worktree
before the next iteration. The user's main checkout is never the campaign
workspace.

Candidate diffs are restricted to production code under `code/`, further
narrowed by a campaign-specific editable-file allowlist. The loop may execute
tests and diagnostics but may not modify the harness, comparator, tests,
manifests, frozen corpus or baselines, acceptance configuration, or campaign
history. A candidate that changes a protected path is invalid regardless of
its measured result.

Machine-level CPU tuning is outside the optimization search space. Candidates
that alter CPU affinity, process priority, operating-system power settings, or
CPU/backend environment configuration are invalid even when behavior-exact.
The campaign may record and stabilize such settings for measurement
repeatability, but it does not optimize the machine configuration. Code-level
parallelism over independent pipeline work is allowed and may choose a worker
count inside production code, subject to exact behavior parity and the normal
runtime gates.

Each iteration tests one coherent performance hypothesis. It may touch several
allowlisted production files only when those edits are necessary parts of the
same change. Mixed candidates that combine independently measurable ideas are
invalid because their runtime and behavior effects cannot be attributed.

The experiment workload is fixed rather than the experiment duration: every
candidate that reaches full evaluation processes the same frozen corpus under
the same warmup and repetition policy. A campaign has no intrinsic iteration
limit and continues until the user stops it, its configured `run_until`
deadline arrives, or subscription-backed work cannot continue.

The complete 41-set software processing session is the primary campaign
objective: prepare and persist all blocks, then process all slides through
durable verdict and required QC completion. The slide-arrival-to-verdict phase
remains a protected operator-facing metric and may not regress beyond measured
noise even when total session time improves. All-pairs diagnostic throughput
is measured separately, never blended into the session score, and its collector
must reuse the production preparation, gate, routing, alignment, and scoring
seams. It is both a secondary optimization target and a parity check against a
diagnostic-only scoring implementation drifting away from production.

The frozen v3 processing workload contains all 41 true claimed pairs. Its block
phase prepares and persists all 41 blocks; its slide phase processes all 41
slides through the real live claim path and ends only after every verdict and
required QC artifact is durable. Report full-session total, phase totals, and
the per-slide median, p90, p95, and maximum. Do not
substitute `run_claim_pipeline`, which prepares each block beside its slide and
therefore measures a different workflow.

Every runtime result is decomposed into image decode/load, block setup,
slide preparation, locked-score cache construction, quality gates, locked
alignment/pair scoring, verdict and QC serialization, and end-to-end time. The
collector may add diagnostic-only metric time for all-pairs runs. This stage
ledger lets the loop attribute a speedup and exposes improvements that merely
move cost into another phase.

Authoritative acceptance timing runs the normal production path with no stage
observer. Full-session, block-phase, and protected slide-phase boundaries are
timed externally. Detailed internal attribution is collected in a separate
fresh-subprocess profiling pass and is explicitly non-authoritative, preventing
a candidate from winning by optimizing only the instrumented path.

For candidate-versus-parent timing, warm each version once and then collect
five measured full-corpus runs per version in alternating order. Compare paired
runtime ratios and report a confidence interval rather than accepting a single
run. Pin and record Python, dependency, OpenCV thread, and relevant machine
settings so environment drift is visible.

At campaign startup, collect a one-minute idle CPU-load calibration while no
worker or harness task is running. Before each timing batch, require AC power,
the baseline power plan, and no concurrent benchmark, then wait at most 60
seconds for a short CPU sample to return near that campaign-relative baseline.
Do not tune the machine or wait indefinitely; if no quiet window appears, run
the batch and let the paired variance and contamination rules decide whether
its evidence is usable.

Concretely, readiness means the latest five-second average total CPU load is at
or below the startup calibration's p90. This is a bounded scheduling hint, not
an acceptance metric.

The production deployment target uses an `Intel(R) Core(TM) i7-10700T CPU @
2.00GHz`. This is target-hardware provenance, not a claim that the campaign's
development benchmark machine uses that processor. The controller records the
actual benchmark-machine identity separately with every run.

All campaign timing comparisons run on the same designated laptop. Results
from different physical PCs are never combined, and changing the benchmark
machine requires a new baseline lineage. Production-hardware confirmation is
not a general promotion gate, so reports describe gains as measured on the
campaign laptop rather than promising the same percentage on the deployment
CPU.

Code-level parallelism is the hardware-sensitive exception to that general
rule. Before it becomes production-promotable, run a separate paired
parent-versus-candidate benchmark on the i7-10700T production PC. Laptop and
production-PC samples are reported as separate measurement lineages and are
never compared directly.

A parallelism candidate is committed as an immutable candidate snapshot on a
dedicated branch rooted at its tested parent. `TARGET_VALIDATION_REQUIRED`
means "committed but not on the accepted optimization line." The branch can be
pushed so the production PC benchmarks the exact parent and candidate commit
hashes. If the accepted line advances before later promotion, the candidate
must be rebased and reevaluated rather than assuming its old timing still
applies.

Experiments may introduce a new runtime dependency, but such a candidate is
`DEPLOYMENT_VALIDATION_REQUIRED` even when behavior-exact. Evaluate it in an
isolated environment, lock its version and transitive dependency set, and
prove that a clean Windows virtual environment can install it and pass the
full pipeline and test suite. A WSL-only installation does not validate the
current Windows deployment target.

Dependency candidates never promote automatically after technical validation;
the user decides whether their measured benefit justifies the continuing
security, licensing, update, and deployment obligation. The research strategy
deprioritizes them unless evidence indicates an obviously significant
full-session gain.

A dependency candidate reaches deployment review only when its full-session
paired median improves by at least 10% and the 95% confidence interval's lower
bound exceeds 5%. Candidates below that higher bar remain ordinary rejected
experiments even if they install and test successfully.

A qualifying dependency experiment is preserved as
`DEPLOYMENT_VALIDATION_REQUIRED` until its clean-install evidence and measured
benefit receive user review. It does not enter the accepted optimization line
automatically.

`campaign.json` may explicitly authorize the controller to push
`TARGET_VALIDATION_REQUIRED` branches to the configured `origin` remote. That
authority is limited to the campaign's dedicated candidate-branch namespace;
the controller may not push or rewrite `main` or any other protected branch.

The default is manual transport after laptop review. A target-validation branch
remains local until the user invokes an explicit send command for its experiment
ID. The overnight loop records the pending state and continues from the latest
accepted parent rather than stalling or pushing automatically.

On the production PC, one validation command fetches the candidate branch,
checks out the exact recorded parent and candidate hashes in isolated
worktrees, runs the paired target-hardware timing protocol, commits the compact
validation CSV/JSON, and pushes that evidence back to the same candidate
branch. Promotion consumes that returned evidence; it does not rely on a
manually copied timing summary.

A timing batch is contaminated and retried when either version's end-to-end
coefficient of variation exceeds 3% or an individual run differs from that
version's median by more than 5%. After three contaminated batches, classify
the candidate as `INCONCLUSIVE` and do not commit it automatically.

Every attempted hypothesis, including rejected and inconclusive experiments,
must enter a durable experiment history with a one-line summary and a linked
detailed record. The loop reads this history before proposing its next change
so it can avoid repetition and build on earlier evidence. Candidate code is
still production-only; the controller, not the candidate edit, owns these
history writes.

Preservation is tiered. Accepted code is preserved by its commit. Approximate,
target-validation, deployment-validation, and inconclusive candidates preserve
their full patch or immutable candidate commit so they can be reviewed or
rerun exactly. A clearly rejected candidate preserves only its structured
hypothesis, touched functions, implementation summary, measurements, rejection
reason, and lesson; retaining every failed diff would add history noise without
enough future value.

The normal research loop is local-first. It commits iterations and updates a
local dashboard but does not push ordinary campaign history automatically.
Git plus the committed CSV/JSON records are the source of truth; the dashboard
is a regenerable view showing outcome, commit, phase timing, incremental and
cumulative improvement, and evidence links. Pushing is reserved for explicit
backup/publication or for a candidate that must reach the production PC for
target-hardware validation.

The dashboard is a fully local static `index.html` regenerated atomically after
each completed iteration. It embeds the current audit data and may refresh
itself in the browser, but requires no localhost server, listening port,
database, or long-running dashboard process.

The controller creates one lightweight audit commit for every completed
iteration, including rejected and inconclusive attempts. That commit contains
the index entry, readable and machine-readable summaries, and compact raw
numeric evidence such as timing and behavior CSV/JSON; an accepted exact
iteration also contains its production diff. Evidence above the campaign's
Git-artifact size cap and rendered images remain in a content-hashed immutable
artifact directory and are referenced from the audit record rather than copied
into Git history. The initial cap is 2 MiB total per iteration and 1 MiB for
any individual file. Each campaign also has a 10 GiB external-artifact budget.
If writing the next artifact would exceed that budget, the controller
checkpoints and stops for user action; it never deletes prior evidence
automatically.

External artifacts default to the canonical checkout's ignored
`outputs/runtime_campaigns/<campaign-id>/` directory rather than the disposable
optimization worktree. `campaign.json` pins that resolved absolute path so the
artifact trail survives worktree removal and remains easy to inspect.

Before a campaign, an agent drafts a concise `program.md` from the canonical
project context, failure review, performance audit, relevant MVP tuning
history, and runtime experiment index. The draft summarizes constraints,
known failed ideas, accepted wins, promising next hypotheses, and links to the
underlying records; it does not concatenate the source documents. The user
reviews this research strategy before the campaign begins. Machine-enforced
settings such as target, corpus, editable paths, commands, authentication, and
deadline policy live separately in `campaign.json` so prompt text is never the
safety boundary.
The reviewed `program.md` is immutable for one campaign and its content hash is
pinned in `campaign.json`. Strategy changes require a newly generated,
user-reviewed program and therefore start a new campaign lineage.

Iterations use a two-stage funnel. A frozen stratified preflight set spans
different tissue types, sparse and dense routes, known fragile cases,
threshold- and router-adjacent cases, and true and hard near-miss
relationships. Only candidates with a meaningful preliminary runtime gain and
no material canary regression advance to the full 41-pair live run, full
behavioral corpus, and repeated timing protocol. Preflight can reject a
candidate early but can never accept or retain one by itself.

A candidate advances from preflight when its canary end-to-end latency improves
by at least 5%, or when its explicitly targeted stage improves by at least 10%
without worsening canary end-to-end latency. These are provisional screening
thresholds, not evidence of a full-corpus win.

An exact candidate is committed automatically only when the paired median for
the complete 41-set processing session improves by at least 3%, the 95%
confidence interval's lower bound exceeds 1%, the operator-facing slide phase
does not regress beyond measured noise, the full-precision behavior and any
applicable masks are identical, and focused tests pass. Smaller exact gains are
retained for user review because their code cost may outweigh a barely
measurable improvement.

Every experiment reports incremental runtime change against its immediate
parent and cumulative runtime change against the campaign's frozen starting
commit, overall and by phase. This makes interactions between accepted
optimizations visible instead of adding isolated percentages on paper.

Focused tests and lint run before expensive evaluation. An exact candidate
that passes behavior and timing gates must then pass the full repository test
suite before its automatic commit. Starting a campaign authorizes these
full-suite runs inside its isolated worktree.

Approximate candidates never stack autonomously. Each starts from the latest
accepted exact commit, is evaluated and exported, and is then removed before
the next iteration. Combining or promoting approximate patches requires user
review and begins a new frozen campaign baseline, while retaining the prior
lineage for cumulative comparison.

The initial frozen v3 corpus contains the 41 manifest claims and the complete
41 by 41 diagnostic matrix. Its identity includes the manifest SHA-256, SHA-256
for all 82 specimen PNGs, ordered claimed and diagnostic relationships, label
source, and creation timestamp. Comparisons refuse mismatched corpus identities
unless the user deliberately creates a new baseline lineage.

The full matrix is used to mine a versioned hard-negative snapshot; appearance
does not determine true-versus-wrong identity. Routine iterations compare the
41 true claims and frozen hard negatives. The full matrix is an
optional escalation only for a significant approximate candidate or an
explicit user request, to detect a newly emerging best wrong identity. Live
timing always uses the 41 true claimed pairs only.

An approximate candidate triggers that background full-matrix escalation when
its complete-session speedup is at least 5% and the frozen reviewed set shows
no severe margin loss. Run the matrix from an isolated candidate snapshot.
While it runs, a worker may read history and research the next hypothesis, but
no other performance benchmark may execute concurrently because CPU contention
would invalidate timing evidence.

Hard-negative mining is tissue-type agnostic: every known wrong slide is
eligible for each block. The frozen review set retains the best-scoring wrong
slide for each block plus at most the next two wrong slides when each lies
within 0.04 score of that block's best wrong. On the initial v3 baseline this
selects 79 pairs. Human visual review may tag which hard negatives are also
visually plausible near misses; visual appearance never changes the
true-versus-wrong identity label.

Contact sheets are generated lazily for consequential approximate findings,
not for routine exact iterations and not for every pair in the frozen baseline.
Generate focused baseline-versus-candidate sheets only for a severe margin loss
over 0.05, a gate or router change, a best-wrong identity change accompanied by
a material margin loss over 0.02, a score-mapping residual outside +/-0.01, or
an explicit user request. Uniform soft drift without one of those findings
requires numeric evidence only. When a visual trigger exists, the soft
candidate is not ready for user judgment until that focused packet is present.

Each collection writes a self-contained run directory containing `run.json`
for provenance, `behavior.csv` for the full pair ledger, `timings.csv` for raw
stage samples, `summary.json` for computed distributions, and `report.md` for
human review. A comparison writes machine-readable `comparison.json`, readable
`comparison.md`, and a focused outlier table. Reports are derived artifacts;
the raw ledger and samples remain the evidence of record.

The measurement surface is one stable reusable harness CLI assembled from the
existing selected-pair timing, all-pairs diagnostic, production scoring, and
overlay tools. It exposes baseline, candidate, comparison, and campaign
operations; experiments do not create custom evaluator scripts.

The default iteration uses one AI worker. When its expected value justifies
the token cost, that worker may delegate narrow roles: a read-only researcher
proposes one stage-specific hypothesis, one implementer owns all allowlisted
production edits, and a read-only reviewer checks the resulting diff before
evaluation. The deterministic controller remains the judge, multiple agents
never edit the candidate concurrently, and every delegated role receives an
explicit token cap and only the context it needs.

The initial token-aware model policy uses Claude Sonnet 5 at medium effort for
routine research and implementation, and escalates to GPT-5.5 at medium effort
for unresolved ambiguity or approximate-candidate review. Do not invoke both
on every iteration. Record model, effort, input/output usage, and outcome so
future campaigns can choose models by accepted improvements per unit of usage.

Automatic workers receive the allowlisted source text in their prompt and run
with no model tools; they return a structured unified patch that only the
controller may validate and apply. The installed Codex CLI's read-only sandbox
does not remove its shell/read capability, so GPT-5.5 escalation is initially a
manual sanitized-packet workflow. Automatic Codex use remains disabled until a
real containment preflight proves it cannot read outside the supplied packet.

Worker invocation uses the user's existing Claude Code and Codex subscriptions,
not API-key billing. The reusable orchestration and manual fallback live in the
project-local `.agents/skills/improve-runtime/` skill, including a PowerShell
campaign launcher. The tested collector and comparator remain repository tools
outside the skill. If a provider reaches its subscription limit, the campaign
checkpoints and stops; it never falls through to paid API credentials.
Cross-provider failover is disabled by default and occurs only when
`campaign.json` explicitly names a subscription-backed fallback provider.

Subscription-only authentication is a fail-closed preflight invariant. Before
invoking a worker, the controller verifies subscription-backed login and
refuses to run when API-key billing is active or provider API-key environment
variables are present. It never opts into paid usage credits or an API fallback.
When the rolling subscription allowance is exhausted, it records a checkpoint
and stops until the allowance resets.

Campaign startup also requires a dated
`subscription_overage_disabled_confirmed` acknowledgment in `campaign.json`,
because the CLIs cannot reliably inspect account- or workspace-level paid
credit settings. If a provider reports a reliable reset timestamp and it falls
before `run_until`, the launcher may sleep locally without token use, repeat
the subscription-only preflight after reset, and resume. If the reset time is
missing or falls after the deadline, it exits at the checkpoint.

`run_until` is a soft wall-clock deadline enforced at atomic phase boundaries.
After it arrives, the controller starts no new worker call, test phase,
evaluation phase, or timing batch, but it lets the currently running phase
finish and checkpoints its evidence before exiting. A small overrun is
preferable to corrupting or discarding an in-flight result.

The PowerShell launcher accepts either `-RunUntil "06:00"`, meaning the next
occurrence in the laptop's local timezone, or a relative duration such as
`-RunForHours 8`. It also accepts a full timestamp for longer campaigns. The
forms are mutually exclusive and resolve to one absolute timestamp with
timezone and original input recorded in `campaign.json`.

Each iteration starts a fresh worker session and ends it after one
hypothesis-and-implementation cycle. The controller supplies the frozen
`program.md`, `campaign.json`, compact experiment memory, and only the relevant
detailed records. Research state persists in files and commits rather than an
ever-growing provider conversation.

The controller derives that compact memory from the canonical full audit log:
accepted wins, failures grouped by phase, recent attempts, and open leads. The
full log remains searchable on disk. Before implementing, the worker searches
it for the proposed functions and hypothesis terms so concise prompt context
does not turn into repeated experiments.
