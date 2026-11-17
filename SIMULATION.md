# Fictional conversation simulation and review

This workflow exercises the existing EN↔ES interpreter with simulated clinician and
patient turns. It keeps source, candidate, and reference visible for human review.
The default fixture adapter replays authored responses, including deliberate
mistakes. Its results measure the workflow and rule coverage.

## Run locally

Use the repository root and the Python environment from [README.md](README.md).
These commands run without API keys, a microphone, a speaker, or an external service.

~~~powershell
python -m src.simulation list
python -m src.simulation list --specialty emergency_medicine --difficulty advanced --min-turns 4
python -m src.simulation run --scenario med-dose
python -m src.simulation run --specialty pediatrics --difficulty intermediate
python -m src.simulation run
python -m src.simulation metrics
python -m src.simulation review
python -m src.simulation export --output out/simulation/evaluation.json
~~~

The unfiltered run executes all 20 scenarios. Filter by specialty, difficulty, and
turn count without truncating dialogue. The corpus has 68 turns across primary care,
emergency medicine, cardiology, respiratory medicine, obstetrics, pediatrics,
medication reconciliation, allergies, informed consent, and mental health. It
includes explicit dates, durations, doses, units, allergies, pain intensity,
similar clinical terms, corrections, clarification, interruption, short replies,
negation scope, and rapid language changes. Each turn references a scripted
clinician or patient ID. The simulated ASR language hint is separate from that
identity.

The default database is out/simulation/reviews.sqlite3. Put --db PATH before the
subcommand to open a separate database. Each run generates a new session ID.
Scenario and review schemas are versioned in src/simulation/schemas.py. Choose
a fresh filename for an export.

## Review a session

1. In the desktop reviewer, select a session and turn. Read the source and engine
   candidate side by side and check the reference underneath. Review the role,
   direction, and hold reason before assigning labels.
2. Select a severity next to each applicable human error. Leave the selectors
   blank if you find no error. Enter a corrected interpretation and notes as
   necessary.
3. Save a draft or mark the turn reviewed. Reloading preserves saved values.
   Reviews are stored separately, with revision checks for concurrent edits.
   The original engine response remains available alongside the correction.
4. Export the selected session from the UI. This produces a structural JSON
   export. The command line can optionally include vetted fictional source,
   candidate, and reference text with --include-fictional-text.

The local database stores reviewer text for continued review. Exports keep
turn IDs, roles, direction, flags, model-suggestion labels, review status,
human error labels, and metrics. They omit corrections, notes, model rationales,
timestamps identifying the run, and local audio paths. Session IDs are random.

### Severity and evidence

| Severity | Human decision |
|---|---|
| Minor | Wording or register shifted without materially changing meaning or the action a listener would understand. |
| Major | A material part of the meaning, understanding, sequence, or attribution changed. |
| Critical | The change could reverse a consequential decision or misstate a time-sensitive symptom, drug, allergy, dose, instruction, or consent. |

Use the source and intended speech act to assign severity. For example, an
omitted greeting and an omitted allergy statement both fit omission, with very
different severity. The UI lets the reviewer choose independently of rule hints.
The default suggestions classify a polarity marker mismatch, medication/dose
difference, or reported wrong direction as critical and other flags as major.
A custom policy overrides the suggested mapping per category.

~~~json
{"schema_version":1,"minor":["addition"],"critical":["negation_reversal","medication_or_dose","wrong_direction"]}
~~~

Pass that JSON file with run --policy PATH. Available categories cover omission,
addition, mistranslation, negation reversal, medication or dose, numbers or units,
speaker attribution, wrong direction, unjustified inference, and uncertainty or failure.
The deterministic rules examine explicit digits, common dose units, author-specified
drug and measurement tokens, polarity marker presence, direction and role metadata,
missing output, and whether a fixture expected a review. The rules label differences
for inspection. Omissions, additions, semantic mistranslation, polarity scope,
first-person meaning, and unsupported inference require human judgment.

Optional model suggestions can be imported with run --suggestions PATH for one
matching scenario. An example versioned input follows.

~~~json
{"schema_version":1,"scenario_id":"med-dose","suggestions":[
  {"turn_id":"t2","category":"medication_or_dose","suggested_severity":"critical","rationale":"The quantities differ in the fictional source and candidate."}
]}
~~~

Rule flags, imported model suggestions, and completed human labels are counted
separately. The automated flag rate divides flagged turns by all turns. The
confirmed error rate divides completed human-reviewed turns with any error by
completed human-reviewed turns. With no reviewed turns the confirmed error rate
is null. Multiple labels on one turn each contribute to the category counts.
Demo reviews have a separate count and are excluded from human metrics.
Optional --measure-latency measures wall time inside transcript processing,
including translation, before QA persistence or audio rendering. It reports
nearest-rank p50 and p95 by adapter. Fixture timing describes local replay.

## Optional providers and audio

- run --audio mock creates short local tones as WAV tracks to exercise the
  review buttons. A source track is available for every turn. An output track
  is available when the engine accepted the candidate and QA raised no exact
  flags. Windows playback uses winsound. Headless evaluation uses no desktop.
- run --audio edge --allow-network uses the existing standard-voice Edge
  provider. You can supply --voice-en and --voice-es to select available voice
  names and accents. ffmpeg is required for conversion. Edge receives only the
  fictional text chosen for audio. Cloud audio errors are counted separately.
- run --adapter deepseek --allow-network exercises the existing paid translator.
  DeepSeek receives the current fictional utterance, relevant terminology, and
  the existing engine's accepted dialogue context.
- generate requires explicit network permission and uses the configured
  DeepSeek endpoint to create a JSON draft from specialty and difficulty.
  Inspect the generated content and bilingual references for fidelity and
  fictional-only data before loading it. The offline adapter needs manually
  authored mock replies to replay candidates for an imported draft.

~~~powershell
python -m src.simulation run --scenario pc-follow-up --audio mock
python -m src.simulation generate --specialty cardiology --difficulty advanced --allow-network --output out/simulation/cardiology-draft.json
python -m src.simulation run --scenario-file out/simulation/cardiology-draft.json --adapter deepseek --allow-network
~~~

Use only fictional content, including review text and locally attached WAVs.
The schemas validate structure and role consistency. A person must verify the
fictional provenance and clinical meaning of an authored or generated draft.

## Reproduce the complete medication session

This demonstration saves four turns, marks four simulated reviews, corrects a
dose in the review record, then exports the session with fictional text.

~~~powershell
$db = 'out/simulation/demo.sqlite3'
$run = python -m src.simulation --db $db run --scenario med-dose --measure-latency | ConvertFrom-Json
$id = $run[0].session
foreach ($turn in 't1', 't3', 't4') {
    python -m src.simulation --db $db record-review --session $id --turn $turn --demo
}
python -m src.simulation --db $db record-review --session $id --turn t2 --demo --error medication_or_dose:critical --correction 'I take 500 mg of metformin with dinner.' --notes 'The fictional candidate changed 500 mg to 50 mg.'
python -m src.simulation --db $db metrics --session $id
python -m src.simulation --db $db export --session $id --include-fictional-text --output out/simulation/demo-evaluation.json
python -m src.simulation --db $db review
~~~

| Turn | Role and direction | Observed fixture result and demonstration review |
|---|---|---|
| t1 | Clinician, EN → ES | Asked about metformin and the dose on the label. The candidate kept both details. The engine held the turn for confidence review. |
| t2 | Patient, ES → EN | Source 500 mg of metformin with dinner. The candidate said 50 mg. The engine held it for the number change. The evaluator recorded digit and dose discrepancies. The simulated review selected a critical medication/dose error and saved the 500 mg correction separately. |
| t3 | Patient, ES → EN | The patient clarified that the label says 500 mg and excludes 50 mg. The candidate retained both quantities and the negation. The engine held it for confidence review. |
| t4 | Clinician, EN → ES | The clinician asked to see the bottle. The engine held the candidate for confidence review. |

The demo has four completed simulated reviews, two automated flags on one
turn, zero human-confirmed errors, and no human error rate. The zero human
count reflects the absence of real reviewer annotations.

## Verification and open issues

Phase 2 starts from main at 98f02fd0fb8c6df1f0000e877d5dc578de717c9b,
which contains the Phase 1 source tree. The Phase 1 baseline reran with
121 passes on Windows under Python 3.12.14 and the pinned requirements.

| Command | Outcome |
|---|---|
| python -m pytest -q | 148 passed in 24.05 s with pinned dependencies. |
| python -m pytest -q tests/ui_smoke.py | 1 passed in 2.85 s. |
| python -m compileall -q src tests | Passed, exit 0. |
| python -m src.simulation run | 20 sessions and 68 turns. The engine accepted 1 and held 67. QA produced 24 flags across 22 turns. |

The Tk smoke exercised real widgets, saved a draft and completed review,
preserved the original candidate, and exported structured data. The sound
backend was mocked while the test used genuine WAV fixture files.

- Confidence review held 44 fixture turns. Phase 1 scoring depends
  on glossary matches, and the current glossary indexes English phrase keys.
  The simulator preserves this behavior so that reviewers can measure it.
- The text “I understand.” is held as unknown direction in the bundled corpus
  despite its EN fixture hint. Several short or mixed turns are also held.
  A text interruption exercises the hold path and supplies no evidence about
  overlapping audio, endpoint timing, or actual diarization.
- The rules miss a laterality swap in card-pressure and a negation-scope swap
  in obs-bleeding. Human review is essential. The fixtures do not measure an
  LLM's interpretation accuracy or the reviewer agreement rate.
- The lexical polarity guard can hold valid phrasing. The source “felt unsafe”
  in mh-correction uses no explicit negation marker, while the authored Spanish
  equivalent uses “no se sentía a salvo.”
- Live DeepSeek replay and generation, cloud voices, acoustic output, actual
  accent handling, STT, and end-to-end latency remain unverified. The suite
  uses mock adapters and fictional text. Tk playback depends on a Windows
  device, and the UI smoke intercepts that final device call.

## Phase 3

1. Have two qualified bilingual medical interpreters independently review a
   balanced, blinded corpus. Adjudicate disagreements and refine references,
   especially for polarity scope, laterality, first person, and medication names.
2. Replay actor or synthetic audio for both directions through the STT and
   device paths. Measure dropped and duplicated turns, false holds, unsafe
   passes, echo, onset, and end-to-end latency by configuration.
3. Evaluate the English-key glossary bias and confidence thresholds against
   reviewed evidence. Report disagreement and completion rates, saved error
   examples, and performance by specialty and direction before considering
   broader use.
