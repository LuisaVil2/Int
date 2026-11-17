# Phase 3 validation report

The baseline is phase2/simulation-qa at bee87a0b8bcc36852de34f1389f64c3a1066a1a0.
Phase 1 audio defenses and Phase 2 simulation and review remain available.
All fixtures here use fictional data.

## Comparison on the same 68 scripted turns

| Measure | Phase 2 | Phase 3 |
|---|---:|---:|
| Accepted / held | 1 / 67 | 1 / 67 |
| EN→ES accepted / held | 1 / 37 | 1 / 37 |
| ES→EN accepted / held | 0 / 30 | 0 / 30 |
| Confidence-review holds | 44 | 42 |
| Specific laterality / negation-scope holds | 0 / 0 | 1 / 1 |
| Authored variants with any rule flag | 14 of 18 | 16 of 18 |
| Authored variants accepted | 0 of 18 | 0 of 18 |

The decrease in confidence-review holds is a change in reason. The two turns are
still held. In card-pressure / t1, the source and output refer to different arms.
In obs-bleeding / t2, the output retains a negation marker and attaches it to
the wrong symptom. Both turns now have explicit rule evidence in the simulator
and the shared interpreter gate. A flag on an authored variant can also represent
a missing candidate. The mutant counts describe the regression corpus.

The prior glossary looked up English terms, abbreviations and some drug names.
The bilingual prompt lookup now recognizes Spanish phrase and drug spellings
and folds accents for matching. On turns that reached confidence scoring,
legacy/bilingual match totals were 10/10 for EN→ES and 0/4 for ES→EN. An
offline shadow recomputation released zero additional turns. The score weights,
approval cutoff (95), review cutoff (80), and confidence inputs that govern
playback stayed the same. The glossary remains a contextual hint. Any future
score change needs a preregistered comparison with independent qualified
reviewers, examples of false holds and critical escapes, and enough evaluated
turns in both directions to quantify the uncertainty.

The comparisons retain per-scenario and per-direction counts and reasons,
and every fixture row's disposition and rule IDs. False holds, adjudicated
major/critical errors, critical escapes, agreement, live STT, and end-to-end
latency are unavailable for this run. Zero accepted authored variants is an
observed fixture count, with 18 authored variants in the denominator.

The recorded runs are available as the [fixture comparison](reports/phase3/fixture-comparison.json)
and [injected audio replay](reports/phase3/audio-replay.json). Replaying a fixed
candidate cannot measure how the bilingual prompt changes provider output.

## What the new rules check

- Scoped polarity and left/right associations use a bounded set of bilingual
  symptom and anatomy anchors. Coordinated and ambiguous clauses can still
  require human interpretation.
- The quantity guard recognizes a small vocabulary of spoken numbers and units.
  It accepts digit/word equivalents when the ordered quantities match and there
  are no remaining digits outside those quantities. Medication names, routes,
  frequency, day/month, duration, allergies and first-person substitutions have
  targeted cases. Distinct reported speaker roles are still recorded separately.
- A single bilingual reference is useful evidence. Semantically equivalent
  rephrasing and clinical consequences still require qualified human judgment.
  Imported model findings remain suggestions. They are stored separately from
  automated flags, human annotations and adjudications.
- Cross-clause attribution, compound negation, drug-to-dose linkage with multiple
  drugs, unrecognized aliases, laterality outside the small vocabulary, and
  general paraphrases remain incompletely covered. The legacy lexical polarity
  check can hold an equivalent phrase such as “felt unsafe” / “no se sentía a
  salvo.” Several valid short replies still enter review for uncertain language.
  These are evaluation targets for the next phase.

## Reproduce the comparison

From the candidate checkout, create a separate worktree for Phase 2 and use the
same Python and pinned requirements in both. Run these commands from the
respective repository roots. Output filenames must be new.

~~~powershell
git worktree add ../int-phase2 bee87a0b8bcc36852de34f1389f64c3a1066a1a0
New-Item -ItemType Directory -Force out/phase3 | Out-Null
$candidate = (Get-Location).Path
Push-Location ../int-phase2
python "$candidate/src/simulation/comparison.py" --output "$candidate/out/phase3/baseline.json" --measure-latency
Pop-Location
python -m src.simulation.validation fixtures --baseline out/phase3/baseline.json --measure-latency --output out/phase3/comparison.json
~~~

The driver imports the engine and corpus from the current worktree. Fixture
latency measures a local replay, including initialization effects. The candidate
also records measured translation and QA boundaries when available. Stage marks
use a monotonic clock and omit transcript content. The live engine can emit
capture-frame, STT-finalized, translation, QA, TTS and playback marks through an
optional trace callback. Live capture and STT marks have no per-turn correlation
ID, so they cannot support turn-level end-to-end timing in this phase.

| Local fixture stage | Samples | p50 ms | p95 ms |
|---|---:|---:|---:|
| Baseline processing | 68 | 4.349 | 22.800 |
| Candidate processing | 68 | 4.281 | 18.552 |
| Candidate fixture translation | 59 | 0.048 | 0.147 |
| Candidate QA after translation | 59 | 0.469 | 0.915 |

## Independent review and adjudication

A study coordinator assigns distinct pseudonyms after verifying both reviewers
are qualified bilingual medical interpreters. The qualified flag records the
coordinator's attestation. Keep the local SQLite file and adjudication commands
under coordinator control. The desktop view hides the other reviewer, prior
adjudications, machine evidence and the engine disposition. This is workflow
blinding for a supervised local study. The database is readable by someone
with filesystem access and pseudonyms are not authentication credentials.

~~~powershell
$db = 'out/phase3/study.sqlite3'
$run = python -m src.simulation --db $db run --scenario card-pressure | ConvertFrom-Json
$sid = $run[0].session
python -m src.simulation.validation --db $db assign --session $sid --slot a --reviewer interpreter-a --qualified
python -m src.simulation.validation --db $db assign --session $sid --slot b --reviewer interpreter-b --qualified
python -m src.simulation.validation --db $db blind-review --session $sid --slot a --reviewer interpreter-a
python -m src.simulation.validation --db $db blind-review --session $sid --slot b --reviewer interpreter-b
python -m src.simulation.validation --db $db adjudication-input --session $sid --turn t1 --output out/phase3/adjudication-input.json
python -m src.simulation.validation --db $db adjudicate --file out/phase3/decision.json
python -m src.simulation.validation --db $db panel-metrics --session $sid
~~~

Each reviewer saves a draft or marks the turn complete. An adjudicator receives
the pair only once both current reviews are complete and qualified. Create
decision.json using the session, turn and revision numbers from the input.
For example, its shape is:

~~~json
{"schema_version":1,"session_id":"ID_FROM_THE_RUN","turn_id":"t1","revision_a":1,"revision_b":1,"adjudicator":"coordinator-c","errors":[{"category":"mistranslation","severity":"major"}],"hold_justified":true,"corrected_interpretation":"Fictional correction","notes":"Why this decision was made"}
~~~

Both review histories and adjudication history are append-only. If either
reviewer revises a turn, the previous adjudication remains recorded and stops
contributing to current metrics. Reports show the number of paired qualified
human reviews, exact label/severity agreement, binary agreement and Cohen's
kappa for any-error judgments. Kappa is null when undefined. Demo annotations,
incomplete reviews, legacy solo reviews and unqualified slots are excluded.
A false-hold count requires an adjudicator to explicitly judge justification
on a held turn. Critical escapes use adjudicated accepted turns as the
denominator. Small or unrepresentative samples need additional review before
drawing conclusions. The automated tests used temporary synthetic records to
exercise these calculations. No qualified humans reviewed this corpus here.

## Fictional audio benchmark

The no-key default creates a mono PCM WAV from tones and replays transcripts.
Its purpose is to check scoring, scheduling, labels and fault injection. Use
the same fixture with the locally installed eSpeak NG executable to generate
speech WAVs in two different voices. Supply a local faster-whisper model
directory for offline recognition, or explicitly opt in to Deepgram batch
recognition. Example commands:

~~~powershell
python -m src.simulation.validation audio --scenario pc-follow-up --output out/phase3/mock-audio.json --pause-ms 40 --noise-dbfs -40 --fault drop-duplicate
python -m src.simulation.validation audio --scenario pc-follow-up --renderer espeak --voice-en en-us --voice-es es --rate-en 185 --rate-es 145 --pause-ms 120 --output out/phase3/whisper-audio.json --stt whisper --whisper-model C:/models/whisper-small
python -m src.simulation.validation audio --scenario resp-bilingual --renderer espeak --output out/phase3/deepgram-audio.json --stt deepgram --allow-network
~~~

Use the provider's valid voice IDs. The eSpeak adapter passes a voice, rate,
and UTF-8 text on stdin and writes WAV rather than broadcasting it. Install
eSpeak NG separately or pass --espeak-executable. [eSpeak NG command reference](https://github.com/espeak-ng/espeak-ng/blob/master/src/espeak-ng.1.ronn)

To use an authorized fictional actor recording, pass --recording WAV
--timeline JSON --authorized-fictional. Use a mono PCM16 WAV and a version-1
timeline containing scenario and turns. Each turn has turn_id, start and end
in seconds, with the IDs matching the corpus in order. The harness writes an
example timeline alongside generated WAVs. Keep these files separate from
patient, workplace and client data. Deepgram receives the supplied audio only
with --allow-network and its key configured privately in the environment or
ignored .env file. Local faster-whisper uses files already on disk. Optional
--play asks Windows to play the benchmark WAV, measuring the blocking call
rather than acoustic arrival. Avoid microphone or loopback capture that could
include other activity.

The harness scores STT before translation. Included metrics cover word and
character edit distance with explicit normalization, missed turns, repeated
timestamp/text segments, split turns, boundary overlaps, language mismatches,
within-recording speaker clustering, and separate clinical-anchor errors.
Reference spans use scripted timing or an annotated actor timeline. The per-turn
assignment uses greatest overlap, so alignment can be wrong at crowded
boundaries. Missing speaker or language metadata remains unavailable or unknown.
The timing report separates WAV read, source synthesis, batch adapter and
scoring, with sample counts and nearest-rank p50/p95. For a one-file batch these
are single measurements, not an estimated latency distribution.

The clean transcript replay counted zero edits. The injected run counted one
missing turn and one repeated segment, with 11/26 word edits and 35/103 character
edits overall. The fixture detected one missing clinical anchor in ES→EN.
These are injected faults in replay, not recognizer results or physical-device
evidence. Existing and new mock-device tests cover suppressed capture during
playback, the interruption and reconnect paths, queue ordering, and stop/new
session behavior.

## Verification and Phase 4 readiness

The baseline passed 148 core tests and one Tk smoke test on Windows under
Python 3.12.14 and the pinned requirements. The earlier scope/laterality test
passed with its expected missing flags before implementation.

| Command on the candidate | Outcome |
|---|---|
| python -m pytest -q | 176 passed in 21.38 s. |
| python -m pytest -q tests/ui_smoke.py | 2 passed in 2.52 s. |
| python -m compileall -q src tests | Passed, exit 0. |
| python -m src.simulation.validation fixtures --baseline ../phase3-baseline.json --measure-latency --output out/phase3/comparison.json | Both cohorts completed, 68 turns each. |
| python -m src.simulation.validation audio --scenario pc-follow-up --output out/phase3/mock-audio.json --pause-ms 40 --noise-dbfs -40 --fault drop-duplicate | Replay completed with the injected loss and duplication. |

No physical device, live provider, spoken-audio recognition run, or qualified-human
adjudication was verified here. This workspace had no eSpeak NG executable on
PATH and no staged Whisper weights or dedicated hardware test rig. The eSpeak
subprocess wiring and both audio evidence and independent review workflows are
covered by fixtures/mocks or the opt-in Tk smoke.

**Ready for a controlled fictional study.** For Phase 4, recruit and verify
independent bilingual reviewers, adjudicate a balanced corpus, then run speech
fixtures through actual STT and dedicated audio devices. Only consider adjusting
confidence after quantifying false holds and critical escapes in both directions.
Add capture-to-utterance correlation for end-to-end timing, and use coordinator
access controls or separate reviewer packets for multi-user studies.
