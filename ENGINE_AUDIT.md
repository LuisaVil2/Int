# Interpretation engine stabilization audit

Scope is the existing Python prototype on Windows, with fictional EN/ES conversations only.
The baseline is main at **dd019e49e8a2dfaa88d7c676e6d76e3a25a5b610**. Severity
describes the consequence in a simulation if the defect occurs. It is independent
of the confidence score printed by the engine.

## Prioritized findings

| File | Defect | Severity | Evidence | Fix on this branch |
|---|---|---|---|---|
| src/live_engine.py, src/translate.py | A dose from the streamed JSON prefix can enter TTS before the response ends, even if the final result is rejected. | Critical | Reproduced with test_invalid_stream_prefix_never_reaches_tts. A mocked first output says 15 mg for a 50 mg source. The old queue contained the rejected phrase. | Request the whole response, verify the provider finish reason, validate required JSON fields and direction, and gate a single complete turn before TTS. |
| src/translate.py, src/prompts.py | A transport error runs dictionary substitutions. Shape and target checks alone accept altered quantities and some omissions. | Critical | Source inspection of the old exception handler and local dictionary path. Fictional mutation fixtures exercise 2.5→25, mg→g, omitted quantities and missing negation. | Return an explicit clarification state on transport failure. Retry one invalid result, reject changed digit sequences, common dose units and some polarity changes, and preserve source voice and meaning in the prompt. |
| src/stt.py | Results are appended without honoring is_final or a repeated audio range. UtteranceEnd is ignored. | High | Reproduced with test_stream_only_delivers_unique_final_ranges. The old receiver emitted duplicates and an interim prefix. An interim event was injected into the parser even though the old request disabled interims. | Accumulate only final ranges, deduplicate by time within each connection, use matched utterance boundaries, and hold known incomplete or overlapping ranges. Two identical replies at different timestamps still survive. |
| src/segmentation.py, src/live_engine.py, src/live_interpreter.py | An ambiguous response defaults to English or inherits a previous speaker's language. The GUI did not use the batch segmentation path. | High | Reproduced with test_unknown_direction_holds_turn. The old turn sent “No.” to the translator as English. | Share the live engine with the CLI. Use same-span ASR evidence and conservative short-response rules. Separate clear sentence-level EN/ES switches and hold unresolved spans. |
| src/stt.py, src/live_engine.py, src/memory.py | A socket failure discards audio silently. A REST speaker label can be fed back as an identity across independent calls. | High | Inspection of sender/receiver reconnect and the old memory call. Fake sockets exercise a lost connection after a finalized prefix and a trailing fragment on the replacement connection. | A single receiver owns reconnection. Report a gap and hold the first post-reconnect endpoint. Local Deepgram diarization can flag multiple speakers in one endpoint. Do not carry those IDs into context across turns. |
| src/live_engine.py, src/emergency.py, src/confidence.py | Review tiers did not stop playback. Emergency labels could be appended and spoken inside the interpretation, including for negated symptoms. | High | Inspection of queue calls and keyword rules. The old instrumentation test explicitly expected alert text inside the displayed interpretation. | Hold both review tiers, report keyword matches as review metadata, and keep speaker words separate. Reject nonfinite confidence inputs for routing. |
| src/live_engine.py, src/gui.py, src/tts/ | Queue emptiness can keep input muted while synthesis is in flight. Frames can straddle playback start/end, and stopping can race with pending play. | High | Inspection of queue and device calls. Fake recorder and player cover overlap, silence substitution and FIFO across a synthesis gap. No hardware trace was available. | Bound queues, check stop before playing, mute only for playback and a 150 ms tail, hold affected buffered STT where observable, and signal empty or failed synthesis. Show listening and repeat notices. |
| src/logging_utils.py, src/stt.py, src/translate.py, .env.example, requirements.txt | Raw exception text or a transcript-debug flag could write provider content. The sample configuration contains unused variables and a machine-specific ffmpeg path. | High | Source inspection. Test checks that a fictional private payload and transcript-debug flag do not transfer content to logs. | Log fixed event names and bounded metadata. Scope logging to application modules, remove the transcript dump flag, align model settings, and remove a duplicate pytest pin. |

Deepgram's documented contract distinguishes [endpointing and final results](https://developers.deepgram.com/docs/understand-endpointing-interim-results), [UtteranceEnd](https://developers.deepgram.com/docs/understanding-end-of-speech-detection), and [word and alternative language tags](https://developers.deepgram.com/docs/nova-3-language-prompting). The tests inject those message shapes. They do not certify the live service or microphone behavior.

The default translation model ID is now `deepseek-flash`. Explicit model IDs still come from `DEEPSEEK_MODEL`. DeepSeek currently documents the old `deepseek-v4-flash` ID as a compatibility alias. [DeepSeek API setup](https://api-docs.deepseek.com/guides/agent_integrations/claude_code)

## Implemented behavior

- Deepgram WebSocket streams mono PCM16 at 16 kHz. Only final transcription ranges enter a turn. Requested silence endpointing is 400 ms, with a 1,000 ms UtteranceEnd fallback tied to the last finalized word. Exact range repeats have no effect on later turns. Overlap, a pending interim tail, multiple speaker labels inside one endpoint, or known interruptions can hold speech. The receiver reconnects with 0.5, 1 and 2 second backoff. Audio lost during the gap is not replayed. Reconnecting holds the first new endpoint conservatively.
- With no Deepgram key at startup, local VAD and Whisper handle speech. Initial Deepgram connection errors are surfaced. The independent REST-to-Whisper path is for per-utterance callers and tests, separate from WebSocket recovery.
- English and Spanish evidence comes from transcript structure, short-response rules, text identification and eligible same-span ASR tags. There is no language history assumption. Distinct sentences can resolve separate directions. Unresolved mixed speech and language-neutral short replies require review.
- DeepSeek output must finish normally and pass schema, opposite direction, language and limited fidelity checks. The model is asked to preserve negations and their scope, numeric details, symptoms, first-person perspective, register, and speaker references. Malformed responses can be retried once. An unavailable provider produces a review state. The experimental dictionary helper has no runtime fallback role.
- Both QA routes are silent. Where there is a structured draft, the UI labels it as a draft. Uncertain translations, missing direction and capture gaps have explicit review reasons. The session memory takes accepted turns only and stores no live diarization IDs. In text-only mode the same routing still applies.
- TTS uses a single FIFO synthesis worker and a single FIFO playback worker. Fish Audio remains available when configured; Edge is the existing audio-rendering fallback for the same text. A stop suppresses queued work and callbacks; an in-flight call or recorder operation can outlive the bounded worker join. Application logs exclude transcript and exception payloads.

## Test report

From the repository root with Python 3.12 on Windows:

    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -p test_engine_regressions.py
    python -m pytest -q
    python -m compileall -q src tests

The host used bundled Python 3.12.14. The command labeled baseline below used the checked-in regression file with the baseline source before any behavior change. No runtime provider keys, live audio, or recordings were needed.

| Check | Outcome |
|---|---|
| Baseline: python -m pytest -q | 86 passed in 17.14 s with the pinned environment. |
| Baseline: python -m unittest discover -s tests -p test_engine_regressions.py | 3 ran, 3 failed. The failures were early TTS queueing, interim and duplicate range accumulation, and forced English direction. |
| Stabilized: python -m unittest discover -s tests -p test_engine_regressions.py | 3 passed. |
| Stabilized: python -m pytest -q | 121 passed in 12.43 s with the pinned environment. |
| Stabilized: python -m compileall -q src tests | Passed, exit 0. |
| Bundled Silero VAD on generated silence | Passed. Pinned runtime imports succeeded and inference returned finite values. |
| Live Deepgram, DeepSeek and cloud TTS integration, including manual root smoke scripts | Blocked for this audit. These require configured test accounts and a separate live simulation run. The configured pytest suite does not execute legacy scripts that call providers at import. |
| Physical WASAPI/microphone, acoustic echo, device-specific stop timing | Blocked for this audit. No controlled physical audio session was run. Fake-device tests exercise the engine paths only. |

The full requirements install passed with pytest 9.1.1, NumPy 2.4.6, Pydantic 2.13.4, Anthropic 0.115.0 and websockets 16.0. The workspace ran the bundled Python executable with the install target on `PYTHONPATH`. The Whisper-specific unit tests patch the package's lazy-import boundary. The separate Silero check used zeros without loading a Whisper transcription model. The legacy root fallback demos describe the retired dictionary behavior and are superseded by the deterministic suite.

    python -c "import anthropic, numpy, pydantic, pytest, soundcard, sounddevice, websockets; from faster_whisper.vad import get_vad_model; out=get_vad_model()(numpy.zeros(1536, dtype=numpy.float32)); assert out.size and numpy.isfinite(out).all()"

The fixture in [tests/fixtures/fictional_dialogues.json](tests/fixtures/fictional_dialogues.json)
has 12 fictional cases, with reference translations where applicable and specific
mutations for deterministic rejection. Tests use stub responses. They do not score
any translation generated by a provider.

## Unresolved issues and Phase 2

1. **Measure STT separately.** Build scripted recordings with actors or synthetic voices, balanced across accents, directions and medical vocabulary. Vary pause length, background noise, turn overlap and device topology. Capture sanitized provider event structure. Compare reference turns with final output and count missing, duplicate, clipped and wrongly split turns. Exercise network drop and reconnect, stale endpoints, clean close and long pauses. Timestamp ranges cannot prove that unseen audio arrived.
2. **Review meaning independently of STT.** Give two qualified bilingual medical interpreters anonymized randomized source/reference/candidate triples. Score omissions, unsupported additions, polarity and scope, numbers in words, units, route, schedule, drug and allergy names, symptoms, speaker attribution, register and first person. Adjudicate disagreements. Check the audio rendering separately for dosage and unit pronunciation. The current checks cannot establish semantic or clinical accuracy.
3. **Evaluate the review path.** Measure how often eligible turns are held, critical errors escape review, and the operator can recover with repetition. Common review thresholds are heuristic and depend heavily on glossary matches. Shared short replies, unsupported languages, rapid switching with weak punctuation and overlapping speakers may still be held or misclassified. Consider operator-controlled speaker turns or separate channels before trying to infer identity.
4. **Exercise shutdown and half-duplex with devices.** Use scripted overlap while the engine speaks, repeated stop/start, input/output disconnects and same-device loopback. Count lost and contaminated turns and measure playback-start and resume timing. Half-duplex currently discards overlapping speech, and the 150 ms tail is uncalibrated. Evaluate action by the operator before considering barge-in or echo cancellation.
5. **Pin a repeatable simulation setup.** Record exact dependency and provider versions, endpoint model and settings, anonymized event counts, corpus size and reproducible commands. Check provider data handling and the local handling of displayed and exported text. Use measured latency percentiles, error rates, review coverage and adjudicated examples to define an acceptance decision before broader testing.
