import os
import asyncio
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
import numpy as np
import soundfile as sf
import io
import json
import re
import base64
import threading
import tempfile
import wave
import subprocess
import httpx
import librosa
from scipy.signal import butter, filtfilt
import azure.cognitiveservices.speech as speechsdk

AZURE_SPEECH_KEY = os.environ.get("AZURE_SPEECH_KEY", "")
AZURE_SPEECH_REGION = os.environ.get("AZURE_SPEECH_REGION", "eastus")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

# Azure SAPI phoneme -> IPA, and difficulty weighting for Vietnamese speakers —
# reused from the ted-pronunciation Space's app.py, where this weighting is already
# proven for this student population (vowels and the consonants Vietnamese speakers
# most often struggle with count for more than easy consonants).
SAPI_TO_IPA = {
    'ae':'æ','ey':'eɪ','ah':'ə','ao':'ɔː','aw':'aʊ',
    'ay':'aɪ','b':'b','ch':'tʃ','d':'d','dh':'ð',
    'eh':'e','er':'ɜːr','f':'f','g':'g','hh':'h',
    'ih':'ɪ','iy':'iː','jh':'dʒ','k':'k','l':'l',
    'm':'m','n':'n','ng':'ŋ','ow':'əʊ','oy':'ɔɪ',
    'p':'p','r':'r','s':'s','sh':'ʃ','t':'t',
    'th':'θ','uh':'ʊ','uw':'uː','v':'v','w':'w',
    'y':'j','z':'z','zh':'ʒ','aa':'ɑː'
}
VOWEL_PHONEMES = {'æ','ɪ','ʊ','ə','ɑː','ɔː','eɪ','aɪ','aʊ','ɜːr','iː','uː','əʊ','ɔɪ','e'}
HARD_CONSONANTS = {'θ','ð','v','z','ʒ','dʒ'}

app = FastAPI()

# ── AUDIO HELPERS ─────────────────────────────────────────
# preprocess_audio / save_wav_temp adapted from the ted-pronunciation Space's app.py

def preprocess_audio(audio_data, sample_rate=16000):
    nyq = sample_rate / 2
    cutoff = 80 / nyq
    b, a = butter(4, cutoff, btype='high')
    audio_data = filtfilt(b, a, audio_data)
    threshold = np.max(np.abs(audio_data)) * 0.02
    audio_data = np.where(np.abs(audio_data) < threshold, 0.0, audio_data)
    peak = np.max(np.abs(audio_data))
    if peak > 0:
        audio_data = audio_data / peak * 0.95
    return audio_data

def save_wav_temp(audio_data, sample_rate=16000):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, 'w') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sample_rate)
        pcm = (audio_data * 32767).astype(np.int16)
        wf.writeframes(pcm.tobytes())
    return tmp.name

def transcode_to_wav(input_bytes, sample_rate=16000, timeout=60):
    """Use ffmpeg (already in the Docker image) to convert any input container —
    mp4, m4a, mov, webm, wav, mp3, whatever — into 16kHz mono PCM WAV bytes.
    soundfile/libsndfile alone can't read mp4/m4a, which is what phone voice
    memos and video recordings are usually saved as. Returns None on failure."""
    in_tmp = tempfile.NamedTemporaryFile(suffix=".input", delete=False)
    in_tmp.write(input_bytes)
    in_tmp.close()
    out_path = in_tmp.name + ".wav"
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", in_tmp.name, "-ar", str(sample_rate), "-ac", "1", "-f", "wav", out_path],
            capture_output=True, timeout=timeout
        )
        if result.returncode != 0 or not os.path.exists(out_path):
            print(f"ffmpeg transcode failed: {result.stderr.decode(errors='ignore')[-500:]}")
            return None
        with open(out_path, "rb") as f:
            return f.read()
    except Exception as e:
        print(f"ffmpeg transcode error: {e}")
        return None
    finally:
        for p in (in_tmp.name, out_path):
            try:
                os.unlink(p)
            except:
                pass

def detect_pauses(audio_data, sample_rate=16000, min_pause=0.3, frame_length=512, hop_length=256):
    """Find internal silence gaps directly from the waveform's energy envelope.
    Leading/trailing silence is excluded — only pauses between speech count."""
    energy = librosa.feature.rms(y=audio_data, frame_length=frame_length, hop_length=hop_length)[0]
    peak = np.max(energy)
    if peak <= 0:
        return []
    threshold = peak * 0.08
    is_silent = energy < threshold
    frame_time = hop_length / sample_rate

    voiced = np.where(~is_silent)[0]
    if len(voiced) == 0:
        return []
    first_voiced, last_voiced = voiced[0], voiced[-1]

    pauses = []
    start = None
    for i in range(first_voiced, last_voiced + 1):
        if is_silent[i] and start is None:
            start = i
        elif not is_silent[i] and start is not None:
            duration = (i - start) * frame_time
            if duration >= min_pause:
                pauses.append({
                    "start": round(start * frame_time, 2),
                    "end": round(i * frame_time, 2),
                    "duration": round(duration, 2)
                })
            start = None
    return pauses

# ── GEMINI HELPERS ────────────────────────────────────────

async def call_gemini(contents, generation_config, retries=2):
    if not GEMINI_API_KEY:
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    # gemini-2.5-flash's internal "thinking" tokens count against maxOutputTokens by
    # default, which can silently eat the whole budget and truncate the real answer.
    # Neither transcription nor rubric-based scoring benefits from that reasoning, so
    # disable it unless a caller explicitly asks for a different thinking budget.
    full_config = {"thinkingConfig": {"thinkingBudget": 0}, **generation_config}
    body = {"contents": contents, "generationConfig": full_config}
    async with httpx.AsyncClient(timeout=90.0) as client:
        for attempt in range(retries + 1):
            try:
                r = await client.post(url, json=body)
                data = r.json()
            except Exception as e:
                print(f"Gemini call error: {e}")
                return None
            if r.status_code == 503 and attempt < retries:
                continue
            if "error" in data:
                print(f"Gemini API error: {data['error']}")
                return None
            try:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                print(f"Gemini response missing expected text field: {data}")
                return None
    return None

def parse_json_loose(text):
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception as e:
        print(f"JSON parse failed ({e}). Raw text: {text[:1000]!r}")
        return None

# Verbatim transcription prompt — reused as-is from the talk-anhnguted-secure app's
# transcribeAudio() (talk-anhnguted-secure-5/index.html), which is already proven in
# production for capturing fillers/false starts/pauses accurately.
def build_transcribe_prompt(pause_count):
    """Grounds Gemini's pause-marking with the exact count already measured, precisely
    and deterministically, from the raw waveform (detect_pauses(), called before this).
    Previously the prompt just said "mark noticeable pauses" and left it to Gemini's own
    subjective judgement of the audio — which meant a real, precisely-timed pause could
    be silently dropped from the transcript entirely if Gemini didn't think it was
    "noticeable enough" to mark. Telling it the exact expected count makes that failure
    mode far less likely, while durations still come from merge_pause_durations()
    filling in the real measured value for each marker in sequence, unchanged."""
    base = (
        "Transcribe this audio VERBATIM. Include all filler words (er, um, uh, like, you "
        "know), false starts, repetitions, and self-corrections exactly as spoken. Do not "
        "clean up or improve the speech. Return ONLY the raw transcript text, nothing else."
    )
    if pause_count > 0:
        base += (
            f" This recording contains exactly {pause_count} pause(s) of 0.3 seconds or "
            f"longer between words or phrases, precisely measured from the audio waveform. "
            f"You MUST mark the position of every single one with [...] in your transcript, "
            f"in chronological order — exactly {pause_count}, no more and no fewer, even if "
            f"a pause feels brief or unremarkable. Do not use your own judgement about "
            f"whether a pause is worth marking."
        )
    else:
        base += " There are no pauses of 0.3 seconds or longer in this recording — do not insert any [...] markers."
    return base

async def transcribe_verbatim_gemini(audio_bytes, pause_count, mime_type="audio/wav"):
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    prompt = build_transcribe_prompt(pause_count)
    contents = [{
        "role": "user",
        "parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": mime_type, "data": b64}}
        ]
    }]
    text = await call_gemini(contents, {"maxOutputTokens": 1500, "temperature": 0})
    return text.strip() if text else None

QUESTION_EXTRACTION_PROMPT = (
    "Extract the exact IELTS speaking question, topic, or cue card text shown in this "
    "file. Return ONLY the question text itself, nothing else — no markdown, no "
    "commentary, no quotation marks around it. If it's a cue card with bullet points, "
    "include all of them as they appear."
)

async def extract_question_from_file(file_bytes, mime_type):
    b64 = base64.b64encode(file_bytes).decode("ascii")
    contents = [{
        "role": "user",
        "parts": [
            {"text": QUESTION_EXTRACTION_PROMPT},
            {"inline_data": {"mime_type": mime_type, "data": b64}}
        ]
    }]
    text = await call_gemini(contents, {"maxOutputTokens": 1000, "temperature": 0})
    return text.strip() if text else None

def merge_pause_durations(transcript, pauses):
    """Replace Gemini's [...] pause markers with precisely-measured durations from
    the waveform, in sequence order. Extra/missing markers fall back gracefully."""
    if not transcript:
        return transcript
    idx = [0]
    def repl(_match):
        i = idx[0]
        idx[0] += 1
        if i < len(pauses):
            return f"[pause {pauses[i]['duration']:.1f}s]"
        return "[pause]"
    return re.sub(r"\[\.\.\.\]", repl, transcript)

def count_transcript_words(transcript):
    """Word count for the speech-rate calculation, with pause markers stripped —
    those are our own annotations, not something the student actually said."""
    stripped = re.sub(r"\[pause[^\]]*\]", " ", transcript)
    return len(re.findall(r"[A-Za-z']+", stripped))

# Content-scoring prompt. The band descriptor text below is the verbatim official
# wording from IELTS's own "Speaking Band Descriptors" document
# (ielts.org/cdn/ielts-guides/ielts-speaking-band-descriptors.pdf), covering the full
# Band 1-9 range for the three criteria Gemini scores — not the abbreviated Band 4-9
# paraphrase this used to carry. Pronunciation is intentionally left out here and
# scored separately from raw audio below (a transcript alone can't judge it).
CONTENT_SCORING_PROMPT_TEMPLATE = """You are an IELTS Speaking examiner. Score the candidate's spoken response below using the official IELTS Speaking band descriptors.

Official marking rule: a candidate must fully fit ALL the positive features of a band's descriptor to be awarded that band — not just some of them. If a response only partially matches a band's description, award the lower band. Do not round up for effort or partial credit, and do not default to a middle score out of caution.

The transcript is verbatim, produced from real audio: [pause Xs] marks a timed silence of X seconds, er/um/uh are fillers, and repeated or corrected phrases are false starts/self-corrections. Occasional pauses under 1 second are normal and should not be penalised. But count the actual pauses and fillers: several pauses of 1-2 seconds, or any single pause over 2 seconds, is concrete evidence of "long pauses while searching for words" (Band 4) or "relies on repetition and self-correction... and/or slow speech" (Band 5) — that evidence should pull the fluency score down accordingly even if vocabulary and grammar are otherwise fine, unless the pauses are clearly content-related (genuinely thinking through a complex idea) rather than language-related (struggling to find a word or restart a sentence).

Isolated fillers ("um", "uh") on their own are NOT automatically a fluency problem — the official Band 9 descriptor explicitly says hesitation is fine when it is "used only to prepare the content of the next utterance and not to find words or grammar." A filler followed by fluent continuation is normal native-like speech, not a defect; only treat fillers as evidence against a band when they cluster with the pauses/repetition/self-correction patterns described above.

This is a SPOKEN test, not a written one. Do not penalise Lexical Resource or Grammar for natural conversational register — contractions, casual vocabulary (e.g. "big" instead of "large corporations", "vibe" instead of "atmosphere"), and informal phrasing are normal, expected, and often evidence of natural fluency, not errors. Only flag genuine word-choice, collocation, or grammatical mistakes — not a candidate simply speaking casually instead of formally. Do not suggest "corrections" that just formalise natural spoken register.
{question_block}
Candidate's response (verbatim transcript):
\"\"\"
{transcript}
\"\"\"

Score three criteria — Fluency & Coherence, Lexical Resource, and Grammatical Range & Accuracy — each from 1.0 to 9.0 in 0.5 steps, using these official IELTS descriptors:

FLUENCY AND COHERENCE:
Band 9: Fluent with only very occasional repetition or self-correction. Any hesitation that occurs is used only to prepare the content of the next utterance and not to find words or grammar. Speech is situationally appropriate and cohesive features are fully acceptable. Topic development is fully coherent and appropriately extended.
Band 8: Fluent with only very occasional repetition or self-correction. Hesitation may occasionally be used to find words or grammar, but most will be content related. Topic development is coherent, appropriate and relevant.
Band 7: Able to keep going and readily produce long turns without noticeable effort. Some hesitation, repetition and/or self-correction may occur, often mid-sentence and indicate problems with accessing appropriate language. However, these will not affect coherence. Flexible use of spoken discourse markers, connectives and cohesive features.
Band 6: Able to keep going and demonstrates a willingness to produce long turns. Coherence may be lost at times as a result of hesitation, repetition and/or self-correction. Uses a range of spoken discourse markers, connectives and cohesive features though not always appropriately.
Band 5: Usually able to keep going, but relies on repetition and self-correction to do so and/or on slow speech. Hesitations are often associated with mid-sentence searches for fairly basic lexis and grammar. Overuse of certain discourse markers, connectives and other cohesive features. More complex speech usually causes disfluency but simpler language may be produced fluently.
Band 4: Unable to keep going without noticeable pauses. Speech may be slow with frequent repetition. Often self-corrects. Can link simple sentences but often with repetitious use of connectives. Some breakdowns in coherence.
Band 3: Frequent, sometimes long, pauses occur while candidate searches for words. Limited ability to link simple sentences and go beyond simple responses to questions. Frequently unable to convey basic message.
Band 2: Lengthy pauses before nearly every word. Isolated words may be recognisable but speech is of virtually no communicative significance.
Band 1: Essentially none. Speech is totally incoherent.

LEXICAL RESOURCE:
Band 9: Total flexibility and precise use in all contexts. Sustained use of accurate and idiomatic language.
Band 8: Wide resource, readily and flexibly used to discuss all topics and convey precise meaning. Skilful use of less common and idiomatic items despite occasional inaccuracies in word choice and collocation. Effective use of paraphrase as required.
Band 7: Resource flexibly used to discuss a variety of topics. Some ability to use less common and idiomatic items and an awareness of style and collocation is evident though inappropriacies occur. Effective use of paraphrase as required.
Band 6: Resource sufficient to discuss topics at length. Vocabulary use may be inappropriate but meaning is clear. Generally able to paraphrase successfully.
Band 5: Resource sufficient to discuss familiar and unfamiliar topics but there is limited flexibility. Attempts paraphrase but not always with success.
Band 4: Resource sufficient for familiar topics but only basic meaning can be conveyed on unfamiliar topics. Frequent inappropriacies and errors in word choice. Rarely attempts paraphrase.
Band 3: Resource limited to simple vocabulary used primarily to convey personal information. Vocabulary inadequate for unfamiliar topics.
Band 2: Very limited resource. Utterances consist of isolated words or memorised utterances. Little communication possible without the support of mime or gesture.
Band 1: No resource bar a few isolated words. No communication possible.

GRAMMATICAL RANGE AND ACCURACY:
Band 9: Structures are precise and accurate at all times, apart from 'mistakes' characteristic of native speaker speech.
Band 8: Wide range of structures, flexibly used. The majority of sentences are error free. Occasional inappropriacies and non-systematic errors occur. A few basic errors may persist.
Band 7: A range of structures flexibly used. Error-free sentences are frequent. Both simple and complex sentences are used effectively despite some errors. A few basic errors persist.
Band 6: Produces a mix of short and complex sentence forms and a variety of structures with limited flexibility. Though errors frequently occur in complex structures, these rarely impede communication.
Band 5: Basic sentence forms are fairly well controlled for accuracy. Complex structures are attempted but these are limited in range, nearly always contain errors and may lead to the need for reformulation.
Band 4: Can produce basic sentence forms and some short utterances are error-free. Subordinate clauses are rare and, overall, turns are short, structures are repetitive and errors are frequent.
Band 3: Basic sentence forms are attempted but grammatical errors are numerous except in apparently memorised utterances.
Band 2: No evidence of basic sentence forms.
Band 1: No rateable language unless memorised.

Do not score Pronunciation — that is assessed separately from the raw audio.

Respond with ONLY a JSON object, no markdown, no extra text, in this exact structure:
{{"fluency_band":6.5,"lexical_band":6.5,"grammar_band":6.5,"feedback":{{"fluency":"2-3 sentences","lexical":"2-3 sentences","grammar":"2-3 sentences"}},"corrections":[{{"original":"exact phrase said, copied verbatim from the transcript so it can be located and highlighted","better":"improved version","why":"brief reason","category":"one of: Word Form, Word Choice, Collocation, Naturalness, Clarity, Grammar"}}]}}
corrections: max 5, real errors only, empty array if none. "original" must be an exact substring of the transcript above (not paraphrased), so it can be matched and highlighted in place."""

async def score_content_gemini(annotated_transcript, question=""):
    question_block = f'\n\nThe question/topic the candidate was responding to: "{question}"\n' if question else ""
    prompt = CONTENT_SCORING_PROMPT_TEMPLATE.format(question_block=question_block, transcript=annotated_transcript)
    contents = [{"role": "user", "parts": [{"text": prompt}]}]
    text = await call_gemini(contents, {"maxOutputTokens": 4096, "temperature": 0.3, "responseMimeType": "application/json"})
    return parse_json_loose(text)

# ── AZURE PRONUNCIATION ASSESSMENT (unscripted) ──────────────────

def assess_pronunciation_azure_unscripted(audio_data, sample_rate=16000, timeout=60):
    """Score pronunciation/fluency directly from open-ended audio, with no reference
    text (unlike the ted-pronunciation Space's word-level assess_with_azure(), which
    needs a known target word). Uses continuous recognition since a full response can
    span multiple recognized segments; scores are aggregated across all of them.

    This function is synchronous and blocks for up to `timeout` seconds
    (threading.Event().wait()). The /analyse route always calls it via
    asyncio.to_thread() — calling it directly from an async route would freeze the
    whole single-worker event loop for up to a minute, including Render's own health
    checks (every ~5s), which is exactly what caused live 502s before this fix."""
    if not AZURE_SPEECH_KEY:
        return None
    wav_path = save_wav_temp(audio_data, sample_rate)
    try:
        speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
        speech_config.speech_recognition_language = "en-US"
        audio_config = speechsdk.audio.AudioConfig(filename=wav_path)
        pronunciation_config = speechsdk.PronunciationAssessmentConfig(
            reference_text="",
            grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
            granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
            enable_miscue=True
        )
        try:
            pronunciation_config.enable_prosody_assessment()
        except AttributeError:
            pass  # older SDK without prosody support — safe to skip

        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        pronunciation_config.apply_to(recognizer)

        segments = []
        done = threading.Event()

        def on_recognized(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                pa = speechsdk.PronunciationAssessmentResult(evt.result)
                phoneme_scores = []
                for word_result in (pa.words or []):
                    if not (hasattr(word_result, 'phonemes') and word_result.phonemes):
                        continue
                    for ph in word_result.phonemes:
                        try:
                            accuracy = ph.accuracy_score
                        except AttributeError:
                            try:
                                accuracy = ph.pronunciation_assessment.accuracy_score
                            except Exception:
                                continue
                        phoneme_scores.append({
                            "phoneme": ph.phoneme.lower(),
                            "accuracy": accuracy,
                            "word": word_result.word
                        })
                segments.append({
                    "text": evt.result.text,
                    "accuracy": pa.accuracy_score,
                    "fluency": pa.fluency_score,
                    "completeness": pa.completeness_score,
                    "pronunciation": pa.pronunciation_score,
                    "prosody": getattr(pa, "prosody_score", None),
                    "phoneme_scores": phoneme_scores
                })

        def on_stopped(_evt):
            done.set()

        recognizer.recognized.connect(on_recognized)
        recognizer.session_stopped.connect(on_stopped)
        recognizer.canceled.connect(on_stopped)

        recognizer.start_continuous_recognition()
        done.wait(timeout=timeout)
        recognizer.stop_continuous_recognition()

        if not segments:
            return None

        def avg(key):
            vals = [s[key] for s in segments if s.get(key) is not None]
            return sum(vals) / len(vals) if vals else None

        all_phoneme_scores = [p for s in segments for p in s.get("phoneme_scores", [])]

        return {
            "recognised": " ".join(s["text"] for s in segments).strip(),
            "accuracy": avg("accuracy"),
            "fluency": avg("fluency"),
            "completeness": avg("completeness"),
            "pronunciation": avg("pronunciation"),
            "prosody": avg("prosody"),
            "segments": len(segments),
            "phoneme_scores": all_phoneme_scores
        }
    except Exception as e:
        print(f"Azure unscripted assessment error: {e}")
        return None
    finally:
        try:
            os.unlink(wav_path)
        except:
            pass

def weighted_accuracy_from_phonemes(phoneme_scores):
    """Vietnamese-learner-weighted average accuracy across every phoneme actually
    detected in the response — vowels and the hard consonants (θ, ð, v, z, ʒ, dʒ)
    count for more than easy consonants, same weighting the ted-pronunciation Space
    already uses. This is what actually grounds the pronunciation score in verified
    per-sound correctness, rather than trusting Azure's black-box PronScore alone."""
    if not phoneme_scores:
        return None
    total_score = 0.0
    total_weight = 0.0
    for p in phoneme_scores:
        ipa = SAPI_TO_IPA.get(p["phoneme"], p["phoneme"])
        weight = 1.5 if ipa in VOWEL_PHONEMES else 1.3 if ipa in HARD_CONSONANTS else 1.0
        total_score += p["accuracy"] * weight
        total_weight += weight
    return total_score / total_weight if total_weight else None

def classify_phoneme_severity(accuracy):
    """Discrete correct/warning/incorrect tier instead of a raw percentage — the same
    3-tier presentation ELSA's API uses (their decision field: correct/warning/
    incorrect), easier for a student to act on than a number. The 60 cutoff for
    "incorrect" isn't arbitrary: Azure's own docs define AccuracyScore < 60 as their
    internal Mispronunciation error type, so we're reusing Azure's own threshold
    rather than inventing one."""
    if accuracy >= 80:
        return "correct"
    if accuracy >= 60:
        return "warning"
    return "incorrect"

def weakest_sounds(phoneme_scores, top_n=5):
    """Group every detected phoneme instance by sound, average its accuracy, and
    return the worst top_n — concrete, actionable "sounds to work on" feedback,
    not just a single opaque number."""
    if not phoneme_scores:
        return []
    by_phoneme = {}
    for p in phoneme_scores:
        ipa = SAPI_TO_IPA.get(p["phoneme"], p["phoneme"])
        entry = by_phoneme.setdefault(ipa, {"scores": [], "example_words": []})
        entry["scores"].append(p["accuracy"])
        word = p.get("word")
        if word and word not in entry["example_words"] and len(entry["example_words"]) < 3:
            entry["example_words"].append(word)
    summary = [
        {
            "phoneme": ipa,
            "avg_accuracy": round(avg_acc, 1),
            "severity": classify_phoneme_severity(avg_acc),
            "count": len(d["scores"]),
            "example_words": d["example_words"]
        }
        for ipa, d in by_phoneme.items()
        for avg_acc in [sum(d["scores"]) / len(d["scores"])]
    ]
    summary.sort(key=lambda x: x["avg_accuracy"])
    return summary[:top_n]

def classify_prosody(prosody_score):
    """Separates rhythm/stress/intonation feedback from individual-sound accuracy,
    the same distinction ELSA draws between its "pronunciation" and "intonation"
    scores — but reported as a secondary diagnostic note here rather than a second
    band, since IELTS's own Pronunciation criterion officially combines both into
    one score (see the band descriptors: "chunking", "stress-timing", "rhythm" all
    sit inside the same Pronunciation column, not a separate criterion)."""
    if prosody_score is None:
        return None
    if prosody_score >= 80:
        return {"level": "good", "note": "Good use of stress, rhythm, and intonation."}
    if prosody_score >= 60:
        return {"level": "mixed", "note": "Stress and intonation are inconsistent — rhythm may be affected by a lack of stress-timing or a rushed pace at times."}
    return {"level": "limited", "note": "Limited control of stress and intonation — practice the natural rhythm of English rather than speaking word-by-word."}

# Piecewise-linear heuristic mapping a 0-100 pronunciation sub-score to an
# IELTS-style 1.0-9.0 band. There is no
# official Microsoft-to-IELTS conversion table — this is
# a judgment-call estimate, calibrated against how the official band descriptors
# read (e.g. Band 5 "L1 accent affects intelligibility at times" implies a score
# well below native-level, not close to it) rather than a straight-line guess.
# It's deliberately conservative in the 55-85 range, since that's where most real
# L2 speakers with a noticeable-but-intelligible accent land. Replace these anchor
# points with real data once we have Azure scores paired with actual examiner-
# assigned pronunciation bands (see the plan to calibrate against graded examples).
def _interpolate_piecewise(value, anchors):
    """Shared piecewise-linear lookup: anchors is a list of (x, band) pairs, sorted by
    x, mapping a raw measurement to an IELTS-style 1.0-9.0 band with anchor points in
    between. Used for every "we don't have an official conversion table, but here's a
    documented, defensible estimate" mapping in this file."""
    xs = [a[0] for a in anchors]
    value = max(xs[0], min(xs[-1], value))
    band = anchors[-1][1]
    for (x0, b0), (x1, b1) in zip(anchors, anchors[1:]):
        if x0 <= value <= x1:
            band = b0 + (value - x0) / (x1 - x0) * (b1 - b0)
            break
    band = round(band * 2) / 2
    return max(1.0, min(9.0, band))

PRON_SCORE_BAND_ANCHORS = [(0, 1.0), (40, 2.5), (55, 4.0), (65, 5.0), (75, 6.0), (85, 7.0), (92, 8.0), (100, 9.0)]

def pron_score_to_band(score):
    """Estimate only — must be labeled as such wherever it's shown."""
    if score is None:
        return None
    return _interpolate_piecewise(score, PRON_SCORE_BAND_ANCHORS)

def pronunciation_band_from_components(weighted_accuracy, prosody):
    """Weakest-link, not weighted-average: the Pronunciation band is capped by
    whichever component is worse, not pulled up by the stronger one. individual
    sound accuracy and rhythm/stress/intonation are two genuinely distinct elements
    of the official Pronunciation descriptors (see the "chunking... rhythm...
    intonation and stress" language alongside "individual words or phonemes...
    mispronounced" at every band level) — a candidate strong in one but weak in the
    other hasn't fully met the positive features of the higher band on both fronts.

    This mirrors two independent sources: IELTS's own official marking rule ("a
    candidate must fully fit ALL the positive features of a band's descriptor to be
    awarded it" — already baked into the content-scoring prompt above) and a real
    competitor's stated approach: ieltsscience.fun explicitly labels this "Điểm
    criteria bị giới hạn bởi tiêu chí thấp nhất" (the criterion score is limited by
    its lowest sub-criterion) directly in its pronunciation breakdown UI. Previously
    this was a 60/40 weighted average, which let a strong accuracy score paper over
    weak prosody (or vice versa) — a real source of over-generous scores."""
    accuracy_band = pron_score_to_band(weighted_accuracy)
    prosody_band = pron_score_to_band(prosody)
    bands = [b for b in [accuracy_band, prosody_band] if b is not None]
    if not bands:
        return None
    return min(bands)

# ── FLUENCY: deterministic speech-timing composite ───────────────────
#
# De Jong et al. (2012) broke L2 utterance fluency into speed, breakdown, and repair
# fluency, measured as speech rate, mean length of run (words spoken fluently between
# pauses), and pause frequency/duration. This is the empirical backbone of deployed
# scorers like ETS's SpeechRater. A meta-analysis (Suzuki et al., 2021) reports these
# correlate with human proficiency judgments at r=.76 (speech rate), r=.72 (mean
# length of run), r=-.59 (pause frequency) — not incidental correlations.
#
# A directly relevant August 2026 paper (Uehara, "...Why Pause Encoding Does Not
# Change LLM Fluency Scores") ran a controlled test on exactly the approach this app
# used before this change: embedding pause markers inline in the transcript text and
# relying on an LLM to weigh them. Finding: inline pause encoding does NOT reliably
# improve an LLM's fluency judgment over just giving it aggregate stats — "the fluency
# signal comes from the measured speech-timing features, not from how pauses are
# written for the LLM." Their winning approach computed a deterministic composite
# SEPARATELY from the LLM, then blended the two scores mathematically, reaching higher
# agreement with human raters than 81% of individual trained human raters.
#
# We don't have their calibration data, so the anchor points below are a documented,
# transparent judgment call informed by published speech-rate/fluency norms — not a
# reproduction of their exact formula. Replace with real data once available.
#
# MLR and pause-ratio anchors were loosened after testing against a real reference
# sample (a spoken response labeled Band 9): genuinely fluent, native-quality
# spontaneous speech ran ~23% pause time and a mean run of ~7 words, which the
# original anchors scored around Band 5 — treating normal, content-related thinking
# pauses (explicitly allowed at Band 8-9 per the official descriptor: "hesitation is
# used only to prepare the content... not to find words or grammar") as a fluency
# deficit just because they exist. Raw pause time can't distinguish *why* a pause
# happened, so the curve was widened rather than trying to fix that blind spot.
# Speech rate wasn't adjusted — it already placed that sample appropriately high.

SPEECH_RATE_BAND_ANCHORS = [(0, 1.0), (50, 3.0), (70, 4.0), (90, 5.0), (110, 6.0), (130, 7.0), (150, 8.0), (180, 9.0)]
MLR_BAND_ANCHORS = [(0, 1.0), (2, 3.0), (3, 4.0), (4, 5.0), (5, 6.0), (7, 7.5), (10, 8.5), (14, 9.0)]
PAUSE_RATIO_BAND_ANCHORS = [(0.0, 9.0), (0.10, 8.5), (0.20, 8.0), (0.30, 7.0), (0.40, 5.5), (0.50, 4.0), (0.60, 2.5), (0.75, 1.0)]

def compute_speech_timing_metrics(word_count, duration_seconds, pauses):
    """The three De Jong features, computed directly from what we already precisely
    measure: word_count from the (pause-marker-stripped) transcript, duration_seconds
    from the actual audio, pauses from detect_pauses()."""
    if duration_seconds <= 0 or word_count <= 0:
        return None
    total_pause_time = sum(p["duration"] for p in pauses)
    return {
        "speech_rate_wpm": round(word_count / (duration_seconds / 60), 1),
        "pause_ratio": round(total_pause_time / duration_seconds, 3),
        "mean_length_of_run": round(word_count / (len(pauses) + 1), 1)
    }

def speech_timing_band(metrics):
    """Deterministic fluency band from the three De Jong features alone, averaged —
    computed entirely independently of the LLM, per the research above."""
    if not metrics:
        return None
    bands = [
        _interpolate_piecewise(metrics["speech_rate_wpm"], SPEECH_RATE_BAND_ANCHORS),
        _interpolate_piecewise(metrics["mean_length_of_run"], MLR_BAND_ANCHORS),
        _interpolate_piecewise(metrics["pause_ratio"], PAUSE_RATIO_BAND_ANCHORS),
    ]
    return round((sum(bands) / len(bands)) * 2) / 2

def blend_fluency_band(deterministic_band, llm_band):
    """Weakest-link, not weighted-average — same principle as
    pronunciation_band_from_components() above, applied here because speech-timing
    (speed/pauses/runs) and coherence (topic development, discourse markers) are two
    genuinely distinct elements of the single Fluency & Coherence criterion. A
    candidate with fast, low-pause delivery but disjointed, hard-to-follow content
    (or the reverse: coherent but halting) hasn't fully met the positive features of
    the higher band on both fronts, so the lower of the two caps the result rather
    than being averaged up by the stronger one."""
    bands = [b for b in [deterministic_band, llm_band] if b is not None]
    if not bands:
        return None
    return min(bands)

def combine_overall_band(fluency, lexical, grammar, pronunciation):
    """Average the 4 criteria and apply IELTS's real rounding convention: round to the
    nearest 0.5, with .25/.75 averages rounding UP (not to-even or down)."""
    scores = [s for s in [fluency, lexical, grammar, pronunciation] if s is not None]
    if not scores:
        return None
    avg = sum(scores) / len(scores)
    doubled = avg * 2 + 1e-9
    floor_val = int(doubled)
    frac = doubled - floor_val
    rounded = floor_val + 1 if frac >= 0.5 else floor_val
    return rounded / 2

# ── ENDPOINT ──────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", encoding="utf-8") as f:
        return f.read()

@app.post("/extract-question")
async def extract_question(file: UploadFile = File(...)):
    if not GEMINI_API_KEY:
        return {"error": "Question extraction unavailable — GEMINI_API_KEY not set in Space settings."}
    mime_type = file.content_type or ""
    if not (mime_type.startswith("image/") or mime_type == "application/pdf"):
        return {"error": "Please upload an image or PDF file."}
    contents = await file.read()
    question = await extract_question_from_file(contents, mime_type)
    if not question:
        return {"error": "Could not read the question from this file — please try a clearer file, or type the question manually."}
    return {"question": question}

@app.post("/analyse")
async def analyse_speaking(question: str = Form(""), file: UploadFile = File(...)):
    if not GEMINI_API_KEY:
        return {"error": "Speaking analysis unavailable — GEMINI_API_KEY not set in Space settings."}
    if not AZURE_SPEECH_KEY:
        return {"error": "Speaking analysis unavailable — AZURE_SPEECH_KEY not set in Space settings."}

    contents = await file.read()
    wav_bytes_in = await asyncio.to_thread(transcode_to_wav, contents)
    if wav_bytes_in is None:
        return {"error": "Could not read this audio file — please try a different recording."}
    try:
        audio_data, _ = sf.read(io.BytesIO(wav_bytes_in))
    except Exception:
        return {"error": "Could not read this audio file — please try a different recording."}
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]
    audio_data = audio_data.astype(np.float32)

    duration = len(audio_data) / 16000
    if duration < 3.0:
        return {"error": "Recording too short — speak for at least a few seconds so there's enough to assess"}
    rms = float(np.sqrt(np.mean(audio_data ** 2)))
    if rms < 0.005:
        return {"error": "No speech detected — please check the recording and try again"}

    clean_audio = preprocess_audio(audio_data)
    rms_after = float(np.sqrt(np.mean(clean_audio ** 2)))
    if rms_after < 0.005:
        return {"error": "Too much background noise — please use a quieter recording"}

    pauses = detect_pauses(clean_audio)

    wav_buf = io.BytesIO()
    sf.write(wav_buf, clean_audio, 16000, format='WAV', subtype='PCM_16')
    wav_bytes = wav_buf.getvalue()

    # Azure's pronunciation assessment only needs clean_audio — it has no dependency
    # on the transcript or content scores below, so start it now and let it run
    # concurrently with the (necessarily sequential) Gemini calls, instead of
    # waiting for both of them to finish first. This was previously the single
    # biggest reason a request took longer than it needed to: Azure's continuous
    # recognition can itself take several seconds to tens of seconds, and it was
    # being run strictly after ~2 sequential Gemini round-trips instead of alongside
    # them.
    azure_task = asyncio.create_task(asyncio.to_thread(assess_pronunciation_azure_unscripted, clean_audio))

    raw_transcript = await transcribe_verbatim_gemini(wav_bytes, len(pauses))
    if not raw_transcript:
        azure_task.cancel()
        return {"error": "Transcription failed — please try again."}
    annotated_transcript = merge_pause_durations(raw_transcript, pauses)

    content_scores = await score_content_gemini(annotated_transcript, question)
    if not content_scores:
        azure_task.cancel()
        return {"error": "Scoring failed — please try again."}

    azure_result = await azure_task
    weighted_accuracy = None
    weak_sounds = []
    pronunciation_band = None
    prosody_feedback = None
    if azure_result:
        weighted_accuracy = weighted_accuracy_from_phonemes(azure_result.get('phoneme_scores', []))
        weak_sounds = weakest_sounds(azure_result.get('phoneme_scores', []))
        prosody_feedback = classify_prosody(azure_result.get('prosody'))
        pronunciation_band = pronunciation_band_from_components(weighted_accuracy, azure_result.get('prosody'))

    word_count = count_transcript_words(annotated_transcript)
    timing_metrics = compute_speech_timing_metrics(word_count, duration, pauses)
    deterministic_fluency_band = speech_timing_band(timing_metrics)
    llm_fluency_band = content_scores.get('fluency_band')
    fluency_band = blend_fluency_band(deterministic_fluency_band, llm_fluency_band)
    lexical_band = content_scores.get('lexical_band')
    grammar_band = content_scores.get('grammar_band')
    overall_band = combine_overall_band(fluency_band, lexical_band, grammar_band, pronunciation_band)

    return {
        "transcript": annotated_transcript,
        "question": question,
        "pauses": pauses,
        "bands": {
            "fluency": fluency_band,
            "lexical": lexical_band,
            "grammar": grammar_band,
            "pronunciation": pronunciation_band,
            "overall": overall_band
        },
        "pronunciation_band_is_estimate": True,
        "fluency_detail": {
            "speech_timing_metrics": timing_metrics,
            "deterministic_band": deterministic_fluency_band,
            "llm_band": llm_fluency_band
        },
        "weakest_sounds": weak_sounds,
        "prosody": prosody_feedback,
        "feedback": content_scores.get('feedback', {}),
        "corrections": content_scores.get('corrections', []),
        "azure_raw": azure_result
    }
