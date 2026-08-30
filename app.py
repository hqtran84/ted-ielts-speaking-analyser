import os
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
    body = {"contents": contents, "generationConfig": generation_config}
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
    except Exception:
        return None

# Verbatim transcription prompt — reused as-is from the talk-anhnguted-secure app's
# transcribeAudio() (talk-anhnguted-secure-5/index.html), which is already proven in
# production for capturing fillers/false starts/pauses accurately.
TRANSCRIBE_PROMPT = (
    "Transcribe this audio VERBATIM. Include all filler words (er, um, uh, like, you "
    "know), false starts, repetitions, and self-corrections exactly as spoken. Mark "
    "noticeable pauses with [...]. Do not clean up or improve the speech. Return ONLY "
    "the raw transcript text, nothing else."
)

async def transcribe_verbatim_gemini(audio_bytes, mime_type="audio/wav"):
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    contents = [{
        "role": "user",
        "parts": [
            {"text": TRANSCRIBE_PROMPT},
            {"inline_data": {"mime_type": mime_type, "data": b64}}
        ]
    }]
    text = await call_gemini(contents, {"maxOutputTokens": 1500, "temperature": 0})
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

# Content-scoring prompt — the band descriptor text below is reused verbatim from the
# talk-anhnguted-secure app's SYSTEM.part1 prompt, which is already tuned/proven in
# production. Adapted here from a per-turn examiner framing to a whole-response framing,
# and pronunciation is intentionally left out (scored separately from raw audio below).
CONTENT_SCORING_PROMPT_TEMPLATE = """You are an IELTS Speaking examiner. Score the candidate's spoken response below using the official IELTS Speaking band descriptors.

The transcript is verbatim, produced from real audio: [pause Xs] marks a timed silence of X seconds, er/um/uh are fillers, and repeated or corrected phrases are false starts/self-corrections. Use these markers to judge fluency accurately — do not penalise pauses under 1 second, but weigh longer or frequent pauses as fluency breakdowns per the descriptors below.
{question_block}
Candidate's response (verbatim transcript):
\"\"\"
{transcript}
\"\"\"

Score three criteria — Fluency & Coherence, Lexical Resource, and Grammatical Range & Accuracy — each from 1.0 to 9.0 in 0.5 steps (use 1.0-3.5 for gibberish/no real English), using these official descriptors:

FLUENCY & COHERENCE:
Band 9: Fluent, only rare repetition/self-correction, hesitation is content-related, fully coherent.
Band 8: Fluent with occasional repetition/self-correction, hesitation usually content-related, develops topics coherently.
Band 7: Speaks at length without noticeable effort, some language-related hesitation or repetition, uses connectives flexibly.
Band 6: Willing to speak at length but loses coherence at times due to repetition/self-correction/hesitation, connectives not always appropriate.
Band 5: Maintains flow but uses repetition and slow speech to keep going, over-uses certain connectives, complex communication causes fluency problems.
Band 4: Cannot always maintain flow, long pauses while searching for words, limited ability to link sentences.

LEXICAL RESOURCE:
Band 9: Full flexibility, precise, natural idiomatic use.
Band 8: Wide resource, natural, flexible, only occasional errors.
Band 7: Uses vocabulary flexibly, some awareness of style, occasional errors in word choice.
Band 6: Generally appropriate vocabulary but lacks flexibility, some errors in word choice/formation.
Band 5: Limited range, repetitive, errors may impede communication.
Band 4: Very limited range, basic vocabulary only.

GRAMMATICAL RANGE & ACCURACY:
Band 9: Full range, consistently accurate.
Band 8: Wide range, mostly accurate, minor errors only.
Band 7: Mix of simple and complex structures, some errors but not impeding communication.
Band 6: Mix of structures, errors are frequent but meaning is clear.
Band 5: Produces basic sentence forms accurately but makes errors in complex grammar.
Band 4: Mainly simple sentences, frequent errors.

Do not score Pronunciation — that is assessed separately from the raw audio.

Respond with ONLY a JSON object, no markdown, no extra text, in this exact structure:
{{"fluency_band":6.5,"lexical_band":6.5,"grammar_band":6.5,"feedback":{{"fluency":"2-3 sentences","lexical":"2-3 sentences","grammar":"2-3 sentences"}},"corrections":[{{"original":"exact phrase said","better":"improved version","why":"brief reason"}}]}}
corrections: max 5, real errors only, empty array if none."""

async def score_content_gemini(annotated_transcript, question=""):
    question_block = f'\n\nThe question/topic the candidate was responding to: "{question}"\n' if question else ""
    prompt = CONTENT_SCORING_PROMPT_TEMPLATE.format(question_block=question_block, transcript=annotated_transcript)
    contents = [{"role": "user", "parts": [{"text": prompt}]}]
    text = await call_gemini(contents, {"maxOutputTokens": 2048, "temperature": 0.3, "responseMimeType": "application/json"})
    return parse_json_loose(text)

# ── AZURE PRONUNCIATION ASSESSMENT (unscripted) ──────────────────

def assess_pronunciation_azure_unscripted(audio_data, sample_rate=16000, timeout=60):
    """Score pronunciation/fluency directly from open-ended audio, with no reference
    text (unlike the ted-pronunciation Space's word-level assess_with_azure(), which
    needs a known target word). Uses continuous recognition since a full response can
    span multiple recognized segments; scores are aggregated across all of them."""
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
                segments.append({
                    "text": evt.result.text,
                    "accuracy": pa.accuracy_score,
                    "fluency": pa.fluency_score,
                    "completeness": pa.completeness_score,
                    "pronunciation": pa.pronunciation_score,
                    "prosody": getattr(pa, "prosody_score", None)
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

        return {
            "recognised": " ".join(s["text"] for s in segments).strip(),
            "accuracy": avg("accuracy"),
            "fluency": avg("fluency"),
            "completeness": avg("completeness"),
            "pronunciation": avg("pronunciation"),
            "prosody": avg("prosody"),
            "segments": len(segments)
        }
    except Exception as e:
        print(f"Azure unscripted assessment error: {e}")
        return None
    finally:
        try:
            os.unlink(wav_path)
        except:
            pass

def pron_score_to_band(score):
    """Rough linear heuristic mapping Azure's 0-100 PronScore to an IELTS-style 1.0-9.0
    band. There is no official Microsoft-to-IELTS conversion table — this is an
    estimate only, and must be labeled as such wherever it's shown."""
    if score is None:
        return None
    band = 1.0 + (score / 100.0) * 8.0
    band = round(band * 2) / 2
    return max(1.0, min(9.0, band))

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

@app.post("/analyse")
async def analyse_speaking(question: str = Form(""), file: UploadFile = File(...)):
    if not GEMINI_API_KEY:
        return {"error": "Speaking analysis unavailable — GEMINI_API_KEY not set in Space settings."}
    if not AZURE_SPEECH_KEY:
        return {"error": "Speaking analysis unavailable — AZURE_SPEECH_KEY not set in Space settings."}

    contents = await file.read()
    wav_bytes_in = transcode_to_wav(contents)
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

    raw_transcript = await transcribe_verbatim_gemini(wav_bytes)
    if not raw_transcript:
        return {"error": "Transcription failed — please try again."}
    annotated_transcript = merge_pause_durations(raw_transcript, pauses)

    content_scores = await score_content_gemini(annotated_transcript, question)
    if not content_scores:
        return {"error": "Scoring failed — please try again."}

    azure_result = assess_pronunciation_azure_unscripted(clean_audio)
    pronunciation_band = pron_score_to_band(azure_result['pronunciation']) if azure_result else None

    fluency_band = content_scores.get('fluency_band')
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
        "feedback": content_scores.get('feedback', {}),
        "corrections": content_scores.get('corrections', []),
        "azure_raw": azure_result
    }
