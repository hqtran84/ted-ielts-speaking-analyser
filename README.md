---
title: Ted Ielts Speaking Analyser
emoji: 🗣️
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# tED IELTS Speaking Analyser

Upload or record any spoken response and get back:

- A verbatim transcript, including fillers, false starts, and precisely-timed pauses.
- Estimated IELTS Speaking band scores for all 4 official criteria (Fluency & Coherence,
  Lexical Resource, Grammatical Range & Accuracy, Pronunciation) plus an overall band.
- Written feedback and specific corrections.

## How it works

- **Pause timing** is measured directly from the waveform (RMS energy silence-gap
  detection) — precise, not guessed by an LLM.
- **Transcription** (verbatim, with fillers/false starts) is done by Gemini, given the
  raw audio.
- **Fluency, Lexical Resource, and Grammar bands** are scored by Gemini against the
  official IELTS band descriptors, using the pause-annotated transcript.
- **Pronunciation** is scored from the raw audio using Azure Cognitive Services'
  Pronunciation Assessment (unscripted/reference-free mode), then mapped to an
  estimated IELTS band via a simple linear heuristic.

## Required secrets (Space settings → Repository secrets)

- `AZURE_SPEECH_KEY` — Azure Speech resource key.
- `AZURE_SPEECH_REGION` — Azure region (defaults to `eastus` if unset).
- `GEMINI_API_KEY` — Google Gemini API key.

## Important caveat

The overall band this tool produces is a **coaching estimate**, not a certified IELTS
score. In particular, there is no official Microsoft-to-IELTS conversion table for the
Pronunciation score — that mapping is a reasonable heuristic only, and is labeled as an
estimate everywhere it's shown.
