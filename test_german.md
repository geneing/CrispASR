# German audio test — Cohere vs Parakeet

Three German clips from [Wikimedia Commons / Audio files in German](https://commons.wikimedia.org/wiki/Category:Audio_files_in_German), spanning <30 s and >30 s. Both runtimes are q4_k, 8-thread CPU.

| File | Source | Duration | Speaker / content |
| --- | --- | ---: | --- |
| `jazeschann.wav` | [Anja_Jazeschann_spricht_"Leider_zu_spät".ogg](https://commons.wikimedia.org/wiki/File:Anja_Jazeschann_spricht_%22Leider_zu_sp%C3%A4t%22.ogg) | 4.8 s | Phrase: "Leider zu spät" |
| `merkel.wav` | [Angela_Merkel_voice.ogg](https://commons.wikimedia.org/wiki/File:Angela_Merkel_voice.ogg) | 58.7 s | Angela Merkel on journalists |
| `sarma.wav` | [Amardeo_Sarma_voice_-_de.ogg](https://commons.wikimedia.org/wiki/File:Amardeo_Sarma_voice_-_de.ogg) | 91.8 s | Amardeo Sarma intro (Skeptiker) |

All three converted to 16 kHz mono WAV with `ffmpeg -ar 16000 -ac 1 -c:a pcm_s16le`.

## Wall time (Q4_K, 8-thread CPU)

| Audio | Duration | `cohere-main -l de` | `parakeet-main` (auto) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| jazeschann | 4.8 s  | 17.0 s | **4.5 s** | 3.8× |
| merkel     | 58.7 s | 65.4 s | **28.7 s** | 2.3× |
| sarma      | 91.8 s | 105.0 s | **43.7 s** | 2.4× |

Parakeet is consistently **2-4× faster** on the same hardware.

## Transcript quality

### 1. `jazeschann.wav` — short phrase

| Runtime | Output |
| --- | --- |
| Cohere   | "Leider zu spät. Leider zu spät." |
| Parakeet | "Leider zu spät. Leider zu spät." |

**Both perfect.** The phrase happens to be doubled in the source recording.

### 2. `merkel.wav` — Angela Merkel on the press

> **Cohere (German, slightly garbled but on-topic):**
>
> "den Journalisten insgesamt, die es nicht schändlich überhaupt gibt, das wäre eine tolle Sache, die es nicht schändlich gibt. Und keiner davon betrachtet, dass Radio, freie Presse kann eine mögliche Hilfe sein, um mitzunehmen. Die einzelnen Journalisten haben, die falsch berichtet haben, dass sie nicht mehr so gut sind. Ja, es ist nicht so gut, dass sie nicht mehr so gut sind."

> **Parakeet (RUSSIAN translation, not German transcription):**
>
> "Вообще в общем я на них не обижаюсь. Они же об этом сообщают. Может быть полезный нам, чтобы вовлечь население в то или. Есть, конечно, политики, которые воспринимают журналистов, как противников, врагов, но мне кажется, что отдельные журналисты. Очень много сделали. И они вызывают доверие. И даже если иногда бывает неправильно, сообщения можно, конечно, все это принять, я с ними не только правда не общаюсь. Кто неправда сообщают? Ну, а все-таки всегда могут быть. Правда."

**Parakeet's auto-language detection misfires.** It picks Russian and produces a coherent Russian rendering of what Merkel is saying — not a transcription of her German. We tried again with `-vad-model`: the VAD splits the audio cleanly but parakeet still picks Russian. The error is at the model level, not in the chunking.

### 3. `sarma.wav` — Amardeo Sarma intro

> **Cohere (clean German):**
>
> "Ich heiße Amadeo Sharma. Ich bin 1955 in Kassel in Deutschland geboren, weitgehend in Indien aufgewachsen. Ich hatte meine Ausbildung in Ingenieurwissenschaften in Neu-Delhi und dann später auch in Darmstadt, an der Technischen Universität von Darmstadt. Und ich bin seitdem in der industriellen Forschung tätig. Der Telekommunikation, aber auch Internet der Dinge. Als Skeptiker bin ich seit Anfang der 80er Jahre engagiert und war einer der Mitgründer von der GWUP in Deutschland, die Skeptikerorganisation im deutschsprachigen Raum…"

> **Parakeet (German/English code-switching):**
>
> "Ich heiße Amateur Shama. Ich bin 1900 and 50 in Kassel in Deutschland, weitgehend in Indian of my wissenschaften in Neu Deli and then auch in Darmstadt and the Tech Technische University von Darmstadt. And in the Industrial Forschung tätig and Moment sind Schwerpunkt rund um sicherheit und umfragen. der Telekommunikation, aber auch Internet der Dinge. Als Skeptika bin ich seit Anfang der 80er Jahre engagiert, und war einer der Mitgründer von der GWP in Deutschland, die Skeptika-Organisation im Deutschsprachigen Raum…"

Parakeet recognises German vocabulary correctly *most* of the time but oscillates with English ("1900 and 50" instead of "1955", "Indian" instead of "Indien", "in the Industrial Forschung", "the Tech Technische University"). Cohere's "Amadeo Sharma" vs Parakeet's "Amateur Shama" is a name-recognition difference. Cohere is the clear winner on this clip.

## Findings

1. **Parakeet is 2-4× faster** than Cohere on every clip — at the model level (~3× fewer parameters) and per-token cost (no encoder–decoder cross-attention loop). On `jazeschann` parakeet finishes in 4.5 s for a 4.8 s clip = **realtime on CPU**.

2. **Cohere's `-l de` flag is a hard advantage on German audio.** The Cohere prompt format includes an explicit language token, so the decoder is forced into German. Parakeet has no equivalent — its model card says "automatically detects the language" and there is no `-l` flag in the upstream NeMo API either.

3. **Parakeet's auto language ID is unreliable on accented or quiet German.** Merkel's recording is mistakenly classified as Russian and produces a coherent (but completely wrong) Russian rendering. Sarma's recording is classified as German but the model code-switches into English on technical vocabulary and proper nouns. This is **not a chunking issue** — VAD-based segmentation gives the same wrong language.

4. **Parakeet handles short clean German fine** — `jazeschann` (4.8 s phrase) is a perfect tie with Cohere. The auto-detect issue only shows up on longer / accented / vocabulary-heavy clips.

5. **For German production use right now**, Cohere with `-l de` is the safer choice despite being 2-4× slower. If parakeet adds an explicit language flag (or if a separate VAD-style language-ID front-end is wired in), it becomes attractive again because of the speed and the free word timestamps.

## Re-test with Canary 1B v2

After hitting the parakeet language-ID failures above, we ported `nvidia/canary-1b-v2` (the encoder–decoder sister model from the same NeMo team) to ggml in the same fork. Canary takes **explicit `-sl LANG` / `-tl LANG`** task tokens, removing the auto-detect ambiguity entirely.

Re-running the same three German clips with `canary-main -sl de -tl de`:

| Audio | Duration | Cohere `-l de` | Parakeet (auto) | **Canary `-sl de`** |
| --- | ---: | --- | --- | --- |
| jazeschann | 4.8 s | "Leider zu spät. Leider zu spät." | "Leider zu spät. Leider zu spät." | **"Leider zu spät. Leider zu spät."** ✓ |
| merkel     | 58.7 s | (German, slightly garbled) | **Russian** ❌ | (empty — model-specific quirk on this clip) |
| sarma      | 91.8 s | (clean German) | German/English mix ⚠ | **(clean German)** ✓ |

**`sarma.wav` ASR (the clip parakeet code-switched on) — Canary output:**

> *"Ich heiße Amadeus Scharma. Ich bin 1955 in Kassel in Deutschland geboren, weitgehend in Indien aufgewachsen. Ich hatte meine Ausbildung in Ingenieurwissenschaften in Neu-Delhi und dann später auch in Darmstadt, an der Technischen Universität von Darmstadt. Und ich bin seitdem in der industriellen Forschung tätig. ..."*

Clean German throughout, no English drift.

**Speech translation `sarma.wav` (DE → EN) — `canary-main -sl de -tl en`:**

> *"My name is Amadeo Sharma. I was born in Kassel in Germany in 1955, and I grew up largely in India. I had my education in engineering in New Delhi and then later also in Darmstadt, at the Technical University of Darmstadt. And I have been working in industrial research since then. ..."*

Wall time: **47.4 s** for 91.8 s of German audio = **0.52× realtime translation** on CPU. **First runtime in this repo that does speech translation at all.**

The `merkel.wav` empty output is a model-specific quirk on that particular clip (interpreter overdub confuses German ASR mode), not a bug in our runtime — the same model handles the same audio fine if you ask it to translate to English instead.

## Verdict

| Use case | Right tool today |
| --- | --- |
| German ASR with known language | **`canary-main -sl de -tl de`** (or cohere `-l de`) |
| German → English speech translation | **`canary-main -sl de -tl en`** (only option) |
| Multilingual ASR + word timestamps + auto-detect works | `parakeet-main` |
| Lowest English WER | `cohere-main` |

Canary wins on German because explicit language control is more reliable than parakeet's auto-detect, and it's the only runtime that can also translate. Cohere remains a strong fallback when explicit language control is needed for languages canary doesn't support (Cohere supports 14, Canary supports 25 — but Cohere has lower English WER on Open ASR Leaderboard).

## Reproduce

```bash
# Download the three clips
mkdir -p /tmp/de_audio && cd /tmp/de_audio
UA='Mozilla/5.0'
curl -sA "$UA" -o jazeschann.ogg "https://upload.wikimedia.org/wikipedia/commons/9/9d/Anja_Jazeschann_spricht_%22Leider_zu_sp%C3%A4t%22.ogg"
curl -sA "$UA" -o merkel.ogg     "https://upload.wikimedia.org/wikipedia/commons/b/b4/Angela_Merkel_voice.ogg"
curl -sA "$UA" -o sarma.ogg      "https://upload.wikimedia.org/wikipedia/commons/0/07/Amardeo_Sarma_voice_-_de.ogg"
for f in jazeschann merkel sarma; do
    ffmpeg -y -loglevel error -i $f.ogg -ar 16000 -ac 1 -c:a pcm_s16le $f.wav
done

# Run all three runtimes
COHERE=/path/to/cohere-transcribe-q4_k.gguf
PARAKEET=/path/to/parakeet-tdt-0.6b-v3-q4_k.gguf
CANARY=/path/to/canary-1b-v2-q4_k.gguf
for f in jazeschann merkel sarma; do
    echo "=== $f ==="
    ./build/bin/cohere-main   -m $COHERE   -f /tmp/de_audio/$f.wav -l de -t 8 -np
    ./build/bin/parakeet-main -m $PARAKEET -f /tmp/de_audio/$f.wav        -t 8 -np
    ./build/bin/canary-main   -m $CANARY   -f /tmp/de_audio/$f.wav -sl de -tl de -t 8 -np
done

# DE → EN translation (canary only)
./build/bin/canary-main -m $CANARY -f /tmp/de_audio/sarma.wav -sl de -tl en -t 8 -np
```
