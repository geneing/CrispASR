// voxtral-main — CLI for Mistral Voxtral-Mini-3B-2507.
//
// Reads a 16 kHz mono WAV, runs the full audio→text pipeline (mel → encoder
// → splice into [INST] prompt → Llama 3 LLM with KV cache → greedy decode),
// prints the transcript.
//
// Usage:
//   voxtral-main -m voxtral-mini-3b-2507.gguf -f audio.wav [-t 4] [-l en]

#include "voxtral.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>

static bool load_wav_16k_mono(const std::string & path, std::vector<float> & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    auto read32 = [&](uint32_t & v) { f.read((char*)&v, 4); };
    auto read16 = [&](uint16_t & v) { f.read((char*)&v, 2); };
    char riff[4]; f.read(riff, 4); uint32_t riff_size; read32(riff_size);
    char wave[4]; f.read(wave, 4);
    if (memcmp(riff,"RIFF",4)!=0||memcmp(wave,"WAVE",4)!=0) return false;
    uint16_t afmt=0,nchan=0,bps=0; uint32_t sr=0; std::vector<uint8_t> data;
    while (f) {
        char cid[4]; f.read(cid,4); uint32_t csz; read32(csz); if(!f) break;
        if (!memcmp(cid,"fmt ",4)) {
            read16(afmt); read16(nchan); read32(sr);
            uint32_t br; read32(br); uint16_t ba; read16(ba); read16(bps);
            if (csz > 16) f.seekg(csz-16, std::ios::cur);
        } else if (!memcmp(cid,"data",4)) {
            data.resize(csz); f.read((char*)data.data(), csz); break;
        } else f.seekg(csz, std::ios::cur);
    }
    if (afmt!=1||bps!=16) { fprintf(stderr, "%s: only 16-bit PCM\n", path.c_str()); return false; }
    if (sr!=16000) { fprintf(stderr, "%s: need 16kHz (got %u)\n", path.c_str(), sr); return false; }
    const int16_t * pcm = (const int16_t *)data.data();
    size_t ns = data.size()/2;
    if (nchan == 1) {
        out.resize(ns); for (size_t i=0;i<ns;i++) out[i]=pcm[i]/32768.0f;
    } else {
        size_t nf = ns/nchan; out.resize(nf);
        for (size_t i=0;i<nf;i++) { float s=0; for(int c=0;c<nchan;c++) s+=pcm[i*nchan+c]/32768.0f; out[i]=s/nchan; }
    }
    return true;
}

int main(int argc, char ** argv) {
    std::string model_path, audio_path, lang = "en";
    int n_threads = 4, max_new = 512;
    for (int i=1;i<argc;i++) {
        std::string a=argv[i];
        if (a=="-m"&&i+1<argc) model_path=argv[++i];
        else if (a=="-f"&&i+1<argc) audio_path=argv[++i];
        else if (a=="-t"&&i+1<argc) n_threads=std::atoi(argv[++i]);
        else if (a=="-l"&&i+1<argc) lang=argv[++i];
        else if (a=="-n"&&i+1<argc) max_new=std::atoi(argv[++i]);
        else if (a=="-h"||a=="--help") {
            fprintf(stderr,"usage: %s -m model.gguf -f audio.wav [-t N] [-l LANG] [-n MAX]\n",argv[0]); return 0;
        }
    }
    if (model_path.empty()||audio_path.empty()) { fprintf(stderr,"missing -m or -f\n"); return 1; }

    std::vector<float> samples;
    if (!load_wav_16k_mono(audio_path, samples)) return 2;
    fprintf(stderr, "audio: %.2f s (%zu samples)\n", samples.size()/16000.0, samples.size());

    auto cp = voxtral_context_default_params(); cp.n_threads = n_threads;
    auto * ctx = voxtral_init_from_file(model_path.c_str(), cp);
    if (!ctx) { fprintf(stderr,"init failed\n"); return 3; }

    auto t0 = std::chrono::steady_clock::now();

    // Mel
    int n_mels=0, T_mel=0;
    float * mel = voxtral_compute_mel(ctx, samples.data(), (int)samples.size(), &n_mels, &T_mel);
    if (!mel) { fprintf(stderr,"mel failed\n"); voxtral_free(ctx); return 4; }
    auto t_mel = std::chrono::steady_clock::now();
    fprintf(stderr, "mel: %d × %d  (%.0f ms)\n", n_mels, T_mel,
            std::chrono::duration<double,std::milli>(t_mel-t0).count());

    // Encoder
    int N_enc=0, pdim=0;
    float * audio_embeds = voxtral_run_encoder(ctx, mel, n_mels, T_mel, &N_enc, &pdim);
    free(mel);
    if (!audio_embeds) { fprintf(stderr,"encoder failed\n"); voxtral_free(ctx); return 5; }
    auto t_enc = std::chrono::steady_clock::now();
    fprintf(stderr, "encoder: %d frames × %d dim  (%.0f ms)\n", N_enc, pdim,
            std::chrono::duration<double,std::milli>(t_enc-t_mel).count());

    // Build prompt: <s> [INST] [BEGIN_AUDIO] <audio_pad>×N [/INST] lang:LANG [TRANSCRIBE]
    // Token IDs from the Tekken tokenizer (hardcoded for the transcription template):
    //   <s>=1  [INST]=3  [BEGIN_AUDIO]=25  audio_pad=24  [/INST]=4
    //   "lang"=9909  ":"=1058  "en"=1262  [TRANSCRIBE]=34
    // For other languages, swap the language token.
    std::vector<int32_t> ids;
    ids.push_back(1); ids.push_back(3); ids.push_back(25);
    for (int i=0;i<N_enc;i++) ids.push_back(24);
    ids.push_back(4); ids.push_back(9909); ids.push_back(1058);
    // Language token — hardcoded for common languages
    if      (lang=="en") ids.push_back(1262);
    else if (lang=="de") ids.push_back(1005);
    else if (lang=="fr") ids.push_back(3288);
    else if (lang=="es") ids.push_back(1050);
    else if (lang=="it") ids.push_back(2285);
    else if (lang=="pt") ids.push_back(10804);
    else if (lang=="nl") ids.push_back(5825);
    else if (lang=="hi") ids.push_back(4979);
    else                  ids.push_back(1262); // fallback to en
    ids.push_back(34);
    int T_prompt = (int)ids.size();
    fprintf(stderr, "prompt: %d tokens (incl. %d audio_pad)\n", T_prompt, N_enc);

    // Embed + splice
    float * text_embeds = voxtral_embed_tokens(ctx, ids.data(), T_prompt);
    if (!text_embeds) { free(audio_embeds); voxtral_free(ctx); return 6; }
    int spliced = 0;
    for (int i=0;i<T_prompt&&spliced<N_enc;i++) {
        if (ids[i]==24) {
            std::memcpy(text_embeds+(size_t)i*pdim, audio_embeds+(size_t)spliced*pdim, pdim*sizeof(float));
            spliced++;
        }
    }
    free(audio_embeds);

    // KV cache + prefill
    if (!voxtral_kv_init(ctx, 4096)) { free(text_embeds); voxtral_free(ctx); return 7; }
    voxtral_kv_reset(ctx);
    auto t_pf0 = std::chrono::steady_clock::now();
    int n_t=0, vocab=0;
    float * logits = voxtral_run_llm_kv(ctx, text_embeds, T_prompt, 0, &n_t, &vocab);
    auto t_pf1 = std::chrono::steady_clock::now();
    if (!logits) { free(text_embeds); voxtral_free(ctx); return 8; }
    free(text_embeds);
    fprintf(stderr, "prefill: %.0f ms\n",
            std::chrono::duration<double,std::milli>(t_pf1-t_pf0).count());

    int next = 0; { float mx=-1e30f; for(int k=0;k<vocab;k++) if(logits[k]>mx){mx=logits[k];next=k;} }
    free(logits);

    // Greedy decode
    const int EOS = 2;
    std::vector<int32_t> gen; gen.push_back(next);
    auto t_dec0 = std::chrono::steady_clock::now();
    int n_past = T_prompt;
    while ((int)gen.size() < max_new && gen.back() != EOS) {
        int32_t last = gen.back();
        float * tail = voxtral_embed_tokens(ctx, &last, 1);
        if (!tail) break;
        float * lg = voxtral_run_llm_kv(ctx, tail, 1, n_past, nullptr, nullptr);
        free(tail); if (!lg) break;
        n_past++;
        int nx=0; float mx=-1e30f;
        for(int k=0;k<vocab;k++) if(lg[k]>mx){mx=lg[k];nx=k;}
        free(lg); gen.push_back(nx);
    }
    auto t_dec1 = std::chrono::steady_clock::now();
    double dec_ms = std::chrono::duration<double,std::milli>(t_dec1-t_dec0).count();
    fprintf(stderr, "decode: %zu tokens in %.0f ms (%.0f ms/token)\n",
            gen.size()-1, dec_ms, dec_ms/std::max((size_t)1,gen.size()-1));

    // Decode tokens to text
    std::string transcript;
    for (auto id : gen) {
        if (id == EOS) break;
        int len = 0;
        const uint8_t * bytes = voxtral_token_text(ctx, id, &len);
        if (bytes && len > 0) transcript.append((const char*)bytes, len);
    }
    auto t_total = std::chrono::steady_clock::now();
    fprintf(stderr, "total: %.0f ms\n",
            std::chrono::duration<double,std::milli>(t_total-t0).count());
    printf("%s\n", transcript.c_str());

    voxtral_free(ctx);
    return 0;
}
