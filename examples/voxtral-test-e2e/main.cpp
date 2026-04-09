// voxtral-test-e2e — end-to-end audio → text test for Voxtral.
// Loads mel from .npy, runs encoder + spliced prompt + LLM decode.
#include "voxtral.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <vector>
#include <string>

static bool load_npy_f32(const std::string & path, std::vector<float> & data, std::vector<int> & shape) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[6]; f.read(magic, 6);
    if (memcmp(magic, "\x93NUMPY", 6) != 0) return false;
    uint8_t maj, mn; f.read((char*)&maj,1); f.read((char*)&mn,1);
    uint32_t hl=0; if(maj==1){uint16_t h;f.read((char*)&h,2);hl=h;}else{f.read((char*)&hl,4);}
    std::string hdr(hl,'\0'); f.read(&hdr[0],hl);
    auto sp=hdr.find("'shape':"); auto lp=hdr.find('(',sp); auto rp=hdr.find(')',lp);
    std::string sh=hdr.substr(lp+1,rp-lp-1); shape.clear();
    size_t i=0;
    while(i<sh.size()){while(i<sh.size()&&(sh[i]==' '||sh[i]==','))i++;if(i>=sh.size())break;
        int v=0;while(i<sh.size()&&sh[i]>='0'&&sh[i]<='9'){v=v*10+(sh[i]-'0');i++;}shape.push_back(v);}
    size_t total=1; for(int s:shape) total*=(size_t)s;
    data.resize(total); f.read((char*)data.data(), total*sizeof(float));
    return (bool)f;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.gguf MEL_NPY_DIR\n", argv[0]);
        return 1;
    }

    // Load mel
    std::vector<float> mel; std::vector<int> mel_shape;
    if (!load_npy_f32(std::string(argv[2]) + "/mel_input.npy", mel, mel_shape)) {
        fprintf(stderr, "cannot load mel\n"); return 2;
    }
    int n_mels = mel_shape.size()==3 ? mel_shape[1] : mel_shape[0];
    int T_mel  = mel_shape.size()==3 ? mel_shape[2] : mel_shape[1];
    fprintf(stderr, "mel: %d × %d\n", n_mels, T_mel);

    // Init
    auto cp = voxtral_context_default_params(); cp.n_threads = 4;
    auto * ctx = voxtral_init_from_file(argv[1], cp);
    if (!ctx) { fprintf(stderr, "init failed\n"); return 3; }

    // Encoder
    int N_enc = 0, pdim = 0;
    fprintf(stderr, "running encoder ...\n");
    float * audio_embeds = voxtral_run_encoder(ctx, mel.data(), n_mels, T_mel, &N_enc, &pdim);
    if (!audio_embeds) { fprintf(stderr, "encoder failed\n"); voxtral_free(ctx); return 4; }
    fprintf(stderr, "encoder: %d frames × %d dim\n", N_enc, pdim);

    // Build prompt: <s> [INST] [BEGIN_AUDIO] <audio_pad>×375 [/INST] lang:en [TRANSCRIBE]
    // Token IDs from the reference dump:
    //   prefix: [1, 3, 25]
    //   375 × 24
    //   suffix: [4, 9909, 1058, 1262, 34]
    std::vector<int32_t> ids;
    ids.push_back(1); ids.push_back(3); ids.push_back(25);
    for (int i = 0; i < N_enc; i++) ids.push_back(24);
    ids.push_back(4); ids.push_back(9909); ids.push_back(1058); ids.push_back(1262); ids.push_back(34);
    int T_prompt = (int)ids.size();
    fprintf(stderr, "prompt: %d tokens (incl. %d audio_pad)\n", T_prompt, N_enc);

    // Embed text tokens
    float * text_embeds = voxtral_embed_tokens(ctx, ids.data(), T_prompt);
    if (!text_embeds) { fprintf(stderr, "embed failed\n"); free(audio_embeds); voxtral_free(ctx); return 5; }

    // Splice audio embeds at audio_pad positions (id=24)
    int spliced = 0;
    for (int i = 0; i < T_prompt && spliced < N_enc; i++) {
        if (ids[i] == 24) {
            std::memcpy(text_embeds + (size_t)i * pdim,
                        audio_embeds + (size_t)spliced * pdim,
                        pdim * sizeof(float));
            spliced++;
        }
    }
    free(audio_embeds);
    fprintf(stderr, "spliced %d audio frames\n", spliced);

    // KV cache + prefill
    if (!voxtral_kv_init(ctx, 4096)) { free(text_embeds); voxtral_free(ctx); return 6; }
    voxtral_kv_reset(ctx);

    int n_t = 0, vocab = 0;
    float * logits = voxtral_run_llm_kv(ctx, text_embeds, T_prompt, 0, &n_t, &vocab);
    if (!logits) { fprintf(stderr, "prefill failed\n"); free(text_embeds); voxtral_free(ctx); return 7; }
    free(text_embeds);
    fprintf(stderr, "prefill done (vocab=%d)\n", vocab);

    // Greedy decode
    const int EOS = 2;  // </s> in Tekken
    const int MAX_NEW = 256;
    std::vector<int32_t> gen;
    { int mx_i = 0; float mx = -1e30f;
      for (int k = 0; k < vocab; k++) if (logits[k] > mx) { mx = logits[k]; mx_i = k; }
      gen.push_back(mx_i); }
    free(logits);
    fprintf(stderr, "step 0: id=%d\n", gen.back());

    int n_past = T_prompt;
    while ((int)gen.size() < MAX_NEW && gen.back() != EOS) {
        int32_t last = gen.back();
        float * tail = voxtral_embed_tokens(ctx, &last, 1);
        if (!tail) break;
        float * lg = voxtral_run_llm_kv(ctx, tail, 1, n_past, nullptr, nullptr);
        free(tail);
        if (!lg) break;
        n_past++;
        int nx = 0; float mx = -1e30f;
        for (int k = 0; k < vocab; k++) if (lg[k] > mx) { mx = lg[k]; nx = k; }
        free(lg);
        gen.push_back(nx);
        if ((int)gen.size() <= 10 || gen.back() == EOS)
            fprintf(stderr, "step %d: id=%d\n", (int)gen.size()-1, gen.back());
    }

    fprintf(stderr, "\ngenerated %zu tokens\n", gen.size());
    // Decode tokens to text using voxtral_token_text
    std::string transcript;
    for (auto id : gen) {
        if (id == EOS) break;
        int len = 0;
        const uint8_t * bytes = voxtral_token_text(ctx, id, &len);
        if (bytes && len > 0) transcript.append((const char*)bytes, len);
    }
    fprintf(stderr, "transcript: %s\n", transcript.c_str());
    printf("%s\n", transcript.c_str());

    voxtral_free(ctx);
    return 0;
}
