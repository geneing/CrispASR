// kokoro.cpp — runtime for hexgrad/Kokoro-82M and yl4579/StyleTTS2-
// LJSpeech (same architecture). The full forward pass at synthesis
// time:
//
//   phonemes (IPA, from espeak-ng or --tts-phonemes) → token IDs
//     ↓
//   text_enc (Embedding → 3× Conv1d k=5 + LN + GELU → bidir LSTM)
//                                                        → t_enc [L, 512]
//   bert (custom ALBERT-base, 12 parameter-shared layers) → bert_dur [L, 512]
//                                                        ↓
//   ref_s = voice_pack[L-1, 0, :]   split [pred 0:128 | dec 128:256]
//                                                        ↓
//   ProsodyPredictor (dur_enc 3× alternating LSTM/AdaLN, post-encoder
//   LSTM, dur_proj, shared LSTM, F0/N AdainResBlk1d stacks) →
//   durations + F0 + N
//                                                        ↓
//   align(t_enc, durations) → en [T_frames, 512]
//                                                        ↓
//   iSTFTNet decoder (encode + 4× decode AdainResBlk1d + asr_res +
//   F0_conv + N_conv → Generator with HnNSF source + 2× upsample +
//   noise injection + 6 resblocks averaged + conv_post → 22 channels)
//                                                        ↓
//   iSTFT on CPU (n_fft=20, hop=5, Hann) → 24 kHz audio
//
// This file is the M1 skeleton: GGUF two-pass load, hparams + vocab,
// voice-pack secondary loader, scheduler setup, and stub C ABI for the
// synth + stage extractors. Subsequent milestones fill in the forward
// pass.

#include "kokoro.h"

#include "core/attention.h"
#include "core/gguf_loader.h"
#include "core/lstm.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

bool env_bool(const char* k) {
    const char* v = std::getenv(k);
    return v && *v && std::strcmp(v, "0") != 0 && std::strcmp(v, "false") != 0;
}

// Hyperparameters read from `kokoro.*` and `kokoro.plbert.*` GGUF KV.
struct kokoro_hp {
    // Top-level
    uint32_t hidden_dim   = 512;
    uint32_t style_dim    = 128;
    uint32_t max_dur      = 50;
    uint32_t n_token      = 178;
    uint32_t n_mels       = 80;  // unused by the synthesis path; kept for completeness
    uint32_t n_layer      = 3;   // duration-encoder depth
    uint32_t text_enc_k   = 5;
    uint32_t sample_rate  = 24000;
    uint32_t vocab_size   = 178;

    // PL-BERT (custom 12-layer parameter-shared ALBERT)
    uint32_t plbert_embd_size  = 128;
    uint32_t plbert_hidden     = 768;
    uint32_t plbert_n_layers   = 12;
    uint32_t plbert_n_heads    = 12;
    uint32_t plbert_ff         = 2048;
    uint32_t plbert_max_pos    = 512;
    uint32_t plbert_vocab_size = 178;

    // iSTFTNet
    uint32_t istft_init_ch  = 512;
    uint32_t istft_n_fft    = 20;
    uint32_t istft_hop      = 5;
    uint32_t istft_n_dilations = 3; // entries per resblock
    std::vector<uint32_t> istft_upsample_rates;        // [10, 6]
    std::vector<uint32_t> istft_upsample_kernel_sizes; // [20, 12]
    std::vector<uint32_t> istft_resblock_kernel_sizes; // [3, 7, 11]
    std::vector<uint32_t> istft_resblock_dilations;    // flat, length 3*n_dilations
};

struct kokoro_vocab {
    std::vector<std::string> id_to_token;                   // size = vocab_size
    std::unordered_map<std::string, int32_t> token_to_id;   // lookup for tokenizer
    int32_t pad_id = 0;                                     // index of "$" (Kokoro/StyleTTS2 pad)
};

struct kokoro_voice_pack {
    std::string name;
    uint32_t max_phonemes = 0;
    uint32_t style_dim = 0;
    ggml_tensor* pack = nullptr;            // (max_phon, 1, 256) F32 — owned by vp_ctx_w/vp_buf_w
    ggml_context* vp_ctx_w = nullptr;
    ggml_backend_buffer_t vp_buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
};

} // namespace

struct kokoro_context {
    kokoro_context_params params{};
    int n_threads = 4;
    std::string espeak_lang = "en-us";

    kokoro_hp    hp;
    kokoro_vocab vocab;

    // Backends. `backend` is the user's choice (Metal/CUDA/CPU); `backend_cpu`
    // is always present. `gen_backend` is forced to CPU when
    // gen_force_metal=false (the default — see header) to avoid a known
    // Metal hang on stride-10 ConvTranspose1d in iSTFTNet's `gen.ups.0`.
    ggml_backend_t backend     = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_t gen_backend = nullptr; // either backend or backend_cpu

    // Schedulers. `sched` runs everything except the iSTFTNet generator;
    // `gen_sched` runs the generator alone (typically CPU-only for the
    // Metal-hang workaround); `bert_sched` is the same as `sched` (BERT
    // is light enough not to need its own scheduler — kept as a named
    // alias so the BERT graph code can stay decoupled).
    ggml_backend_sched_t sched     = nullptr;
    ggml_backend_sched_t gen_sched = nullptr;

    // Primary weights (talker GGUF). Tensors stay resident here for the
    // life of the context.
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;
    std::vector<uint8_t> compute_meta;

    // Voice pack (secondary GGUF).
    kokoro_voice_pack vp;
    bool vp_loaded = false;
};

namespace {

ggml_tensor* try_get(const kokoro_context* c, const char* name) {
    auto it = c->tensors.find(name);
    return it == c->tensors.end() ? nullptr : it->second;
}

ggml_tensor* require(const kokoro_context* c, const char* name) {
    auto* t = try_get(c, name);
    if (!t) {
        fprintf(stderr, "kokoro: required tensor missing: %s\n", name);
    }
    return t;
}

// Read a uint32 array from GGUF metadata into a std::vector. Empty result if
// the key is absent or the array element type is unexpected.
std::vector<uint32_t> kv_u32_array(gguf_context* g, const char* key) {
    std::vector<uint32_t> out;
    const int k = gguf_find_key(g, key);
    if (k < 0)
        return out;
    if (gguf_get_kv_type(g, k) != GGUF_TYPE_ARRAY)
        return out;
    const gguf_type elem = gguf_get_arr_type(g, k);
    const int n = gguf_get_arr_n(g, k);
    out.reserve((size_t)n);
    if (elem == GGUF_TYPE_UINT32 || elem == GGUF_TYPE_INT32) {
        const auto* d = (const uint32_t*)gguf_get_arr_data(g, k);
        out.assign(d, d + n);
    } else if (elem == GGUF_TYPE_UINT64 || elem == GGUF_TYPE_INT64) {
        const auto* d = (const uint64_t*)gguf_get_arr_data(g, k);
        for (int i = 0; i < n; i++)
            out.push_back((uint32_t)d[i]);
    }
    return out;
}

// Sanity-check the loaded weights. Soft-validates a representative
// subset rather than enumerating every tensor (the full list is 459
// names; load_weights already provides the full map).
bool sanity_check_weights(const kokoro_context* c) {
    const char* must_have[] = {
        "bert.embd.tok.weight",
        "bert.embd_proj.weight",
        "bert.attn_q.weight",
        "bert.attn_q.bias",
        "bert.ffn_up.weight",
        "bert.pooler.weight",
        "bert_proj.weight",
        "bert_proj.bias",
        "text_enc.embd.weight",
        "text_enc.cnn.0.conv.weight",
        "text_enc.lstm.weight_ih_l0",
        "text_enc.lstm.weight_ih_l0_reverse",
        "pred.F0_proj.weight",
        "pred.N_proj.weight",
        "dec.gen.conv_post.weight",
    };
    bool ok = true;
    for (const char* name : must_have) {
        if (c->tensors.find(name) == c->tensors.end()) {
            fprintf(stderr, "kokoro: required tensor missing: %s\n", name);
            ok = false;
        }
    }
    return ok;
}

// ---------------------------------------------------------------------------
// M3 — BERT (PL-BERT, parameter-shared 12-layer ALBERT)
//
// Architecture (from kokoro/custom_albert.py + the AlbertConfig defaults
// inherited by PL-BERT):
//   * Embeddings: tok(178→128) + pos(512→128) + tt(2→128, idx 0)  → LN
//   * embd_proj: Linear 128 → 768
//   * 12× shared layers (post-norm):
//       y = LN(x + self_attn(x), attn_ln, eps=1e-12)
//       y = LN(y + GELU_new(ffn_up(y))→ffn_down, ffn_ln, eps=1e-12)
//   * (pooler exists in the GGUF but is unused by the synthesis path —
//     Kokoro reads `last_hidden_state`, not `pooler_output`.)
//   * bert_proj: Linear 768 → 512, applied per-token.
//
// The "bert_pooler_out" stage name in kokoro.h is legacy from the plan;
// the value at this stage is the per-token last_hidden_state of shape
// (768, L), not a pooled vector. Kept for ABI stability.
// ---------------------------------------------------------------------------

static const float kBertLayerNormEps = 1e-12f;

// Build the BERT graph for L tokens. The graph has two int32 inputs
// ("ids" and "positions", both length L) and exposes two outputs by
// name: "bert_pooler_out" (768, L) and "bert_proj_out" (512, L).
static ggml_cgraph* kokoro_build_graph_bert(kokoro_context* c, int L) {
    const auto& hp = c->hp;
    const int D_emb = (int)hp.plbert_embd_size;   // 128
    const int D     = (int)hp.plbert_hidden;      // 768
    const int n_h   = (int)hp.plbert_n_heads;     // 12
    const int n_lay = (int)hp.plbert_n_layers;    // 12 shared
    const int hd    = D / n_h;                    // 64
    const float attn_scale = 1.0f / std::sqrt((float)hd);

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), /*no_alloc=*/true};
    ggml_context* ctx0 = ggml_init(ip);
    // ~30 ops per layer × 12 + embd/proj/pooler = a few hundred. 1024 is plenty.
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 1024, false);

    ggml_tensor* ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, L);
    ggml_set_name(ids, "ids");
    ggml_set_input(ids);

    ggml_tensor* positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, L);
    ggml_set_name(positions, "positions");
    ggml_set_input(positions);

    // ---- Embeddings ----
    ggml_tensor* tok_w  = require(c, "bert.embd.tok.weight");
    ggml_tensor* pos_w  = require(c, "bert.embd.pos.weight");
    ggml_tensor* tt_w   = require(c, "bert.embd.tt.weight");
    ggml_tensor* eln_w  = require(c, "bert.embd.ln.weight");
    ggml_tensor* eln_b  = require(c, "bert.embd.ln.bias");
    ggml_tensor* ep_w   = require(c, "bert.embd_proj.weight");
    ggml_tensor* ep_b   = require(c, "bert.embd_proj.bias");

    ggml_tensor* tok_emb = ggml_get_rows(ctx0, tok_w, ids);          // (D_emb, L) F32
    ggml_tensor* pos_emb = ggml_get_rows(ctx0, pos_w, positions);    // (D_emb, L) F32
    // token_type_ids defaults to zero — slice tt_w[:, 0:1] and broadcast on L.
    ggml_tensor* tt_view = ggml_view_2d(ctx0, tt_w, D_emb, 1, tt_w->nb[1], 0);  // (D_emb, 1) F16
    ggml_tensor* tt_emb  = ggml_cast(ctx0, tt_view, GGML_TYPE_F32);             // (D_emb, 1) F32

    ggml_tensor* x = ggml_add(ctx0, tok_emb, pos_emb);
    x = ggml_add(ctx0, x, tt_emb);

    // Embedding LayerNorm (eps=1e-12).
    x = ggml_norm(ctx0, x, kBertLayerNormEps);
    x = ggml_mul(ctx0, x, eln_w);
    x = ggml_add(ctx0, x, eln_b);

    // embd_proj: Linear D_emb → D
    x = ggml_mul_mat(ctx0, ep_w, x);
    x = ggml_add(ctx0, x, ep_b);                                     // (D, L)

    // ---- 12 shared ALBERT layers ----
    ggml_tensor* q_w  = require(c, "bert.attn_q.weight");
    ggml_tensor* q_b  = require(c, "bert.attn_q.bias");
    ggml_tensor* k_w  = require(c, "bert.attn_k.weight");
    ggml_tensor* k_b  = require(c, "bert.attn_k.bias");
    ggml_tensor* v_w  = require(c, "bert.attn_v.weight");
    ggml_tensor* v_b  = require(c, "bert.attn_v.bias");
    ggml_tensor* o_w  = require(c, "bert.attn_o.weight");
    ggml_tensor* o_b  = require(c, "bert.attn_o.bias");
    ggml_tensor* aln_w = require(c, "bert.attn_ln.weight");
    ggml_tensor* aln_b = require(c, "bert.attn_ln.bias");
    ggml_tensor* fu_w = require(c, "bert.ffn_up.weight");
    ggml_tensor* fu_b = require(c, "bert.ffn_up.bias");
    ggml_tensor* fd_w = require(c, "bert.ffn_down.weight");
    ggml_tensor* fd_b = require(c, "bert.ffn_down.bias");
    ggml_tensor* fln_w = require(c, "bert.ffn_ln.weight");
    ggml_tensor* fln_b = require(c, "bert.ffn_ln.bias");

    core_attn::EncoderSelfAttnParams eap{};
    eap.n_heads     = n_h;
    eap.n_kv_heads  = n_h;       // MHA
    eap.head_dim    = hd;
    eap.n_kv_grp    = 1;
    eap.attn_scale  = attn_scale;
    eap.n_ctx_orig  = 0;
    eap.rope_theta  = 0.0f;
    eap.permute_cont = true;

    for (int il = 0; il < n_lay; il++) {
        // Self-attention with biased Q/K/V/O, no RoPE, no mask.
        ggml_tensor* attn_out = core_attn::encoder_self_attn(
            ctx0, x, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b,
            /*positions*/ nullptr, /*mask*/ nullptr, eap);
        x = ggml_add(ctx0, x, attn_out);
        x = ggml_norm(ctx0, x, kBertLayerNormEps);
        x = ggml_mul(ctx0, x, aln_w);
        x = ggml_add(ctx0, x, aln_b);

        // FFN: up → GELU(tanh approx, "gelu_new") → down (with biases).
        ggml_tensor* ffn = ggml_mul_mat(ctx0, fu_w, x);
        ffn = ggml_add(ctx0, ffn, fu_b);
        ffn = ggml_gelu(ctx0, ffn);                                  // tanh-approx == ALBERT "gelu_new"
        ffn = ggml_mul_mat(ctx0, fd_w, ffn);
        ffn = ggml_add(ctx0, ffn, fd_b);
        x = ggml_add(ctx0, x, ffn);
        x = ggml_norm(ctx0, x, kBertLayerNormEps);
        x = ggml_mul(ctx0, x, fln_w);
        x = ggml_add(ctx0, x, fln_b);
    }

    // Per-token last_hidden_state — exposed for diff-harness as the
    // "bert_pooler_out" stage (legacy name; not actually pooled).
    ggml_tensor* seq = x;                                            // (D, L)
    seq = ggml_cont(ctx0, seq);
    ggml_set_name(seq, "bert_pooler_out");
    ggml_set_output(seq);
    ggml_build_forward_expand(gf, seq);

    // bert_proj: Linear 768 → 512, applied per-token.
    ggml_tensor* bp_w = require(c, "bert_proj.weight");
    ggml_tensor* bp_b = require(c, "bert_proj.bias");
    ggml_tensor* proj = ggml_mul_mat(ctx0, bp_w, seq);               // (512, L)
    proj = ggml_add(ctx0, proj, bp_b);
    ggml_set_name(proj, "bert_proj_out");
    ggml_set_output(proj);
    ggml_build_forward_expand(gf, proj);

    ggml_free(ctx0);
    return gf;
}

// Wrap raw phoneme ids with the StyleTTS2 pad convention:
//   wrapped = [pad_id, raw_ids..., pad_id]
// The reference Python (KModel.forward) does this before feeding into
// BERT / text_enc / predictor. The pad_id is the StyleTTS2 "$" token
// (vocab index 0). All downstream stages see length L+2.
static std::vector<int32_t> kokoro_pad_wrap_ids(const kokoro_context* c, const int32_t* raw, int n_raw) {
    std::vector<int32_t> w;
    w.reserve((size_t)n_raw + 2);
    w.push_back(c->vocab.pad_id);
    w.insert(w.end(), raw, raw + n_raw);
    w.push_back(c->vocab.pad_id);
    return w;
}

// ---------------------------------------------------------------------------
// AdaLayerNorm (used by the predictor's DurationEncoder).
//
// Reference: kokoro/modules.py AdaLayerNorm (eps=1e-5).
//   gamma, beta = chunk(fc(s), 2)        # fc: Linear(style_dim, 2*C)
//   y = (1 + gamma) * LayerNorm(x) + beta
//
// We compute (1 + γ)·x + β as x + x·γ + β to avoid materialising a "1.0"
// tensor for the (1 + γ) term. Mathematically identical, one fewer
// constant input.
//
// Inputs:
//   x      ne = (C, T)  F32, channel-major (LN normalises over ne[0])
//   style  ne = (sty, 1) F32 — predictor reference vector
//   fc_w   ne = (sty, 2C) F16 — Linear weight
//   fc_b   ne = (2C,) F32 — Linear bias
//
// Output: (C, T) F32.
static const float kAdaLnEps = 1e-5f;

static inline ggml_tensor* kokoro_adaln(ggml_context* ctx, ggml_tensor* x, ggml_tensor* style,
                                        ggml_tensor* fc_w, ggml_tensor* fc_b) {
    const int C = (int)x->ne[0];
    // h = fc(s) → (2C, 1)
    ggml_tensor* h = ggml_mul_mat(ctx, fc_w, style);
    h = ggml_add(ctx, h, fc_b);
    // chunk along ne[0]: γ ∈ [0, C), β ∈ [C, 2C).
    const size_t ts = ggml_type_size(GGML_TYPE_F32);
    ggml_tensor* gamma = ggml_view_2d(ctx, h, C, 1, h->nb[1], (size_t)0 * C * ts);
    ggml_tensor* beta  = ggml_view_2d(ctx, h, C, 1, h->nb[1], (size_t)1 * C * ts);

    ggml_tensor* normed = ggml_norm(ctx, x, kAdaLnEps);
    // x*γ + x → broadcast γ (C, 1) over T
    ggml_tensor* x_gamma = ggml_mul(ctx, normed, gamma);
    ggml_tensor* out = ggml_add(ctx, normed, x_gamma);
    out = ggml_add(ctx, out, beta);
    return out;
}

// ---------------------------------------------------------------------------
// M4 — TextEncoder
//
// Reference: kokoro/modules.py TextEncoder.
//   Embedding(178, 512) → 3× Sequential(
//       Conv1d(512, 512, k=5, pad=2),
//       LayerNorm(512)        # over channel dim, ε=1e-5
//       LeakyReLU(0.2),
//       Dropout(0.2)          # no-op at inference
//   )
//   bidirectional LSTM(in=512, hidden=256) → output (B, D=512, T) after
//   the final transpose.
//
// Input: pad-wrapped phoneme ids (length L).
// Output stage `text_enc_out`: ne = (512, L)  F32.
// ---------------------------------------------------------------------------

static const float kTextEncLayerNormEps = 1e-5f;
static const float kTextEncLeakySlope   = 0.2f;

static ggml_cgraph* kokoro_build_graph_text_enc(kokoro_context* c, int L) {
    const auto& hp = c->hp;
    const int D = (int)hp.hidden_dim;       // 512
    const int H_lstm = D / 2;               // 256 (bidir → 2*H = D)
    const int K = (int)hp.text_enc_k;       // 5

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), /*no_alloc=*/true};
    ggml_context* ctx0 = ggml_init(ip);
    // 3 conv blocks + 1 bidir LSTM at modest L → a few thousand nodes max.
    // For L up to plbert_max_pos + 2 = 514, the LSTM dominates: 2 directions
    // * L * ~15 ops/step ≈ 15400 nodes. 32k headroom.
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 32768, false);

    ggml_tensor* ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, L);
    ggml_set_name(ids, "ids");
    ggml_set_input(ids);

    // ---- Embedding: (178, 512) → (D, L) ----
    ggml_tensor* emb_w = require(c, "text_enc.embd.weight");
    ggml_tensor* x = ggml_get_rows(ctx0, emb_w, ids);     // (D, L) F32

    // ---- 3× Conv1d(k=5, pad=2) + LN(channel) + LeakyReLU(0.2) ----
    // Layout flow per block:
    //   in:        (D, L)         ne[0]=D, ne[1]=L
    //   transpose: (L, D)         for ggml_conv_1d which expects (T, C_in)
    //   conv1d:    (L, D, 1)      output is 3D (OL, OC, B=1)
    //   add bias:  bias reshaped (1, D, 1) broadcasts on L and B
    //   reshape:   (L, D)         drop trailing 1
    //   transpose: (D, L)         channel-major for ggml_norm
    //   norm:      (D, L)         normalises over ne[0]=D
    //   * gamma + bias beta:      (D, L)
    //   leaky:     (D, L)
    auto bias_1d = [&](ggml_tensor* b) { return ggml_reshape_3d(ctx0, b, 1, b->ne[0], 1); };

    for (int il = 0; il < 3; il++) {
        char nm[64];
        std::snprintf(nm, sizeof(nm), "text_enc.cnn.%d.conv.weight", il);
        ggml_tensor* cw = require(c, nm);
        std::snprintf(nm, sizeof(nm), "text_enc.cnn.%d.conv.bias", il);
        ggml_tensor* cb = require(c, nm);
        std::snprintf(nm, sizeof(nm), "text_enc.cnn.%d.ln.gamma", il);
        ggml_tensor* lg = require(c, nm);
        std::snprintf(nm, sizeof(nm), "text_enc.cnn.%d.ln.beta", il);
        ggml_tensor* lb = require(c, nm);

        // (D, L) → (L, D) for conv input layout.
        x = ggml_cont(ctx0, ggml_transpose(ctx0, x));     // (L, D)
        x = ggml_conv_1d(ctx0, cw, x, /*s=*/1, /*p=*/(K - 1) / 2, /*d=*/1); // (L, D, 1)
        x = ggml_add(ctx0, x, bias_1d(cb));
        // Drop the trailing batch dim → (L, D).
        x = ggml_reshape_2d(ctx0, x, L, D);

        // (L, D) → (D, L) for channel-major LN.
        x = ggml_cont(ctx0, ggml_transpose(ctx0, x));     // (D, L)
        x = ggml_norm(ctx0, x, kTextEncLayerNormEps);
        x = ggml_mul(ctx0, x, lg);
        x = ggml_add(ctx0, x, lb);
        x = ggml_leaky_relu(ctx0, x, kTextEncLeakySlope, /*inplace=*/false);
    }

    // ---- Bidirectional LSTM ----
    // Input shape requirement: (in, T) = (D=512, L). x is already (D, L). ✓
    ggml_tensor* W_ih_f = require(c, "text_enc.lstm.weight_ih_l0");
    ggml_tensor* W_hh_f = require(c, "text_enc.lstm.weight_hh_l0");
    ggml_tensor* b_ih_f = require(c, "text_enc.lstm.bias_ih_l0");
    ggml_tensor* b_hh_f = require(c, "text_enc.lstm.bias_hh_l0");
    ggml_tensor* W_ih_r = require(c, "text_enc.lstm.weight_ih_l0_reverse");
    ggml_tensor* W_hh_r = require(c, "text_enc.lstm.weight_hh_l0_reverse");
    ggml_tensor* b_ih_r = require(c, "text_enc.lstm.bias_ih_l0_reverse");
    ggml_tensor* b_hh_r = require(c, "text_enc.lstm.bias_hh_l0_reverse");

    ggml_tensor* lstm_out = core_lstm::lstm_bidir(
        ctx0, gf, x,
        W_ih_f, W_hh_f, b_ih_f, b_hh_f,
        W_ih_r, W_hh_r, b_ih_r, b_hh_r,
        H_lstm);  // (2H = D, L)

    ggml_set_name(lstm_out, "text_enc_out");
    ggml_set_output(lstm_out);
    ggml_build_forward_expand(gf, lstm_out);

    ggml_free(ctx0);
    return gf;
}

// Run text_enc and copy the named stage back to a malloc'd float buffer.
// Pad-wraps the raw ids before computing.
static float* kokoro_run_text_enc(kokoro_context* c, const int32_t* raw_ids, int n_raw,
                                  const char* stage_name, int* out_n) {
    if (out_n)
        *out_n = 0;
    if (n_raw <= 0)
        return nullptr;
    std::vector<int32_t> padded = kokoro_pad_wrap_ids(c, raw_ids, n_raw);
    const int L = (int)padded.size();
    if (L > (int)c->hp.plbert_max_pos) {
        fprintf(stderr, "kokoro: text_enc L_padded=%d exceeds max=%u\n",
                L, c->hp.plbert_max_pos);
        return nullptr;
    }

    ggml_cgraph* gf = kokoro_build_graph_text_enc(c, L);
    if (!gf)
        return nullptr;

    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "kokoro: sched_alloc_graph failed for text_enc\n");
        return nullptr;
    }

    ggml_tensor* in_ids = ggml_graph_get_tensor(gf, "ids");
    ggml_backend_tensor_set(in_ids, padded.data(), 0, (size_t)L * sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "kokoro: text_enc graph compute failed\n");
        return nullptr;
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, stage_name);
    if (!out) {
        fprintf(stderr, "kokoro: text_enc graph missing output '%s'\n", stage_name);
        return nullptr;
    }

    const size_t n_floats = (size_t)out->ne[0] * (size_t)out->ne[1];
    float* r = (float*)std::malloc(n_floats * sizeof(float));
    if (!r)
        return nullptr;
    ggml_backend_tensor_get(out, r, 0, n_floats * sizeof(float));
    if (out_n)
        *out_n = (int)n_floats;
    return r;
}

// Run the BERT graph and copy the named stage tensor back to a malloc'd
// float buffer. Returns nullptr on failure. *out_n is set to the total
// number of float elements (D × L_padded) on success. The input `ids`
// are RAW phoneme ids; this function adds the StyleTTS2 pad-wrap before
// computing, so the output corresponds to L+2 tokens.
static float* kokoro_run_bert(kokoro_context* c, const int32_t* raw_ids, int n_raw,
                              const char* stage_name, int* out_n) {
    if (out_n)
        *out_n = 0;
    if (n_raw <= 0) {
        fprintf(stderr, "kokoro: bert needs L > 0 (got %d)\n", n_raw);
        return nullptr;
    }
    std::vector<int32_t> padded = kokoro_pad_wrap_ids(c, raw_ids, n_raw);
    const int L = (int)padded.size();
    if (L > (int)c->hp.plbert_max_pos) {
        fprintf(stderr, "kokoro: bert L_padded=%d exceeds max_position_embeddings=%u\n",
                L, c->hp.plbert_max_pos);
        return nullptr;
    }

    ggml_cgraph* gf = kokoro_build_graph_bert(c, L);
    if (!gf)
        return nullptr;

    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "kokoro: sched_alloc_graph failed for bert\n");
        return nullptr;
    }

    // Fill inputs.
    ggml_tensor* in_ids = ggml_graph_get_tensor(gf, "ids");
    ggml_tensor* in_pos = ggml_graph_get_tensor(gf, "positions");
    ggml_backend_tensor_set(in_ids, padded.data(), 0, (size_t)L * sizeof(int32_t));

    std::vector<int32_t> positions(L);
    for (int i = 0; i < L; i++)
        positions[i] = i;
    ggml_backend_tensor_set(in_pos, positions.data(), 0, (size_t)L * sizeof(int32_t));

    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "kokoro: bert graph compute failed\n");
        return nullptr;
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, stage_name);
    if (!out) {
        fprintf(stderr, "kokoro: bert graph missing output '%s'\n", stage_name);
        return nullptr;
    }

    const size_t n_floats = (size_t)out->ne[0] * (size_t)out->ne[1];
    float* r = (float*)std::malloc(n_floats * sizeof(float));
    if (!r)
        return nullptr;
    ggml_backend_tensor_get(out, r, 0, n_floats * sizeof(float));
    if (out_n)
        *out_n = (int)n_floats;
    return r;
}

// ---------------------------------------------------------------------------
// M6 — Alignment (CPU)
//
// Reference (model.py:110-114):
//   indices = repeat_interleave(arange(L), pred_dur)
//   aln[indices, arange(T_frames)] = 1
//   en = d.transpose(-1, -2) @ aln       # (640, T_frames)
//
// The matmul is just a column-repeat: en[:, j] = d[:, indices[j]]. We
// implement that directly to avoid materialising the (L, T_frames) one-hot
// alignment matrix.
//
// features:  (D, L) F32 input
// durations: (L,)   integer counts
// Returns malloc'd (D, T_frames) F32 buffer with T_frames = sum(durations).
// ---------------------------------------------------------------------------

static float* kokoro_align_repeat(const float* features, int D, int L, const int* durations, int* out_T_frames) {
    int T = 0;
    for (int i = 0; i < L; i++)
        T += durations[i];
    if (out_T_frames) *out_T_frames = T;
    if (T <= 0) return nullptr;
    float* out = (float*)std::malloc((size_t)D * T * sizeof(float));
    if (!out) return nullptr;
    int j = 0;
    for (int i = 0; i < L; i++) {
        const int n = durations[i];
        const float* col = features + (size_t)i * D;
        for (int k = 0; k < n; k++) {
            std::memcpy(out + (size_t)j * D, col, (size_t)D * sizeof(float));
            j++;
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// AdaIN1d (used by AdainResBlk1d in M5b predictor F0/N stacks and M7
// decoder).
//
// Reference: kokoro/istftnet.py AdaIN1d.
//   InstanceNorm1d(C, affine=True) but the affine γ'/β' were *not* trained
//   meaningfully (kokoro author note), so the converter dropped them. We
//   only have fc.weight (sty, 2C) and fc.bias (2C,).
//
// Math:
//   gamma, beta = chunk(fc(s), 2)
//   y = (1 + gamma) * InstanceNorm1d(x) + beta
//
// InstanceNorm1d on (C, T) layout: per-channel mean/var across T. ggml_norm
// normalises along ne[0]; with (T, C) layout that's per-channel-along-T,
// which is exactly instance norm 1D. So we transpose, norm, transpose back.
// ---------------------------------------------------------------------------

static const float kAdaIn1dEps = 1e-5f;   // PyTorch InstanceNorm1d default

static inline ggml_tensor* kokoro_adain1d(ggml_context* ctx, ggml_tensor* x, ggml_tensor* style,
                                          ggml_tensor* fc_w, ggml_tensor* fc_b) {
    const int C = (int)x->ne[0];

    // Instance norm: transpose (C, T) → (T, C); ggml_norm normalises along ne[0]=T
    // per other dim ⇒ per-channel mean+var along T; transpose back.
    ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, x));
    xt = ggml_norm(ctx, xt, kAdaIn1dEps);
    ggml_tensor* normed = ggml_cont(ctx, ggml_transpose(ctx, xt));   // (C, T)

    // gamma, beta from fc(s).
    ggml_tensor* h = ggml_mul_mat(ctx, fc_w, style);
    h = ggml_add(ctx, h, fc_b);
    const size_t ts = ggml_type_size(GGML_TYPE_F32);
    ggml_tensor* gamma = ggml_view_2d(ctx, h, C, 1, h->nb[1], (size_t)0 * C * ts);
    ggml_tensor* beta  = ggml_view_2d(ctx, h, C, 1, h->nb[1], (size_t)1 * C * ts);

    // (1 + γ) * normed + β  →  normed + normed*γ + β  (saves the "1" tensor).
    ggml_tensor* x_gamma = ggml_mul(ctx, normed, gamma);
    ggml_tensor* out = ggml_add(ctx, normed, x_gamma);
    out = ggml_add(ctx, out, beta);
    return out;
}

// ---------------------------------------------------------------------------
// Depthwise ConvTranspose1d for the AdainResBlk1d pool layer.
//
// Specific shape used by the predictor's F0/N pool: kernel (3, 1, C),
// stride 2, padding 1, output_padding 1. Output length = 2*T (per
// ConvTranspose1d formula). ggml has no `groups` argument on
// conv_transpose_1d, so we open-code the math.
//
// With s=2, k=3, p=1, op=1 the output collapses to:
//   y[c, 2t]   = w[c, 1] * x[c, t]
//   y[c, 2t+1] = w[c, 0] * x[c, t] + w[c, 2] * x[c, t+1]    (x[c, T]=0 boundary)
// Plus per-channel bias.
// ---------------------------------------------------------------------------

static inline ggml_tensor* kokoro_pool_2x_depthwise(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w_kernel,
                                                   ggml_tensor* w_bias) {
    const int C = (int)x->ne[0];
    const int T = (int)x->ne[1];

    // w_kernel ne=(K=3, 1, C) F16. Permute to (C, 3, 1), cast to F32 (so the
    // element-wise muls below work — F32*F16 mul is not portable across
    // backends), reshape to (C, 3), then take 3 column views.
    ggml_tensor* w_perm = ggml_cont(ctx, ggml_permute(ctx, w_kernel, 2, 0, 1, 3));   // (C, 3, 1) F16
    ggml_tensor* w_perm_f32 = ggml_cast(ctx, w_perm, GGML_TYPE_F32);
    ggml_tensor* w_2d = ggml_reshape_2d(ctx, w_perm_f32, C, 3);                      // (C, 3) F32
    const size_t row_b = w_2d->nb[1];
    ggml_tensor* w0 = ggml_view_2d(ctx, w_2d, C, 1, row_b, (size_t)0 * row_b);
    ggml_tensor* w1 = ggml_view_2d(ctx, w_2d, C, 1, row_b, (size_t)1 * row_b);
    ggml_tensor* w2 = ggml_view_2d(ctx, w_2d, C, 1, row_b, (size_t)2 * row_b);

    // x_shifted[c, t] = x[c, t+1] for t < T-1, 0 for t = T-1.
    // Take x[:, 1:] (C, T-1) and zero-pad on the right to (C, T).
    ggml_tensor* x_tail = ggml_view_2d(ctx, x, C, T - 1, x->nb[1], x->nb[1]);        // (C, T-1)
    x_tail = ggml_cont(ctx, x_tail);                                                  // contiguous
    ggml_tensor* x_shifted = ggml_pad_ext(ctx, x_tail, 0, 0, 0, 1, 0, 0, 0, 0);       // (C, T)

    // y_even (C, T) = w1 ⊙ x  (broadcast w1 over T)
    ggml_tensor* y_even = ggml_mul(ctx, x, w1);
    // y_odd (C, T) = w0 ⊙ x + w2 ⊙ x_shifted
    ggml_tensor* y_odd = ggml_add(ctx, ggml_mul(ctx, x, w0), ggml_mul(ctx, x_shifted, w2));

    // Interleave: reshape both to (C, 1, T), concat dim=1 → (C, 2, T),
    // reshape to (C, 2*T). Memory layout means consecutive time positions
    // alternate even/odd, which is the desired interleaving.
    ggml_tensor* even_3d = ggml_reshape_3d(ctx, y_even, C, 1, T);
    ggml_tensor* odd_3d  = ggml_reshape_3d(ctx, y_odd,  C, 1, T);
    ggml_tensor* stacked = ggml_concat(ctx, even_3d, odd_3d, /*dim=*/1);              // (C, 2, T)
    ggml_tensor* y = ggml_cont(ctx, ggml_reshape_2d(ctx, stacked, C, 2 * T));         // (C, 2T)

    // Per-channel bias (broadcast over time).
    if (w_bias)
        y = ggml_add(ctx, y, w_bias);
    return y;
}

// ---------------------------------------------------------------------------
// AdainResBlk1d (predictor F0/N stacks).
//
// Reference: kokoro/istftnet.py AdainResBlk1d (note: M7 decoder uses a
// *different* class AdaINResBlock1 which has Snake-α; this one uses
// LeakyReLU(0.2)).
//
// Forward:
//   residual = x
//   x' = norm1(residual)              # AdaIN1d on dim_in
//   x' = LeakyReLU(0.2)(x')
//   x' = pool(x')                     # ConvTranspose1d depthwise 2× if upsample, else identity
//   x' = conv1(x')                    # Conv1d(dim_in→dim_out, k=3, pad=1)
//   x' = norm2(x')                    # AdaIN1d on dim_out
//   x' = LeakyReLU(0.2)(x')
//   x' = conv2(x')                    # Conv1d(dim_out→dim_out, k=3, pad=1)
//   shortcut = upsample(residual)     # F.interpolate(2x, nearest) if upsample
//   shortcut = conv1x1(shortcut)      # if learned_sc (dim_in != dim_out)
//   out = (x' + shortcut) / sqrt(2)
//
// Pass nullptr for `pool_w/pool_b` to skip pool (upsample=False blocks),
// and nullptr for `conv1x1_w` to skip the learned shortcut conv (when
// dim_in == dim_out).
// ---------------------------------------------------------------------------

static const float kAdainLeakySlope = 0.2f;

static inline ggml_tensor* kokoro_adain_resblk(ggml_context* ctx, ggml_tensor* x, ggml_tensor* style,
                                              ggml_tensor* adain1_w, ggml_tensor* adain1_b,
                                              ggml_tensor* adain2_w, ggml_tensor* adain2_b,
                                              ggml_tensor* conv1_w, ggml_tensor* conv1_b,
                                              ggml_tensor* conv2_w, ggml_tensor* conv2_b,
                                              ggml_tensor* pool_w, ggml_tensor* pool_b,
                                              ggml_tensor* conv1x1_w) {
    const bool upsample = (pool_w != nullptr);
    const int dim_in = (int)x->ne[0];
    (void)dim_in;

    // ---- Residual path ----
    ggml_tensor* xt = kokoro_adain1d(ctx, x, style, adain1_w, adain1_b);              // (Cin, T)
    xt = ggml_leaky_relu(ctx, xt, kAdainLeakySlope, /*inplace=*/false);

    if (upsample) {
        xt = kokoro_pool_2x_depthwise(ctx, xt, pool_w, pool_b);                       // (Cin, 2T)
    }

    // conv1: Conv1d(dim_in → dim_out, k=3, pad=1). Layout flow:
    //   (Cin, T') → transpose → (T', Cin) → ggml_conv_1d → (T', Cout, 1) → squeeze → (T', Cout) → transpose → (Cout, T')
    auto conv_k3 = [&](ggml_tensor* in, ggml_tensor* w, ggml_tensor* b) -> ggml_tensor* {
        const int Tin = (int)in->ne[1];
        ggml_tensor* y = ggml_cont(ctx, ggml_transpose(ctx, in));                     // (T, Cin)
        y = ggml_conv_1d(ctx, w, y, /*s*/ 1, /*p*/ 1, /*d*/ 1);                       // (T, Cout, 1)
        // Add bias broadcast: bias ne=(Cout,) → reshape (1, Cout, 1)
        ggml_tensor* b3 = ggml_reshape_3d(ctx, b, 1, b->ne[0], 1);
        y = ggml_add(ctx, y, b3);
        const int Cout = (int)w->ne[2];
        y = ggml_reshape_2d(ctx, y, Tin, Cout);                                       // (T, Cout)
        return ggml_cont(ctx, ggml_transpose(ctx, y));                                // (Cout, T)
    };

    xt = conv_k3(xt, conv1_w, conv1_b);                                               // (Cout, T')
    xt = kokoro_adain1d(ctx, xt, style, adain2_w, adain2_b);                          // (Cout, T')
    xt = ggml_leaky_relu(ctx, xt, kAdainLeakySlope, /*inplace=*/false);
    xt = conv_k3(xt, conv2_w, conv2_b);                                               // (Cout, T')

    // ---- Shortcut path ----
    ggml_tensor* sc = x;
    if (upsample) {
        // F.interpolate(scale=2, mode='nearest'): each input column is duplicated.
        // Equivalent to (Cin, 1, T) → concat with itself on a new dim → (Cin, 2, T) → reshape (Cin, 2T).
        ggml_tensor* sc_3d = ggml_reshape_3d(ctx, sc, sc->ne[0], 1, sc->ne[1]);
        ggml_tensor* sc_dup = ggml_concat(ctx, sc_3d, sc_3d, /*dim=*/1);              // (Cin, 2, T)
        sc = ggml_cont(ctx, ggml_reshape_2d(ctx, sc_dup, sc->ne[0], 2 * (int)sc->ne[1]));
    }
    if (conv1x1_w) {
        // Conv1d k=1, no bias. (Cin, T') → transpose → (T', Cin) → conv → (T', Cout, 1) → reshape → transpose
        const int Tin = (int)sc->ne[1];
        ggml_tensor* sct = ggml_cont(ctx, ggml_transpose(ctx, sc));                   // (T', Cin)
        sct = ggml_conv_1d(ctx, conv1x1_w, sct, /*s*/ 1, /*p*/ 0, /*d*/ 1);           // (T', Cout, 1)
        const int Cout = (int)conv1x1_w->ne[2];
        sct = ggml_reshape_2d(ctx, sct, Tin, Cout);                                   // (T', Cout)
        sc = ggml_cont(ctx, ggml_transpose(ctx, sct));                                // (Cout, T')
    }

    // out = (xt + sc) / sqrt(2)
    ggml_tensor* sum = ggml_add(ctx, xt, sc);
    return ggml_scale(ctx, sum, 1.0f / std::sqrt(2.0f));
}

// ---------------------------------------------------------------------------
// M5a — ProsodyPredictor (dur_enc + pred.lstm + dur_proj → durations)
//
// Reference: kokoro/modules.py ProsodyPredictor + DurationEncoder.
//
// Voice pack split (corrects the briefing — see model.py:104, 118):
//   ref_s[:, :128]   → DECODER style (used in M7)
//   ref_s[:, 128:]   → PREDICTOR style (used here)
//
// Forward at synth time:
//   bert_proj_out  (D=512, L)         from M3
//   style_pred     (sty=128, 1)        from voice.pack[idx, 0, 128:256]
//
//   x = cat([bert_proj_out, style_pred.broadcast(L)], axis=0)   (640, L)
//   for il in 0..2:
//       x = bidir_LSTM(x)                                        (512, L)
//       x = AdaLN(x, style_pred)                                 (512, L)
//       x = cat([x, style_pred.broadcast(L)], axis=0)            (640, L)
//   dur_enc_out = x                                              (640, L)
//
//   y = bidir_LSTM_pred(x)                                       (512, L)
//   z = Linear_dur_proj(y)                                       (50, L)
//   durations_pre = sum_rows(sigmoid(z))                         (1, L)
//   # ↑ runtime-side: round + clamp(min=1) → int durations[L]
// ---------------------------------------------------------------------------

// Pull the per-voice predictor style vector from the loaded voice pack.
// voice.pack ne = (256, 1, 510) per the converter; index along ne[2] by
// (clamp(L_raw, 1, max_phon) - 1) to get a (256, 1) slice, then take
// the back half (offset 128*sizeof(F32)) for the predictor side.
//
// We bake the index into the graph (graph is rebuilt per-utterance
// anyway), so this returns a graph-time view of the voice-pack tensor.
static ggml_tensor* kokoro_voice_style_slice(ggml_context* ctx, kokoro_context* c, int L_raw, int byte_offset) {
    ggml_tensor* pack = c->vp.pack;
    const uint32_t max_phon = (uint32_t)pack->ne[2];
    int idx = L_raw;
    if (idx < 1) idx = 1;
    if (idx > (int)max_phon) {
        fprintf(stderr, "kokoro: L_raw=%d clamped to max_phon=%u for voice pack\n", L_raw, max_phon);
        idx = (int)max_phon;
    }
    idx -= 1;
    ggml_tensor* slice = ggml_view_2d(ctx, pack, /*ne0*/ 256, /*ne1*/ 1, pack->nb[1],
                                      (size_t)idx * pack->nb[2]);
    return ggml_view_2d(ctx, slice, /*ne0*/ 128, /*ne1*/ 1, slice->nb[1], (size_t)byte_offset);
}

// Predictor style: back half (offset 128*F32). model.py:104 → ref_s[:, 128:]
static ggml_tensor* kokoro_voice_style_pred_view(ggml_context* ctx, kokoro_context* c, int L_raw) {
    return kokoro_voice_style_slice(ctx, c, L_raw, 128 * (int)sizeof(float));
}

// Decoder style: front half (offset 0). model.py:118 → ref_s[:, :128]
static ggml_tensor* kokoro_voice_style_dec_view(ggml_context* ctx, kokoro_context* c, int L_raw) {
    return kokoro_voice_style_slice(ctx, c, L_raw, 0);
}

static ggml_cgraph* kokoro_build_graph_predictor(kokoro_context* c, int L, int L_raw) {
    const auto& hp = c->hp;
    const int D     = (int)hp.hidden_dim;       // 512
    const int Hsty  = (int)hp.style_dim;        // 128
    const int H_lstm = D / 2;                    // 256

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), /*no_alloc=*/true};
    ggml_context* ctx0 = ggml_init(ip);
    // 4 bidir LSTMs at small L (≤ ~520) + ~10 cat/AdaLN ops.
    // Per LSTM: 2 dirs × L × ~14 ops ≈ 14600 nodes at L=520. 4 LSTMs ≈ 60k.
    // 65536 buffer is generous but well within the 262144 sched ceiling.
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 65536, false);

    // ---- Inputs ----
    // bert_dur is the per-token output of `bert_proj` (M3 stage `bert_proj_out`),
    // re-supplied as a graph input. Layout (D, L_padded) F32.
    ggml_tensor* bert_dur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, D, L);
    ggml_set_name(bert_dur, "bert_dur");
    ggml_set_input(bert_dur);

    // ---- Style: bake voice-pack slice into the graph ----
    // s_pred is F16 (the voice pack is loaded F32 — wait, voice.pack is F32
    // per the GGUF dump — so style is F32 and ggml_mul_mat with the AdaLN
    // fc_w (F16) works fine).
    ggml_tensor* style_pred = kokoro_voice_style_pred_view(ctx0, c, L_raw);  // (128, 1) F32

    // Broadcast style to (Hsty, L) for the cat operations.
    ggml_tensor* s_template = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, Hsty, L);
    ggml_tensor* s_broad = ggml_repeat(ctx0, style_pred, s_template);  // (128, L)

    // Initial cat([bert_dur, s_broad], dim=0) → (D + Hsty, L) = (640, L)
    ggml_tensor* x = ggml_concat(ctx0, bert_dur, s_broad, /*dim=*/0);

    // ---- 3× alternating bidir-LSTM + AdaLN + cat-with-style ----
    for (int il = 0; il < 3; il++) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "pred.dur_enc.%d.lstm.weight_ih_l0", il);
        ggml_tensor* W_ih_f = require(c, buf);
        std::snprintf(buf, sizeof(buf), "pred.dur_enc.%d.lstm.weight_hh_l0", il);
        ggml_tensor* W_hh_f = require(c, buf);
        std::snprintf(buf, sizeof(buf), "pred.dur_enc.%d.lstm.bias_ih_l0", il);
        ggml_tensor* b_ih_f = require(c, buf);
        std::snprintf(buf, sizeof(buf), "pred.dur_enc.%d.lstm.bias_hh_l0", il);
        ggml_tensor* b_hh_f = require(c, buf);
        std::snprintf(buf, sizeof(buf), "pred.dur_enc.%d.lstm.weight_ih_l0_reverse", il);
        ggml_tensor* W_ih_r = require(c, buf);
        std::snprintf(buf, sizeof(buf), "pred.dur_enc.%d.lstm.weight_hh_l0_reverse", il);
        ggml_tensor* W_hh_r = require(c, buf);
        std::snprintf(buf, sizeof(buf), "pred.dur_enc.%d.lstm.bias_ih_l0_reverse", il);
        ggml_tensor* b_ih_r = require(c, buf);
        std::snprintf(buf, sizeof(buf), "pred.dur_enc.%d.lstm.bias_hh_l0_reverse", il);
        ggml_tensor* b_hh_r = require(c, buf);

        x = core_lstm::lstm_bidir(ctx0, gf, x, W_ih_f, W_hh_f, b_ih_f, b_hh_f, W_ih_r, W_hh_r, b_ih_r, b_hh_r,
                                  H_lstm);  // (512, L)

        std::snprintf(buf, sizeof(buf), "pred.dur_enc.%d.adaln.weight", il);
        ggml_tensor* fc_w = require(c, buf);
        std::snprintf(buf, sizeof(buf), "pred.dur_enc.%d.adaln.bias", il);
        ggml_tensor* fc_b = require(c, buf);
        x = kokoro_adaln(ctx0, x, style_pred, fc_w, fc_b);                    // (512, L)

        x = ggml_concat(ctx0, x, s_broad, /*dim=*/0);                          // (640, L)
    }

    // dur_enc_out — the (640, L) input to pred.lstm and to alignment.
    ggml_tensor* dur_enc_out = ggml_cont(ctx0, x);
    ggml_set_name(dur_enc_out, "dur_enc_out");
    ggml_set_output(dur_enc_out);
    ggml_build_forward_expand(gf, dur_enc_out);

    // ---- pred.lstm: (640, L) → (512, L) ----
    ggml_tensor* pl_W_ih_f = require(c, "pred.lstm.weight_ih_l0");
    ggml_tensor* pl_W_hh_f = require(c, "pred.lstm.weight_hh_l0");
    ggml_tensor* pl_b_ih_f = require(c, "pred.lstm.bias_ih_l0");
    ggml_tensor* pl_b_hh_f = require(c, "pred.lstm.bias_hh_l0");
    ggml_tensor* pl_W_ih_r = require(c, "pred.lstm.weight_ih_l0_reverse");
    ggml_tensor* pl_W_hh_r = require(c, "pred.lstm.weight_hh_l0_reverse");
    ggml_tensor* pl_b_ih_r = require(c, "pred.lstm.bias_ih_l0_reverse");
    ggml_tensor* pl_b_hh_r = require(c, "pred.lstm.bias_hh_l0_reverse");

    ggml_tensor* pl_out = core_lstm::lstm_bidir(ctx0, gf, dur_enc_out, pl_W_ih_f, pl_W_hh_f, pl_b_ih_f, pl_b_hh_f,
                                                 pl_W_ih_r, pl_W_hh_r, pl_b_ih_r, pl_b_hh_r, H_lstm);  // (512, L)
    ggml_set_name(pl_out, "pred_lstm_out");
    ggml_set_output(pl_out);

    // ---- pred.dur_proj: Linear(512 → 50) ----
    ggml_tensor* dp_w = require(c, "pred.dur_proj.weight");                    // ne=(512, 50)
    ggml_tensor* dp_b = require(c, "pred.dur_proj.bias");                      // ne=(50,)
    ggml_tensor* dp = ggml_mul_mat(ctx0, dp_w, pl_out);                        // (50, L)
    dp = ggml_add(ctx0, dp, dp_b);
    dp = ggml_sigmoid(ctx0, dp);
    // sum over the 50 dim (ne[0]) → (1, L). Runtime does the round+clamp.
    ggml_tensor* dur_pre = ggml_sum_rows(ctx0, dp);                            // (1, L)
    ggml_set_name(dur_pre, "durations_pre");
    ggml_set_output(dur_pre);
    ggml_build_forward_expand(gf, dur_pre);

    ggml_free(ctx0);
    return gf;
}

// Run the predictor graph and return the named stage as malloc'd float[].
// Stage handling:
//   "dur_enc_out"  → (640, L) raw output
//   "durations"    → (L,) post-round, post-clamp(min=1), cast to float
static float* kokoro_run_predictor(kokoro_context* c, const int32_t* raw_ids, int n_raw,
                                   const char* stage_name, int* out_n) {
    if (out_n)
        *out_n = 0;
    if (!c->vp_loaded) {
        fprintf(stderr, "kokoro: predictor needs voice pack — call kokoro_load_voice_pack first\n");
        return nullptr;
    }
    if (n_raw <= 0)
        return nullptr;

    // 1. Run BERT to get bert_proj_out.
    int n_bp = 0;
    float* bp = kokoro_run_bert(c, raw_ids, n_raw, "bert_proj_out", &n_bp);
    if (!bp)
        return nullptr;
    const int D = (int)c->hp.hidden_dim;
    const int L = n_bp / D;
    if (L * D != n_bp) {
        fprintf(stderr, "kokoro: bert_proj_out size mismatch: %d not divisible by %d\n", n_bp, D);
        std::free(bp);
        return nullptr;
    }

    // 2. Build + run predictor graph.
    ggml_cgraph* gf = kokoro_build_graph_predictor(c, L, n_raw);
    if (!gf) {
        std::free(bp);
        return nullptr;
    }
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "kokoro: sched_alloc_graph failed for predictor\n");
        std::free(bp);
        return nullptr;
    }
    ggml_tensor* in_bd = ggml_graph_get_tensor(gf, "bert_dur");
    ggml_backend_tensor_set(in_bd, bp, 0, (size_t)n_bp * sizeof(float));
    std::free(bp);

    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "kokoro: predictor graph compute failed\n");
        return nullptr;
    }

    if (std::strcmp(stage_name, "dur_enc_out") == 0) {
        ggml_tensor* out = ggml_graph_get_tensor(gf, "dur_enc_out");
        const size_t n_floats = (size_t)out->ne[0] * (size_t)out->ne[1];
        float* r = (float*)std::malloc(n_floats * sizeof(float));
        if (!r) return nullptr;
        ggml_backend_tensor_get(out, r, 0, n_floats * sizeof(float));
        if (out_n) *out_n = (int)n_floats;
        return r;
    }
    if (std::strcmp(stage_name, "durations") == 0) {
        ggml_tensor* out = ggml_graph_get_tensor(gf, "durations_pre");
        // out ne = (1, L) F32
        std::vector<float> raw_dur((size_t)L);
        ggml_backend_tensor_get(out, raw_dur.data(), 0, (size_t)L * sizeof(float));
        // Banker's rounding to match PyTorch's torch.round (R5).
        // fesetround(FE_TONEAREST) is already set globally for nearbyintf
        // to use round-half-to-even; we set it once at process init below.
        float* r = (float*)std::malloc((size_t)L * sizeof(float));
        if (!r) return nullptr;
        for (int i = 0; i < L; i++) {
            float v = std::nearbyintf(raw_dur[i]);
            if (v < 1.0f) v = 1.0f;
            r[i] = v;
        }
        if (out_n) *out_n = L;
        return r;
    }
    if (std::strcmp(stage_name, "pred_lstm_out") == 0) {
        ggml_tensor* out = ggml_graph_get_tensor(gf, "pred_lstm_out");
        const size_t n_floats = (size_t)out->ne[0] * (size_t)out->ne[1];
        float* r = (float*)std::malloc(n_floats * sizeof(float));
        if (!r) return nullptr;
        ggml_backend_tensor_get(out, r, 0, n_floats * sizeof(float));
        if (out_n) *out_n = (int)n_floats;
        return r;
    }
    fprintf(stderr, "kokoro: unknown predictor stage '%s'\n", stage_name);
    return nullptr;
}

// ---------------------------------------------------------------------------
// M5b — F0Ntrain (pred.shared LSTM + F0/N AdainResBlk1d × 3 + F0/N_proj)
//
// Reference: kokoro/modules.py ProsodyPredictor.F0Ntrain.
//   x, _ = self.shared(en.transpose(-1,-2))   # bidir LSTM 640 → 512
//   F0 = x.transpose(-1,-2)                   # (B, 512, T)
//   for block in self.F0:                     # 3 AdainResBlk1d:
//       F0 = block(F0, s)                     # F0[1] upsamples 2×
//   F0 = self.F0_proj(F0)                     # Conv1d 256 → 1, k=1
//   # N mirrors F0 with separate weights
//
// Output dims after the F0 stack:
//   F0[0]: (512, T) → (512, T)
//   F0[1]: (512, T) → (256, 2T)   [upsample=True, dim_in=512, dim_out=256]
//   F0[2]: (256, 2T) → (256, 2T)
//   F0_proj: (256, 2T) → (1, 2T)
//
// Final F0 / N curves are 1D length 2*T_frames.
// ---------------------------------------------------------------------------

static ggml_cgraph* kokoro_build_graph_f0n(kokoro_context* c, int T_frames, int L_raw) {
    const auto& hp = c->hp;
    const int D     = (int)hp.hidden_dim;       // 512
    const int H_lstm = D / 2;                    // 256

    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), /*no_alloc=*/true};
    ggml_context* ctx0 = ggml_init(ip);
    // shared LSTM at T_frames (~60-1500) is the dominant cost: 2 dirs × T × ~14 ops.
    // 6 AdainResBlk1d (3 F0 + 3 N), each ~25 ops. 1024-node margin.
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 65536, false);

    // ---- Input ----
    ggml_tensor* en = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, /*ne0=D+sty=*/640, T_frames);
    ggml_set_name(en, "en");
    ggml_set_input(en);

    // ---- Style: bake voice-pack predictor slice ----
    ggml_tensor* style_pred = kokoro_voice_style_pred_view(ctx0, c, L_raw);          // (128, 1) F32

    // ---- pred.shared bidir LSTM (640 → 512) ----
    ggml_tensor* sh_W_ih_f = require(c, "pred.shared.weight_ih_l0");
    ggml_tensor* sh_W_hh_f = require(c, "pred.shared.weight_hh_l0");
    ggml_tensor* sh_b_ih_f = require(c, "pred.shared.bias_ih_l0");
    ggml_tensor* sh_b_hh_f = require(c, "pred.shared.bias_hh_l0");
    ggml_tensor* sh_W_ih_r = require(c, "pred.shared.weight_ih_l0_reverse");
    ggml_tensor* sh_W_hh_r = require(c, "pred.shared.weight_hh_l0_reverse");
    ggml_tensor* sh_b_ih_r = require(c, "pred.shared.bias_ih_l0_reverse");
    ggml_tensor* sh_b_hh_r = require(c, "pred.shared.bias_hh_l0_reverse");

    ggml_tensor* shared_out = core_lstm::lstm_bidir(ctx0, gf, en, sh_W_ih_f, sh_W_hh_f, sh_b_ih_f, sh_b_hh_f,
                                                     sh_W_ih_r, sh_W_hh_r, sh_b_ih_r, sh_b_hh_r,
                                                     H_lstm);                        // (512, T_frames)

    // Helper to load AdainResBlk1d weights for prefix "pred.X.{idx}".
    auto load_resblk = [&](const char* prefix, int idx, bool has_pool) {
        struct W {
            ggml_tensor* a1w; ggml_tensor* a1b;
            ggml_tensor* a2w; ggml_tensor* a2b;
            ggml_tensor* c1w; ggml_tensor* c1b;
            ggml_tensor* c2w; ggml_tensor* c2b;
            ggml_tensor* poolw; ggml_tensor* poolb;
            ggml_tensor* sc;
        } w;
        char buf[96];
        std::snprintf(buf, sizeof(buf), "%s.%d.adain1.weight", prefix, idx); w.a1w = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.%d.adain1.bias",   prefix, idx); w.a1b = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.%d.adain2.weight", prefix, idx); w.a2w = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.%d.adain2.bias",   prefix, idx); w.a2b = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.%d.conv1.weight",  prefix, idx); w.c1w = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.%d.conv1.bias",    prefix, idx); w.c1b = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.%d.conv2.weight",  prefix, idx); w.c2w = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.%d.conv2.bias",    prefix, idx); w.c2b = require(c, buf);
        if (has_pool) {
            std::snprintf(buf, sizeof(buf), "%s.%d.pool.weight", prefix, idx); w.poolw = require(c, buf);
            std::snprintf(buf, sizeof(buf), "%s.%d.pool.bias",   prefix, idx); w.poolb = require(c, buf);
            std::snprintf(buf, sizeof(buf), "%s.%d.conv1x1.weight", prefix, idx); w.sc = require(c, buf);
        } else {
            w.poolw = w.poolb = w.sc = nullptr;
        }
        return w;
    };

    auto run_stack = [&](const char* prefix, ggml_tensor* in) -> ggml_tensor* {
        // F0/N stacks all share: idx 0 (no upsample, dim 512→512), idx 1 (upsample, 512→256), idx 2 (no upsample, 256→256).
        ggml_tensor* y = in;
        auto w0 = load_resblk(prefix, 0, /*has_pool=*/false);
        auto w1 = load_resblk(prefix, 1, /*has_pool=*/true);
        auto w2 = load_resblk(prefix, 2, /*has_pool=*/false);
        y = kokoro_adain_resblk(ctx0, y, style_pred,
                                w0.a1w, w0.a1b, w0.a2w, w0.a2b,
                                w0.c1w, w0.c1b, w0.c2w, w0.c2b,
                                /*pool*/ nullptr, nullptr, /*conv1x1*/ nullptr);
        y = kokoro_adain_resblk(ctx0, y, style_pred,
                                w1.a1w, w1.a1b, w1.a2w, w1.a2b,
                                w1.c1w, w1.c1b, w1.c2w, w1.c2b,
                                w1.poolw, w1.poolb, w1.sc);
        y = kokoro_adain_resblk(ctx0, y, style_pred,
                                w2.a1w, w2.a1b, w2.a2w, w2.a2b,
                                w2.c1w, w2.c1b, w2.c2w, w2.c2b,
                                /*pool*/ nullptr, nullptr, /*conv1x1*/ nullptr);
        return y;   // (256, 2*T_frames)
    };

    // F0 stack
    ggml_tensor* F0 = run_stack("pred.F0", shared_out);
    // F0_proj: Conv1d(256 → 1, k=1).
    ggml_tensor* fp_w = require(c, "pred.F0_proj.weight");                            // ne=(1, 256, 1)
    ggml_tensor* fp_b = require(c, "pred.F0_proj.bias");                              // ne=(1,)
    {
        const int Tf = (int)F0->ne[1];
        ggml_tensor* y = ggml_cont(ctx0, ggml_transpose(ctx0, F0));                   // (T, 256)
        y = ggml_conv_1d(ctx0, fp_w, y, /*s*/ 1, /*p*/ 0, /*d*/ 1);                   // (T, 1, 1)
        y = ggml_add(ctx0, y, ggml_reshape_3d(ctx0, fp_b, 1, 1, 1));                  // bias broadcast
        F0 = ggml_reshape_2d(ctx0, y, Tf, 1);                                         // (T, 1)
        // ggml_squeeze isn't available; just keep as (T, 1) and treat the 1 dim
        // as a no-op channel for the downstream extractor.
    }
    F0 = ggml_cont(ctx0, F0);
    ggml_set_name(F0, "f0_curve");
    ggml_set_output(F0);
    ggml_build_forward_expand(gf, F0);

    // N stack (mirror)
    ggml_tensor* N = run_stack("pred.N", shared_out);
    ggml_tensor* np_w = require(c, "pred.N_proj.weight");
    ggml_tensor* np_b = require(c, "pred.N_proj.bias");
    {
        const int Tf = (int)N->ne[1];
        ggml_tensor* y = ggml_cont(ctx0, ggml_transpose(ctx0, N));
        y = ggml_conv_1d(ctx0, np_w, y, /*s*/ 1, /*p*/ 0, /*d*/ 1);
        y = ggml_add(ctx0, y, ggml_reshape_3d(ctx0, np_b, 1, 1, 1));
        N = ggml_reshape_2d(ctx0, y, Tf, 1);
    }
    N = ggml_cont(ctx0, N);
    ggml_set_name(N, "n_curve");
    ggml_set_output(N);
    ggml_build_forward_expand(gf, N);

    ggml_free(ctx0);
    return gf;
}

// Run the full predictor + alignment + F0Ntrain pipeline. Returns the named
// stage as malloc'd float[]. Stages handled:
//   "align_out" → (640, T_frames)
//   "f0_curve"  → (2*T_frames,)   (already squeezed from (1, 2*T_frames))
//   "n_curve"   → (2*T_frames,)
static float* kokoro_run_f0n(kokoro_context* c, const int32_t* raw_ids, int n_raw,
                             const char* stage_name, int* out_n) {
    if (out_n) *out_n = 0;
    if (!c->vp_loaded) {
        fprintf(stderr, "kokoro: F0Ntrain needs voice pack\n");
        return nullptr;
    }
    // 1. Run predictor → dur_enc_out + durations.
    int n_de = 0, n_dr = 0;
    float* de = kokoro_run_predictor(c, raw_ids, n_raw, "dur_enc_out", &n_de);
    if (!de) return nullptr;
    float* dr = kokoro_run_predictor(c, raw_ids, n_raw, "durations", &n_dr);
    if (!dr) { std::free(de); return nullptr; }

    const int D = 640;
    const int L = n_dr;
    if (n_de != D * L) {
        fprintf(stderr, "kokoro: dur_enc/durations length mismatch %d vs %d*%d\n", n_de, D, L);
        std::free(de); std::free(dr); return nullptr;
    }

    // 2. CPU alignment (M6).
    std::vector<int> dur_int((size_t)L);
    for (int i = 0; i < L; i++) dur_int[i] = (int)dr[i];
    std::free(dr);
    int T_frames = 0;
    float* en = kokoro_align_repeat(de, D, L, dur_int.data(), &T_frames);
    std::free(de);
    if (!en || T_frames <= 0) { std::free(en); return nullptr; }

    if (std::strcmp(stage_name, "align_out") == 0) {
        if (out_n) *out_n = D * T_frames;
        return en;
    }

    // 3. Run F0/N graph.
    ggml_cgraph* gf = kokoro_build_graph_f0n(c, T_frames, n_raw);
    if (!gf) { std::free(en); return nullptr; }
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "kokoro: sched_alloc_graph failed for F0Ntrain\n");
        std::free(en); return nullptr;
    }
    ggml_tensor* in_en = ggml_graph_get_tensor(gf, "en");
    ggml_backend_tensor_set(in_en, en, 0, (size_t)D * T_frames * sizeof(float));
    std::free(en);

    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "kokoro: F0Ntrain graph compute failed\n");
        return nullptr;
    }

    const char* tname = stage_name;   // "f0_curve" or "n_curve" — names match graph
    ggml_tensor* out = ggml_graph_get_tensor(gf, tname);
    if (!out) {
        fprintf(stderr, "kokoro: F0Ntrain graph missing output '%s'\n", tname);
        return nullptr;
    }
    const size_t n_floats = (size_t)out->ne[0] * (size_t)out->ne[1];
    float* r = (float*)std::malloc(n_floats * sizeof(float));
    if (!r) return nullptr;
    ggml_backend_tensor_get(out, r, 0, n_floats * sizeof(float));
    if (out_n) *out_n = (int)n_floats;
    return r;
}

// ---------------------------------------------------------------------------
// M7a/M7b — Decoder body (F0_conv + N_conv + asr_res + encode + 4× decode)
//
// Reference: kokoro/istftnet.py Decoder.forward.
//   F0 = self.F0_conv(F0_curve.unsqueeze(1))    # Conv1d 1→1, k=3, s=2, pad=1
//   N  = self.N_conv(N_curve.unsqueeze(1))      # same
//   x  = cat([asr, F0, N], axis=1)              # (514, T_frames)
//   x  = self.encode(x, s_dec)                  # AdainResBlk1d 514→1024
//   asr_res = self.asr_res(asr)                 # Conv1d 512→64, k=1
//   for i in 0..3:
//       x = cat([x, asr_res, F0, N], axis=1)    # → (1090, T)
//       x = self.decode[i](x, s_dec)            # AdainResBlk1d, last has upsample=True
//
// Output dims:
//   dec_encode_out:    (1024, T_frames)
//   dec.decode[0..2]:  (1024, T_frames)
//   dec.decode[3]:     (512, 2*T_frames)        [upsample=True]
//
// All AdainResBlk1d in the decoder use s_dec (voice pack [0:128], NOT
// the predictor's [128:256]).
// ---------------------------------------------------------------------------

static ggml_cgraph* kokoro_build_graph_decoder_body(kokoro_context* c, int T_frames, int L_raw) {
    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), /*no_alloc=*/true};
    ggml_context* ctx0 = ggml_init(ip);
    // Decoder body has no LSTMs, just convs+AdaIN+LeakyReLU. ~50 ops/block × 5 blocks
    // + a few cat/F0_conv/asr_res steps. 16k node budget is generous.
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, 32768, false);

    // ---- Inputs (asr, F0_curve, N_curve from upstream stages) ----
    // asr ne=(512, T_frames) — duration-aligned text_enc_out
    ggml_tensor* asr_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 512, T_frames);
    ggml_set_name(asr_in, "asr");
    ggml_set_input(asr_in);
    // F0_curve ne=(2*T_frames,) → unsqueeze to (1, 2*T_frames) before F0_conv
    ggml_tensor* f0_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, 2 * T_frames);
    ggml_set_name(f0_in, "f0");
    ggml_set_input(f0_in);
    ggml_tensor* n_in = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, 2 * T_frames);
    ggml_set_name(n_in, "n");
    ggml_set_input(n_in);

    ggml_tensor* style_dec = kokoro_voice_style_dec_view(ctx0, c, L_raw);             // (128, 1)

    // ---- F0_conv / N_conv: Conv1d(1, 1, k=3, s=2, pad=1) ----
    auto small_conv1d = [&](ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, int s, int p) {
        // Input x ne=(1, 2T). Conv expects (T, C, 1) layout — we have (1, 2T) which IS (C=1, T=2T) in ggml,
        // but conv_1d wants (T, C). Transpose to (2T, 1).
        const int Tin = (int)x->ne[1];
        ggml_tensor* y = ggml_cont(ctx0, ggml_transpose(ctx0, x));                    // (2T, 1)
        y = ggml_conv_1d(ctx0, w, y, s, p, /*d*/ 1);                                  // (Tout, 1, 1)
        const int Cout = (int)w->ne[2];
        ggml_tensor* b3 = ggml_reshape_3d(ctx0, b, 1, b->ne[0], 1);
        y = ggml_add(ctx0, y, b3);
        const int Tout = (int)y->ne[0];
        y = ggml_reshape_2d(ctx0, y, Tout, Cout);
        (void)Tin;
        return ggml_cont(ctx0, ggml_transpose(ctx0, y));                              // (Cout, Tout)
    };
    ggml_tensor* F0_conv_w = require(c, "dec.F0_conv.weight");
    ggml_tensor* F0_conv_b = require(c, "dec.F0_conv.bias");
    ggml_tensor* N_conv_w  = require(c, "dec.N_conv.weight");
    ggml_tensor* N_conv_b  = require(c, "dec.N_conv.bias");
    ggml_tensor* F0_d = small_conv1d(f0_in, F0_conv_w, F0_conv_b, /*s*/ 2, /*p*/ 1);  // (1, T_frames)
    ggml_tensor* N_d  = small_conv1d(n_in,  N_conv_w,  N_conv_b,  /*s*/ 2, /*p*/ 1);  // (1, T_frames)

    // ---- cat([asr, F0_d, N_d], axis=0) → (514, T_frames) ----
    ggml_tensor* x = ggml_concat(ctx0, asr_in, F0_d, /*dim=*/0);
    x = ggml_concat(ctx0, x, N_d, /*dim=*/0);                                          // (514, T)

    // ---- encode: AdainResBlk1d(514→1024) ----
    auto load_decoder_resblk = [&](const char* prefix, bool has_pool) {
        struct W { ggml_tensor* a1w, *a1b, *a2w, *a2b, *c1w, *c1b, *c2w, *c2b, *poolw, *poolb, *sc; } w;
        char buf[96];
        std::snprintf(buf, sizeof(buf), "%s.adain1.weight", prefix); w.a1w = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.adain1.bias",   prefix); w.a1b = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.adain2.weight", prefix); w.a2w = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.adain2.bias",   prefix); w.a2b = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.conv1.weight",  prefix); w.c1w = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.conv1.bias",    prefix); w.c1b = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.conv2.weight",  prefix); w.c2w = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.conv2.bias",    prefix); w.c2b = require(c, buf);
        std::snprintf(buf, sizeof(buf), "%s.conv1x1.weight", prefix); w.sc = require(c, buf);
        if (has_pool) {
            std::snprintf(buf, sizeof(buf), "%s.pool.weight", prefix); w.poolw = require(c, buf);
            std::snprintf(buf, sizeof(buf), "%s.pool.bias",   prefix); w.poolb = require(c, buf);
        } else {
            w.poolw = w.poolb = nullptr;
        }
        return w;
    };

    {
        auto e = load_decoder_resblk("dec.encode", /*has_pool=*/false);
        x = kokoro_adain_resblk(ctx0, x, style_dec,
                                e.a1w, e.a1b, e.a2w, e.a2b,
                                e.c1w, e.c1b, e.c2w, e.c2b,
                                /*pool*/ nullptr, nullptr, e.sc);                      // (1024, T)
    }
    ggml_tensor* dec_encode_out = ggml_cont(ctx0, x);
    ggml_set_name(dec_encode_out, "dec_encode_out");
    ggml_set_output(dec_encode_out);
    ggml_build_forward_expand(gf, dec_encode_out);

    // ---- asr_res: Conv1d(512→64, k=1) ----
    ggml_tensor* asr_res_w = require(c, "dec.asr_res.weight");
    ggml_tensor* asr_res_b = require(c, "dec.asr_res.bias");
    ggml_tensor* asr_res_out;
    {
        const int Tin = (int)asr_in->ne[1];
        ggml_tensor* y = ggml_cont(ctx0, ggml_transpose(ctx0, asr_in));                // (T, 512)
        y = ggml_conv_1d(ctx0, asr_res_w, y, /*s*/ 1, /*p*/ 0, /*d*/ 1);               // (T, 64, 1)
        ggml_tensor* b3 = ggml_reshape_3d(ctx0, asr_res_b, 1, asr_res_b->ne[0], 1);
        y = ggml_add(ctx0, y, b3);
        y = ggml_reshape_2d(ctx0, y, Tin, 64);                                          // (T, 64)
        asr_res_out = ggml_cont(ctx0, ggml_transpose(ctx0, y));                         // (64, T)
    }

    // ---- 4× decode AdainResBlk1d, with cat([x, asr_res, F0_d, N_d]) before each ----
    x = dec_encode_out;
    for (int il = 0; il < 4; il++) {
        // cat → (1024 + 64 + 1 + 1, T) = (1090, T)
        ggml_tensor* xc = ggml_concat(ctx0, x, asr_res_out, /*dim=*/0);                 // (1088, T)
        xc = ggml_concat(ctx0, xc, F0_d, /*dim=*/0);
        xc = ggml_concat(ctx0, xc, N_d, /*dim=*/0);                                     // (1090, T)

        char prefix[64];
        std::snprintf(prefix, sizeof(prefix), "dec.decode.%d", il);
        const bool has_pool = (il == 3);
        auto w = load_decoder_resblk(prefix, has_pool);
        x = kokoro_adain_resblk(ctx0, xc, style_dec,
                                w.a1w, w.a1b, w.a2w, w.a2b,
                                w.c1w, w.c1b, w.c2w, w.c2b,
                                w.poolw, w.poolb, w.sc);
    }
    // After block 3 (upsample=True): x is (512, 2*T_frames).
    ggml_tensor* dec_decode_3_out = ggml_cont(ctx0, x);
    ggml_set_name(dec_decode_3_out, "dec_decode_3_out");
    ggml_set_output(dec_decode_3_out);
    ggml_build_forward_expand(gf, dec_decode_3_out);

    ggml_free(ctx0);
    return gf;
}

// Run the full preamble (BERT, predictor, alignment, F0Ntrain, text_enc) +
// decoder body, returning the named stage as malloc'd float[].
static float* kokoro_run_decoder_body(kokoro_context* c, const int32_t* raw_ids, int n_raw,
                                      const char* stage_name, int* out_n) {
    if (out_n) *out_n = 0;
    if (!c->vp_loaded) {
        fprintf(stderr, "kokoro: decoder needs voice pack\n");
        return nullptr;
    }

    // 1. dur_enc_out + durations from predictor.
    int n_de = 0, n_dr = 0;
    float* de = kokoro_run_predictor(c, raw_ids, n_raw, "dur_enc_out", &n_de);
    if (!de) return nullptr;
    float* dr = kokoro_run_predictor(c, raw_ids, n_raw, "durations", &n_dr);
    if (!dr) { std::free(de); return nullptr; }
    const int L = n_dr;
    std::vector<int> dur_int((size_t)L);
    for (int i = 0; i < L; i++) dur_int[i] = (int)dr[i];
    std::free(dr);
    int T_frames = 0;
    float* en = kokoro_align_repeat(de, 640, L, dur_int.data(), &T_frames);
    std::free(de);
    if (!en) return nullptr;

    // 2. F0/N curves from F0Ntrain (uses en internally — recompute here for simplicity).
    int n_f = 0, n_nc = 0;
    float* f0 = kokoro_run_f0n(c, raw_ids, n_raw, "f0_curve", &n_f);
    if (!f0) { std::free(en); return nullptr; }
    float* nc = kokoro_run_f0n(c, raw_ids, n_raw, "n_curve", &n_nc);
    if (!nc) { std::free(en); std::free(f0); return nullptr; }

    // 3. text_enc_out, then duration-align to asr.
    int n_te = 0;
    float* te = kokoro_run_text_enc(c, raw_ids, n_raw, "text_enc_out", &n_te);
    if (!te) { std::free(en); std::free(f0); std::free(nc); return nullptr; }
    int T_frames_asr = 0;
    float* asr = kokoro_align_repeat(te, 512, L, dur_int.data(), &T_frames_asr);
    std::free(te);
    if (!asr || T_frames_asr != T_frames) {
        fprintf(stderr, "kokoro: T_frames mismatch (%d != %d)\n", T_frames_asr, T_frames);
        std::free(asr); std::free(en); std::free(f0); std::free(nc); return nullptr;
    }
    std::free(en);   // not directly used by decoder body — F0Ntrain consumed it already

    // 4. Build + run decoder-body graph.
    ggml_cgraph* gf = kokoro_build_graph_decoder_body(c, T_frames, n_raw);
    if (!gf) { std::free(asr); std::free(f0); std::free(nc); return nullptr; }
    ggml_backend_sched_reset(c->sched);
    if (!ggml_backend_sched_alloc_graph(c->sched, gf)) {
        fprintf(stderr, "kokoro: sched_alloc_graph failed for decoder body\n");
        std::free(asr); std::free(f0); std::free(nc); return nullptr;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "asr"), asr, 0,
                            (size_t)512 * T_frames * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "f0"),  f0,  0,
                            (size_t)2 * T_frames * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "n"),   nc,  0,
                            (size_t)2 * T_frames * sizeof(float));
    std::free(asr); std::free(f0); std::free(nc);

    if (ggml_backend_sched_graph_compute(c->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "kokoro: decoder-body graph compute failed\n");
        return nullptr;
    }

    ggml_tensor* out = ggml_graph_get_tensor(gf, stage_name);
    if (!out) {
        fprintf(stderr, "kokoro: decoder-body graph missing output '%s'\n", stage_name);
        return nullptr;
    }
    const size_t n_floats = (size_t)out->ne[0] * (size_t)out->ne[1];
    float* r = (float*)std::malloc(n_floats * sizeof(float));
    if (!r) return nullptr;
    ggml_backend_tensor_get(out, r, 0, n_floats * sizeof(float));
    if (out_n) *out_n = (int)n_floats;
    return r;
}

} // namespace

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------

extern "C" struct kokoro_context_params kokoro_context_default_params(void) {
    kokoro_context_params p{};
    p.n_threads = 4;
    p.verbosity = 1;
    p.use_gpu = true;
    p.gen_force_metal = env_bool("KOKORO_GEN_FORCE_METAL");
    std::strncpy(p.espeak_lang, "en-us", sizeof(p.espeak_lang) - 1);
    return p;
}

extern "C" struct kokoro_context* kokoro_init_from_file(const char* path_model,
                                                         struct kokoro_context_params params) {
    if (!path_model) {
        fprintf(stderr, "kokoro: null model path\n");
        return nullptr;
    }
    // Match PyTorch's torch.round behaviour (banker's rounding) when the
    // runtime later applies `nearbyintf` to durations (R5 in the plan).
    // Setting this globally is benign — the only place we round is the
    // duration post-processing.
    std::fesetround(FE_TONEAREST);
    auto* c = new kokoro_context();
    c->params = params;
    c->n_threads = params.n_threads > 0 ? params.n_threads : 4;
    if (params.espeak_lang[0])
        c->espeak_lang = params.espeak_lang;

    // ---- Pass 1: metadata (hparams + vocab) ----
    {
        gguf_context* g = core_gguf::open_metadata(path_model);
        if (!g) {
            fprintf(stderr, "kokoro: failed to read GGUF '%s'\n", path_model);
            delete c;
            return nullptr;
        }

        // Architecture sanity. The converter writes "kokoro" for both
        // hexgrad/Kokoro-82M and yl4579/StyleTTS2-LJSpeech.
        std::string arch = core_gguf::kv_str(g, "general.architecture", "");
        if (arch != "kokoro") {
            fprintf(stderr, "kokoro: unexpected general.architecture='%s' (want 'kokoro')\n", arch.c_str());
            gguf_free(g);
            delete c;
            return nullptr;
        }

        auto& hp = c->hp;
        hp.hidden_dim   = core_gguf::kv_u32(g, "kokoro.hidden_dim", hp.hidden_dim);
        hp.style_dim    = core_gguf::kv_u32(g, "kokoro.style_dim", hp.style_dim);
        hp.max_dur      = core_gguf::kv_u32(g, "kokoro.max_dur", hp.max_dur);
        hp.n_token      = core_gguf::kv_u32(g, "kokoro.n_token", hp.n_token);
        hp.n_mels       = core_gguf::kv_u32(g, "kokoro.n_mels", hp.n_mels);
        hp.n_layer      = core_gguf::kv_u32(g, "kokoro.n_layer", hp.n_layer);
        hp.text_enc_k   = core_gguf::kv_u32(g, "kokoro.text_encoder_kernel_size", hp.text_enc_k);
        hp.sample_rate  = core_gguf::kv_u32(g, "kokoro.sample_rate", hp.sample_rate);
        hp.vocab_size   = core_gguf::kv_u32(g, "kokoro.vocab_size", hp.vocab_size);

        hp.plbert_embd_size  = core_gguf::kv_u32(g, "kokoro.plbert.embedding_size", hp.plbert_embd_size);
        hp.plbert_hidden     = core_gguf::kv_u32(g, "kokoro.plbert.hidden_size", hp.plbert_hidden);
        hp.plbert_n_layers   = core_gguf::kv_u32(g, "kokoro.plbert.num_hidden_layers", hp.plbert_n_layers);
        hp.plbert_n_heads    = core_gguf::kv_u32(g, "kokoro.plbert.num_attention_heads", hp.plbert_n_heads);
        hp.plbert_ff         = core_gguf::kv_u32(g, "kokoro.plbert.intermediate_size", hp.plbert_ff);
        hp.plbert_max_pos    = core_gguf::kv_u32(g, "kokoro.plbert.max_position_embeddings", hp.plbert_max_pos);
        hp.plbert_vocab_size = core_gguf::kv_u32(g, "kokoro.plbert.vocab_size", hp.plbert_vocab_size);

        hp.istft_init_ch     = core_gguf::kv_u32(g, "kokoro.istft.init_channel", hp.istft_init_ch);
        hp.istft_n_fft       = core_gguf::kv_u32(g, "kokoro.istft.n_fft", hp.istft_n_fft);
        hp.istft_hop         = core_gguf::kv_u32(g, "kokoro.istft.hop_size", hp.istft_hop);
        hp.istft_n_dilations = core_gguf::kv_u32(g, "kokoro.istft.resblock_n_dilations", hp.istft_n_dilations);

        hp.istft_upsample_rates        = kv_u32_array(g, "kokoro.istft.upsample_rates");
        hp.istft_upsample_kernel_sizes = kv_u32_array(g, "kokoro.istft.upsample_kernel_sizes");
        hp.istft_resblock_kernel_sizes = kv_u32_array(g, "kokoro.istft.resblock_kernel_sizes");
        hp.istft_resblock_dilations    = kv_u32_array(g, "kokoro.istft.resblock_dilation_sizes");

        if (hp.istft_upsample_rates.empty())
            hp.istft_upsample_rates = {10, 6};
        if (hp.istft_upsample_kernel_sizes.empty())
            hp.istft_upsample_kernel_sizes = {20, 12};
        if (hp.istft_resblock_kernel_sizes.empty())
            hp.istft_resblock_kernel_sizes = {3, 7, 11};
        if (hp.istft_resblock_dilations.empty())
            hp.istft_resblock_dilations = {1, 3, 5, 1, 3, 5, 1, 3, 5};

        c->vocab.id_to_token = core_gguf::kv_str_array(g, "tokenizer.ggml.tokens");
        c->vocab.token_to_id.reserve(c->vocab.id_to_token.size());
        for (int i = 0; i < (int)c->vocab.id_to_token.size(); i++) {
            const auto& t = c->vocab.id_to_token[i];
            if (!t.empty())
                c->vocab.token_to_id[t] = i;
        }
        // StyleTTS2/Kokoro convention: index 0 is the "$" pad token.
        c->vocab.pad_id = 0;

        gguf_free(g);
    }

    if (params.verbosity >= 1) {
        const auto& hp = c->hp;
        fprintf(stderr,
                "kokoro: arch=kokoro  hidden=%u style=%u max_dur=%u n_token=%u "
                "vocab=%zu plbert=%uL/%uH/ff=%u  iSTFT n_fft=%u hop=%u sr=%u\n",
                hp.hidden_dim, hp.style_dim, hp.max_dur, hp.n_token,
                c->vocab.id_to_token.size(),
                hp.plbert_n_layers, hp.plbert_hidden, hp.plbert_ff,
                hp.istft_n_fft, hp.istft_hop, hp.sample_rate);
    }

    // ---- Backends ----
    c->backend_cpu = ggml_backend_cpu_init();
    if (!c->backend_cpu) {
        fprintf(stderr, "kokoro: failed to init CPU backend\n");
        delete c;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(c->backend_cpu, c->n_threads);
    c->backend = params.use_gpu ? ggml_backend_init_best() : c->backend_cpu;
    if (!c->backend)
        c->backend = c->backend_cpu;
    c->gen_backend = params.gen_force_metal ? c->backend : c->backend_cpu;

    // ---- Pass 2: weights ----
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path_model, c->backend, "kokoro", wl)) {
        fprintf(stderr, "kokoro: failed to load weights from '%s'\n", path_model);
        kokoro_free(c);
        return nullptr;
    }
    c->ctx_w = wl.ctx;
    c->buf_w = wl.buf;
    c->tensors = std::move(wl.tensors);

    if (!sanity_check_weights(c)) {
        fprintf(stderr, "kokoro: weight sanity check failed for '%s'\n", path_model);
        kokoro_free(c);
        return nullptr;
    }

    // ---- Schedulers ----
    {
        ggml_backend_t backends[2];
        int n_be = 0;
        backends[n_be++] = c->backend;
        if (c->backend_cpu && c->backend_cpu != c->backend)
            backends[n_be++] = c->backend_cpu;
        // 256k nodes — predictor has 4 bidir LSTMs at T_frames up to ~1500
        // (~54k LSTM nodes) plus the decoder graph; the upper bound is
        // generous to accommodate AdainResBlk1d + AdaIN expansion.
        c->sched = ggml_backend_sched_new(backends, nullptr, n_be, /*graph_size=*/262144,
                                          /*parallel=*/false, /*op_offload=*/false);
        if (!c->sched) {
            fprintf(stderr, "kokoro: failed to allocate main scheduler\n");
            kokoro_free(c);
            return nullptr;
        }

        // Generator scheduler — single-backend (CPU by default; GPU only if
        // KOKORO_GEN_FORCE_METAL=1). Same node ceiling: the generator graph
        // is dominated by snake-α + transpose-1d + the 22-channel conv_post.
        ggml_backend_t gen_backends[1] = {c->gen_backend};
        c->gen_sched = ggml_backend_sched_new(gen_backends, nullptr, 1, /*graph_size=*/262144, false, false);
        if (!c->gen_sched) {
            fprintf(stderr, "kokoro: failed to allocate generator scheduler\n");
            kokoro_free(c);
            return nullptr;
        }
    }
    c->compute_meta.resize(ggml_tensor_overhead() * 262144 + ggml_graph_overhead_custom(262144, false));

    if (params.verbosity >= 1) {
        fprintf(stderr, "kokoro: loaded %zu tensors from '%s'  gen=%s\n",
                c->tensors.size(), path_model,
                c->gen_backend == c->backend_cpu ? "CPU (Metal-hang workaround)" : "GPU (KOKORO_GEN_FORCE_METAL)");
    }
    return c;
}

extern "C" int kokoro_load_voice_pack(struct kokoro_context* ctx, const char* path) {
    if (!ctx || !path)
        return -1;

    // Read voice metadata: kokoro_voice.{name,max_phonemes,style_dim}.
    {
        gguf_context* g = core_gguf::open_metadata(path);
        if (!g) {
            fprintf(stderr, "kokoro: failed to read voice pack '%s'\n", path);
            return -1;
        }

        std::string arch = core_gguf::kv_str(g, "general.architecture", "");
        if (arch != "kokoro-voice") {
            fprintf(stderr, "kokoro: voice pack '%s' has architecture='%s' (want 'kokoro-voice')\n",
                    path, arch.c_str());
            gguf_free(g);
            return -1;
        }

        kokoro_voice_pack vp;
        vp.name          = core_gguf::kv_str(g, "kokoro_voice.name", "");
        vp.max_phonemes  = core_gguf::kv_u32(g, "kokoro_voice.max_phonemes", 0);
        vp.style_dim     = core_gguf::kv_u32(g, "kokoro_voice.style_dim", 0);
        gguf_free(g);

        // Replace any previously-loaded pack.
        if (ctx->vp.vp_buf_w)
            ggml_backend_buffer_free(ctx->vp.vp_buf_w);
        if (ctx->vp.vp_ctx_w)
            ggml_free(ctx->vp.vp_ctx_w);
        ctx->vp = std::move(vp);
        ctx->vp_loaded = false;
    }

    // Load the single F32 tensor `voice.pack`. Use the main `backend` so
    // ggml_get_rows / ggml_view_2d access works without a backend hop
    // during graph build.
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(path, ctx->backend, "kokoro.voice", wl)) {
        fprintf(stderr, "kokoro: failed to load voice pack tensors from '%s'\n", path);
        return -1;
    }

    auto it = wl.tensors.find("voice.pack");
    if (it == wl.tensors.end() || !it->second) {
        fprintf(stderr, "kokoro: voice pack '%s' missing 'voice.pack' tensor\n", path);
        ggml_backend_buffer_free(wl.buf);
        ggml_free(wl.ctx);
        return -1;
    }

    ctx->vp.tensors  = std::move(wl.tensors);
    ctx->vp.pack     = it->second;
    ctx->vp.vp_ctx_w = wl.ctx;
    ctx->vp.vp_buf_w = wl.buf;
    ctx->vp_loaded   = true;

    if (ctx->params.verbosity >= 1) {
        fprintf(stderr, "kokoro: voice '%s' (max_phonemes=%u style_dim=%u) loaded from '%s'\n",
                ctx->vp.name.c_str(), ctx->vp.max_phonemes, ctx->vp.style_dim, path);
    }
    return 0;
}

extern "C" int kokoro_set_language(struct kokoro_context* ctx, const char* espeak_lang) {
    if (!ctx || !espeak_lang)
        return -1;
    ctx->espeak_lang = espeak_lang;
    return 0;
}

extern "C" int32_t* kokoro_phonemes_to_ids(struct kokoro_context* ctx, const char* phonemes, int* out_n) {
    if (out_n)
        *out_n = 0;
    if (!ctx || !phonemes)
        return nullptr;

    // The vocab is 178 IPA symbols, each typically 1 codepoint (1-4 UTF-8
    // bytes). We greedy-tokenise: at each position, try the longest
    // matching token first (combining marks like ̃ or ː are 2 bytes and
    // attach to the previous letter, so they tokenise as their own ids
    // when they appear standalone). Unknown codepoints emit pad_id and a
    // stderr warning.
    std::vector<int32_t> ids;
    ids.reserve(std::strlen(phonemes));

    auto utf8_len = [](unsigned char c) -> int {
        if ((c & 0x80) == 0)    return 1;
        if ((c & 0xE0) == 0xC0) return 2;
        if ((c & 0xF0) == 0xE0) return 3;
        if ((c & 0xF8) == 0xF0) return 4;
        return 1; // invalid leading byte — consume one byte and move on
    };

    const char* p = phonemes;
    while (*p) {
        // Try lookahead: 4-byte → 3-byte → 2-byte → 1-byte greedy.
        // Combining marks following a base letter can form 2-codepoint
        // tokens (e.g. "̃" follows a base) — we don't try to combine them
        // because StyleTTS2's vocab stores them as standalone entries.
        int best_len = 0;
        int best_id = -1;
        for (int try_len = 4; try_len >= 1; try_len--) {
            // Fast path: skip if the current byte's UTF-8 length forbids it.
            const int real_len = utf8_len((unsigned char)*p);
            if (try_len > real_len)
                continue;
            std::string tok(p, (size_t)try_len);
            auto it = ctx->vocab.token_to_id.find(tok);
            if (it != ctx->vocab.token_to_id.end()) {
                best_len = try_len;
                best_id = it->second;
                break;
            }
        }
        if (best_id >= 0) {
            ids.push_back(best_id);
            p += best_len;
        } else {
            // Match the reference KModel.forward() behaviour: unknown
            // phonemes are dropped (Python uses
            // `filter(lambda i: i is not None, map(vocab.get, phonemes))`).
            // Emitting pad here would diverge from the reference and
            // poison every downstream stage's diff.
            const int n = utf8_len((unsigned char)*p);
            if (ctx->params.verbosity >= 1) {
                std::string bad(p, (size_t)n);
                fprintf(stderr, "kokoro: unknown phoneme '%s' — skipped\n", bad.c_str());
            }
            p += n;
        }
    }

    int32_t* out = (int32_t*)std::malloc(ids.size() * sizeof(int32_t));
    if (!out)
        return nullptr;
    std::memcpy(out, ids.data(), ids.size() * sizeof(int32_t));
    if (out_n)
        *out_n = (int)ids.size();
    return out;
}

extern "C" float* kokoro_synthesize(struct kokoro_context* ctx, const char* text, int* out_n_samples) {
    if (out_n_samples)
        *out_n_samples = 0;
    (void)ctx;
    (void)text;
    fprintf(stderr, "kokoro: synthesize not yet implemented (M9 milestone)\n");
    return nullptr;
}

extern "C" float* kokoro_synthesize_phonemes(struct kokoro_context* ctx, const char* phonemes, int* out_n_samples) {
    if (out_n_samples)
        *out_n_samples = 0;
    (void)ctx;
    (void)phonemes;
    fprintf(stderr, "kokoro: synthesize_phonemes not yet implemented (M9 milestone)\n");
    return nullptr;
}

extern "C" float* kokoro_extract_stage(struct kokoro_context* ctx, const char* phonemes, const char* stage_name,
                                        int* out_n) {
    if (out_n)
        *out_n = 0;
    if (!ctx || !phonemes || !stage_name)
        return nullptr;

    // Stages: token_ids (M1), bert_pooler_out / bert_proj_out (M3).
    // Future: text_enc_out, dur_enc_out, durations, align_out, f0_curve,
    // n_curve, dec_*_out, mag, phase, audio_out.
    if (std::strcmp(stage_name, "token_ids") == 0) {
        int n = 0;
        int32_t* ids = kokoro_phonemes_to_ids(ctx, phonemes, &n);
        if (!ids)
            return nullptr;
        float* out = (float*)std::malloc((size_t)n * sizeof(float));
        if (!out) {
            std::free(ids);
            return nullptr;
        }
        for (int i = 0; i < n; i++)
            out[i] = (float)ids[i];
        std::free(ids);
        if (out_n)
            *out_n = n;
        return out;
    }

    if (std::strcmp(stage_name, "bert_pooler_out") == 0 ||
        std::strcmp(stage_name, "bert_proj_out") == 0) {
        int n_ids = 0;
        int32_t* ids = kokoro_phonemes_to_ids(ctx, phonemes, &n_ids);
        if (!ids || n_ids == 0) {
            std::free(ids);
            fprintf(stderr, "kokoro: empty phoneme tokenisation for '%s'\n", phonemes);
            return nullptr;
        }
        float* r = kokoro_run_bert(ctx, ids, n_ids, stage_name, out_n);
        std::free(ids);
        return r;
    }

    if (std::strcmp(stage_name, "text_enc_out") == 0) {
        int n_ids = 0;
        int32_t* ids = kokoro_phonemes_to_ids(ctx, phonemes, &n_ids);
        if (!ids || n_ids == 0) {
            std::free(ids);
            fprintf(stderr, "kokoro: empty phoneme tokenisation for '%s'\n", phonemes);
            return nullptr;
        }
        float* r = kokoro_run_text_enc(ctx, ids, n_ids, stage_name, out_n);
        std::free(ids);
        return r;
    }

    if (std::strcmp(stage_name, "dur_enc_out") == 0 ||
        std::strcmp(stage_name, "durations") == 0 ||
        std::strcmp(stage_name, "pred_lstm_out") == 0) {
        int n_ids = 0;
        int32_t* ids = kokoro_phonemes_to_ids(ctx, phonemes, &n_ids);
        if (!ids || n_ids == 0) {
            std::free(ids);
            fprintf(stderr, "kokoro: empty phoneme tokenisation for '%s'\n", phonemes);
            return nullptr;
        }
        float* r = kokoro_run_predictor(ctx, ids, n_ids, stage_name, out_n);
        std::free(ids);
        return r;
    }

    if (std::strcmp(stage_name, "align_out") == 0 ||
        std::strcmp(stage_name, "f0_curve")  == 0 ||
        std::strcmp(stage_name, "n_curve")   == 0) {
        int n_ids = 0;
        int32_t* ids = kokoro_phonemes_to_ids(ctx, phonemes, &n_ids);
        if (!ids || n_ids == 0) {
            std::free(ids);
            fprintf(stderr, "kokoro: empty phoneme tokenisation for '%s'\n", phonemes);
            return nullptr;
        }
        float* r = kokoro_run_f0n(ctx, ids, n_ids, stage_name, out_n);
        std::free(ids);
        return r;
    }

    if (std::strcmp(stage_name, "dec_encode_out") == 0 ||
        std::strcmp(stage_name, "dec_decode_3_out") == 0) {
        int n_ids = 0;
        int32_t* ids = kokoro_phonemes_to_ids(ctx, phonemes, &n_ids);
        if (!ids || n_ids == 0) {
            std::free(ids);
            fprintf(stderr, "kokoro: empty phoneme tokenisation for '%s'\n", phonemes);
            return nullptr;
        }
        float* r = kokoro_run_decoder_body(ctx, ids, n_ids, stage_name, out_n);
        std::free(ids);
        return r;
    }

    fprintf(stderr, "kokoro: stage '%s' not yet implemented\n", stage_name);
    return nullptr;
}

extern "C" void kokoro_pcm_free(float* pcm) {
    std::free(pcm);
}

extern "C" void kokoro_set_n_threads(struct kokoro_context* ctx, int n_threads) {
    if (!ctx || n_threads <= 0)
        return;
    ctx->n_threads = n_threads;
    if (ctx->backend_cpu)
        ggml_backend_cpu_set_n_threads(ctx->backend_cpu, n_threads);
}

extern "C" void kokoro_free(struct kokoro_context* ctx) {
    if (!ctx)
        return;
    if (ctx->gen_sched)
        ggml_backend_sched_free(ctx->gen_sched);
    if (ctx->sched)
        ggml_backend_sched_free(ctx->sched);
    if (ctx->vp.vp_buf_w)
        ggml_backend_buffer_free(ctx->vp.vp_buf_w);
    if (ctx->vp.vp_ctx_w)
        ggml_free(ctx->vp.vp_ctx_w);
    if (ctx->buf_w)
        ggml_backend_buffer_free(ctx->buf_w);
    if (ctx->ctx_w)
        ggml_free(ctx->ctx_w);
    if (ctx->backend && ctx->backend != ctx->backend_cpu)
        ggml_backend_free(ctx->backend);
    if (ctx->backend_cpu)
        ggml_backend_free(ctx->backend_cpu);
    delete ctx;
}
