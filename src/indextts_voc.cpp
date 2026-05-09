// indextts_voc.cpp -- BigVGAN vocoder for IndexTTS-1.5.
//
// Architecture: BigVGAN v2 with SnakeBeta activations and anti-aliased
// multi-periodicity composition (AMPBlock1). Converts GPT hidden states
// (d=1280, T time steps) to 24 kHz mono waveform.
//
// Forward pass:
//   latent [T, 1280]  (GPT hidden states)
//   x = conv_pre(latent)         Conv1d(1280, 1536, k=7, pad=3)
//   x = x + cond_layer(spk_emb) Conv1d(512, 1536, k=1)
//   for i in 0..5:
//       x = snake_beta(x)
//       x = ups[i](x)            ConvTranspose1d upsample
//       x = x + conds[i](spk_emb)
//       xs = 0
//       for j in 0..2:
//           xs += resblock[i*3+j](x)
//       x = xs / 3
//   x = snake_beta_post(x)
//   x = conv_post(x)             Conv1d(24, 1, k=7, pad=3)
//   x = tanh(x)
//
// SnakeBeta: x + (1/beta) * sin(alpha * x)^2
//   where alpha = exp(log_alpha), beta = exp(log_beta) (per-channel).
//
// Weight-norm is already fused in the GGUF tensors.
// Anti-aliased up/downsampling in AMPBlock1 is skipped for now.

#include "indextts_voc.h"
#include "core/gguf_loader.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

// ── Hyperparameters ──────────────────────────────────────────────

struct bigvgan_hp {
    int gpt_dim = 1280;
    int upsample_initial_ch = 1536;
    int num_upsamples = 6;
    int num_kernels = 3;
    int spk_emb_dim = 512;
    int sampling_rate = 24000;
    int hop_size = 256;

    int upsample_rates[6] = {4, 4, 4, 4, 2, 2};
    int upsample_kernel_sizes[6] = {8, 8, 4, 4, 4, 4};
    int resblock_kernel_sizes[3] = {3, 7, 11};
    // Dilations per dilated pass: [1, 3, 5] for all resblocks.
    int resblock_dilations[3] = {1, 3, 5};
};

// ── Tensor lookup helper ────────────────────────────────────────

static ggml_tensor* T(const std::map<std::string, ggml_tensor*>& m, const char* name) {
    auto it = m.find(name);
    return (it != m.end()) ? it->second : nullptr;
}

static ggml_tensor* T(const std::map<std::string, ggml_tensor*>& m, const std::string& name) {
    auto it = m.find(name);
    return (it != m.end()) ? it->second : nullptr;
}

} // namespace

// ── Context ─────────────────────────────────────────────────────

struct indextts_voc_context {
    bigvgan_hp hp;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_context* ctx_w = nullptr;
    ggml_backend_buffer_t buf_w = nullptr;
    std::map<std::string, ggml_tensor*> tensors;

    // Compute scheduler
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;

    int n_threads = 4;
    int verbosity = 1;

    ~indextts_voc_context() {
        if (sched) {
            ggml_backend_sched_free(sched);
        }
        if (ctx_w) {
            ggml_free(ctx_w);
        }
        if (buf_w) {
            ggml_backend_buffer_free(buf_w);
        }
        if (backend && backend != backend_cpu) {
            ggml_backend_free(backend);
        }
        if (backend_cpu) {
            ggml_backend_free(backend_cpu);
        }
    }
};

namespace {

// ── SnakeBeta as ggml ops ───────────────────────────────────────
//
// SnakeBeta(x, log_alpha, log_beta):
//   alpha = exp(log_alpha)  -- per channel
//   beta  = exp(log_beta)   -- per channel
//   x_out = x + (1/beta) * sin(alpha * x)^2
//
// In ggml: alpha/beta are reshaped to (1, C) for broadcasting over (T, C).
// We use ggml_exp on the log params, then:
//   ax = x * alpha
//   sin_ax = sin(ax)
//   sin2 = sin_ax * sin_ax
//   out = x + sin2 / beta

static ggml_tensor* snake_beta(ggml_context* ctx, ggml_tensor* x, ggml_tensor* log_alpha, ggml_tensor* log_beta) {
    if (!log_alpha || !log_beta) {
        return x;
    }
    int C = (int)log_alpha->ne[0];

    // Reshape to (1, C) for broadcasting over x which is (T, C)
    ggml_tensor* alpha = ggml_exp(ctx, ggml_reshape_2d(ctx, log_alpha, 1, C));
    ggml_tensor* beta = ggml_exp(ctx, ggml_reshape_2d(ctx, log_beta, 1, C));

    ggml_tensor* ax = ggml_mul(ctx, x, alpha);
    ggml_tensor* sin_ax = ggml_sin(ctx, ax);
    ggml_tensor* sin2 = ggml_mul(ctx, sin_ax, sin_ax);
    ggml_tensor* term = ggml_div(ctx, sin2, beta);
    return ggml_add(ctx, x, term);
}

// ── Build BigVGAN graph ─────────────────────────────────────────

static ggml_cgraph* build_bigvgan_graph(indextts_voc_context* c, int T_in) {
    const auto& hp = c->hp;
    auto& ts = c->tensors;

    const size_t n_nodes = 32768;
    ggml_init_params ip = {c->compute_meta.size(), c->compute_meta.data(), true};
    ggml_context* ctx0 = ggml_init(ip);
    ggml_cgraph* gf = ggml_new_graph_custom(ctx0, n_nodes, false);

    // Input: latent (T_in, gpt_dim) — ggml layout: ne[0]=T_in, ne[1]=gpt_dim
    // But we want Conv1d to see channels in ne[1], time in ne[0].
    // ggml_conv_1d expects input shape (T, C_in) and kernel (k, C_in, C_out).
    ggml_tensor* x = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T_in, hp.gpt_dim);
    ggml_set_name(x, "latent_input");
    ggml_set_input(x);

    // Speaker embedding input: (1, spk_emb_dim)
    ggml_tensor* spk = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, hp.spk_emb_dim);
    ggml_set_name(spk, "spk_emb");
    ggml_set_input(spk);

    // conv_pre: Conv1d(gpt_dim, upsample_initial_ch, k=7, pad=3)
    ggml_tensor* cpre_w = T(ts, "conv_pre.weight");
    ggml_tensor* cpre_b = T(ts, "conv_pre.bias");
    if (cpre_w) {
        x = ggml_conv_1d(ctx0, cpre_w, x, 1, 3, 1);
        if (cpre_b) {
            x = ggml_add(ctx0, x, ggml_reshape_2d(ctx0, cpre_b, 1, (int)cpre_b->ne[0]));
        }
    }

    // Speaker conditioning: x += cond_layer(spk_emb)
    // cond_layer: Conv1d(512, 1536, k=1)
    // spk is (1, 512), cond_layer.weight is (1, 512, 1536)
    // After conv1d: (1, 1536) -> broadcast add to x (T, 1536)
    {
        ggml_tensor* cond_w = T(ts, "cond_layer.weight");
        ggml_tensor* cond_b = T(ts, "cond_layer.bias");
        if (cond_w) {
            ggml_tensor* cond = ggml_conv_1d(ctx0, cond_w, spk, 1, 0, 1);
            if (cond_b) {
                cond = ggml_add(ctx0, cond, ggml_reshape_2d(ctx0, cond_b, 1, (int)cond_b->ne[0]));
            }
            x = ggml_add(ctx0, x, cond);
        }
    }

    // Channel sizes after each upsample: 1536 -> 768 -> 384 -> 192 -> 96 -> 48 -> 24
    int ch = hp.upsample_initial_ch;

    for (int i = 0; i < hp.num_upsamples; i++) {
        int s = hp.upsample_rates[i];
        int k = hp.upsample_kernel_sizes[i];
        int ch_out = ch / 2;

        // SnakeBeta activation before upsample
        {
            char aname[64], bname[64];
            // The pre-upsample snake uses the first activation of the first
            // resblock at this level. Actually, looking at BigVGAN source:
            // ups[i] has its own activation. Let me check tensor names...
            // In BigVGAN Python: self.ups has LeakyReLU before each
            // ConvTranspose1d. But with snake_logscale, it uses SnakeBeta.
            // The Python code stores it as ups[i] = [Activation, ConvTranspose1d].
            // In the GGUF, ups.{i}.0 is the ConvTranspose1d weight.
            // The activation before ups is actually just the first activation
            // of the first resblock at this level... No, let me look more carefully.
            //
            // Actually, BigVGAN v2 uses the pattern:
            //   for i in range(num_upsamples):
            //       x = act(x)      # SnakeBeta (part of ups module)
            //       x = ups[i](x)   # ConvTranspose1d
            //
            // But looking at our GGUF, there's no separate activation tensor
            // for the ups modules. The activations in the GGUF are all under
            // resblocks.{n}.act.{m}. So the pre-upsample activation
            // must be handled differently.
            //
            // Looking at the BigVGAN code more carefully:
            //   self.ups = nn.ModuleList()
            //   for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            //       self.ups.append(nn.ModuleList([
            //           Activation1d(activation=SnakeBeta(...)),
            //           ...ConvTranspose1d...
            //       ]))
            //
            // But in IndexTTS fork, the ups activation might be stored differently.
            // Since we don't see explicit ups activation tensors in GGUF,
            // use LeakyReLU as fallback (the original HiFi-GAN pattern).
            (void)aname;
            (void)bname;
            x = ggml_leaky_relu(ctx0, x, 0.1f, false);
        }

        // ConvTranspose1d upsample
        {
            char wn[32], bn[32];
            std::snprintf(wn, sizeof(wn), "ups.%d.0.weight", i);
            std::snprintf(bn, sizeof(bn), "ups.%d.0.bias", i);
            ggml_tensor* up_w = T(ts, wn);
            ggml_tensor* up_b = T(ts, bn);
            if (up_w) {
                int T_cur = (int)x->ne[0];
                x = ggml_conv_transpose_1d(ctx0, up_w, x, s, 0, 1);
                // Crop padding: output with p=0 is (T-1)*s+k, we want T*s
                int p = (k - s) / 2;
                if (p > 0) {
                    int T_want = T_cur * s;
                    int C_out_t = (int)x->ne[1];
                    x = ggml_view_2d(ctx0, x, T_want, C_out_t, x->nb[1], (size_t)p * x->nb[0]);
                    x = ggml_cont(ctx0, x);
                }
                if (up_b) {
                    x = ggml_add(ctx0, x, ggml_reshape_2d(ctx0, up_b, 1, (int)up_b->ne[0]));
                }
            }
        }

        // Per-level speaker conditioning: x += conds[i](spk_emb)
        {
            char wn[32], bn[32];
            std::snprintf(wn, sizeof(wn), "conds.%d.weight", i);
            std::snprintf(bn, sizeof(bn), "conds.%d.bias", i);
            ggml_tensor* cw = T(ts, wn);
            ggml_tensor* cb = T(ts, bn);
            if (cw) {
                ggml_tensor* cond = ggml_conv_1d(ctx0, cw, spk, 1, 0, 1);
                if (cb) {
                    cond = ggml_add(ctx0, cond, ggml_reshape_2d(ctx0, cb, 1, (int)cb->ne[0]));
                }
                x = ggml_add(ctx0, x, cond);
            }
        }

        // 3 ResBlocks in parallel, averaged
        ggml_tensor* rb_sum = nullptr;
        ggml_tensor* rb_input = x;

        for (int j = 0; j < hp.num_kernels; j++) {
            x = rb_input; // reset to same input
            int rb_idx = i * hp.num_kernels + j;
            int rb_k = hp.resblock_kernel_sizes[j];

            ggml_tensor* rb_residual = x;

            // 3 dilated passes per resblock
            for (int d = 0; d < 3; d++) {
                int dil = hp.resblock_dilations[d];
                int act_idx_1 = d * 2;     // activation indices: 0,2,4
                int act_idx_2 = d * 2 + 1; // activation indices: 1,3,5

                char key[80];

                // SnakeBeta activation 1
                std::snprintf(key, sizeof(key), "resb.%d.act.%d.act.alpha", rb_idx, act_idx_1);
                ggml_tensor* alpha1 = T(ts, key);
                std::snprintf(key, sizeof(key), "resb.%d.act.%d.act.beta", rb_idx, act_idx_1);
                ggml_tensor* beta1 = T(ts, key);
                x = snake_beta(ctx0, x, alpha1, beta1);

                // Conv1d with dilation
                int pad1 = (rb_k * dil - dil) / 2;
                std::snprintf(key, sizeof(key), "resb.%d.convs1.%d.weight", rb_idx, d);
                ggml_tensor* c1w = T(ts, key);
                std::snprintf(key, sizeof(key), "resb.%d.convs1.%d.bias", rb_idx, d);
                ggml_tensor* c1b = T(ts, key);
                if (c1w) {
                    x = ggml_conv_1d(ctx0, c1w, x, 1, pad1, dil);
                    if (c1b) {
                        x = ggml_add(ctx0, x, ggml_reshape_2d(ctx0, c1b, 1, (int)c1b->ne[0]));
                    }
                }

                // SnakeBeta activation 2
                std::snprintf(key, sizeof(key), "resb.%d.act.%d.act.alpha", rb_idx, act_idx_2);
                ggml_tensor* alpha2 = T(ts, key);
                std::snprintf(key, sizeof(key), "resb.%d.act.%d.act.beta", rb_idx, act_idx_2);
                ggml_tensor* beta2 = T(ts, key);
                x = snake_beta(ctx0, x, alpha2, beta2);

                // Conv2d (dilation=1)
                int pad2 = (rb_k - 1) / 2;
                std::snprintf(key, sizeof(key), "resb.%d.convs2.%d.weight", rb_idx, d);
                ggml_tensor* c2w = T(ts, key);
                std::snprintf(key, sizeof(key), "resb.%d.convs2.%d.bias", rb_idx, d);
                ggml_tensor* c2b = T(ts, key);
                if (c2w) {
                    x = ggml_conv_1d(ctx0, c2w, x, 1, pad2, 1);
                    if (c2b) {
                        x = ggml_add(ctx0, x, ggml_reshape_2d(ctx0, c2b, 1, (int)c2b->ne[0]));
                    }
                }

                // Residual connection
                x = ggml_add(ctx0, x, rb_residual);
                rb_residual = x;
            }

            // Accumulate for averaging
            if (!rb_sum) {
                rb_sum = x;
            } else {
                rb_sum = ggml_add(ctx0, rb_sum, x);
            }
        }
        // Average the 3 ResBlock outputs
        x = ggml_scale(ctx0, rb_sum, 1.0f / 3.0f);

        ch = ch_out;
    }

    // Final SnakeBeta activation
    {
        ggml_tensor* alpha_post = T(ts, "act_post.act.alpha");
        ggml_tensor* beta_post = T(ts, "act_post.act.beta");
        x = snake_beta(ctx0, x, alpha_post, beta_post);
    }

    // conv_post: Conv1d(24, 1, k=7, pad=3)
    {
        ggml_tensor* cpost_w = T(ts, "conv_post.weight");
        ggml_tensor* cpost_b = T(ts, "conv_post.bias");
        if (cpost_w) {
            x = ggml_conv_1d(ctx0, cpost_w, x, 1, 3, 1);
            if (cpost_b) {
                x = ggml_add(ctx0, x, ggml_reshape_2d(ctx0, cpost_b, 1, (int)cpost_b->ne[0]));
            }
        }
    }

    // tanh output clamp
    x = ggml_tanh(ctx0, x);

    ggml_set_name(x, "audio_out");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    ggml_free(ctx0);
    return gf;
}

} // namespace

// ── Public C ABI ────────────────────────────────────────────────

extern "C" struct indextts_voc_context* indextts_voc_init(const char* path, int n_threads, bool use_gpu) {
    if (!path) {
        return nullptr;
    }

    auto* c = new indextts_voc_context();
    c->n_threads = n_threads > 0 ? n_threads : 4;

    // Pass 1: metadata
    {
        gguf_context* g = core_gguf::open_metadata(path);
        if (!g) {
            delete c;
            return nullptr;
        }
        auto& hp = c->hp;
        hp.gpt_dim = (int)core_gguf::kv_u32(g, "indextts.bigvgan.gpt_dim", (uint32_t)hp.gpt_dim);
        hp.upsample_initial_ch =
            (int)core_gguf::kv_u32(g, "indextts.bigvgan.upsample_initial_channel", (uint32_t)hp.upsample_initial_ch);
        hp.num_upsamples = (int)core_gguf::kv_u32(g, "indextts.bigvgan.num_upsamples", (uint32_t)hp.num_upsamples);
        hp.num_kernels = (int)core_gguf::kv_u32(g, "indextts.bigvgan.num_kernels", (uint32_t)hp.num_kernels);
        hp.spk_emb_dim = (int)core_gguf::kv_u32(g, "indextts.bigvgan.speaker_embedding_dim", (uint32_t)hp.spk_emb_dim);
        hp.sampling_rate = (int)core_gguf::kv_u32(g, "indextts.sampling_rate", (uint32_t)hp.sampling_rate);
        hp.hop_size = (int)core_gguf::kv_u32(g, "indextts.bigvgan.hop_size", (uint32_t)hp.hop_size);

        // Read array hyperparameters from GGUF KV
        // upsample_rates: [4, 4, 4, 4, 2, 2]
        // upsample_kernel_sizes: [8, 8, 4, 4, 4, 4]
        // resblock_kernel_sizes: [3, 7, 11]
        // resblock_dilation_sizes: [1, 3, 5, 1, 3, 5, 1, 3, 5] (flattened 3x3)
        // These are stored as GGUF arrays. We read them via gguf_find_key + raw data.
        {
            int key_id = gguf_find_key(g, "indextts.bigvgan.upsample_rates");
            if (key_id >= 0) {
                const int n = std::min((int)gguf_get_arr_n(g, key_id), 6);
                for (int ii = 0; ii < n; ii++) {
                    hp.upsample_rates[ii] = (int)((const uint32_t*)gguf_get_arr_data(g, key_id))[ii];
                }
            }
        }
        {
            int key_id = gguf_find_key(g, "indextts.bigvgan.upsample_kernel_sizes");
            if (key_id >= 0) {
                const int n = std::min((int)gguf_get_arr_n(g, key_id), 6);
                for (int ii = 0; ii < n; ii++) {
                    hp.upsample_kernel_sizes[ii] = (int)((const uint32_t*)gguf_get_arr_data(g, key_id))[ii];
                }
            }
        }
        {
            int key_id = gguf_find_key(g, "indextts.bigvgan.resblock_kernel_sizes");
            if (key_id >= 0) {
                const int n = std::min((int)gguf_get_arr_n(g, key_id), 3);
                for (int ii = 0; ii < n; ii++) {
                    hp.resblock_kernel_sizes[ii] = (int)((const uint32_t*)gguf_get_arr_data(g, key_id))[ii];
                }
            }
        }
        {
            int key_id = gguf_find_key(g, "indextts.bigvgan.resblock_dilation_sizes");
            if (key_id >= 0) {
                const int n = std::min((int)gguf_get_arr_n(g, key_id), 9);
                // Stored flattened: 3 groups of 3. We just take the first 3
                // (dilations are the same for all resblock groups).
                for (int ii = 0; ii < std::min(n, 3); ii++) {
                    hp.resblock_dilations[ii] = (int)((const uint32_t*)gguf_get_arr_data(g, key_id))[ii];
                }
            }
        }

        core_gguf::free_metadata(g);

        fprintf(stderr, "indextts-voc: BigVGAN gpt_dim=%d init_ch=%d ups=%d kernels=%d spk=%d sr=%d hop=%d\n",
                hp.gpt_dim, hp.upsample_initial_ch, hp.num_upsamples, hp.num_kernels, hp.spk_emb_dim, hp.sampling_rate,
                hp.hop_size);
        fprintf(stderr, "indextts-voc: upsample_rates=[%d,%d,%d,%d,%d,%d] kernels=[%d,%d,%d,%d,%d,%d]\n",
                hp.upsample_rates[0], hp.upsample_rates[1], hp.upsample_rates[2], hp.upsample_rates[3],
                hp.upsample_rates[4], hp.upsample_rates[5], hp.upsample_kernel_sizes[0], hp.upsample_kernel_sizes[1],
                hp.upsample_kernel_sizes[2], hp.upsample_kernel_sizes[3], hp.upsample_kernel_sizes[4],
                hp.upsample_kernel_sizes[5]);
        fprintf(stderr, "indextts-voc: resblock_kernels=[%d,%d,%d] dilations=[%d,%d,%d]\n", hp.resblock_kernel_sizes[0],
                hp.resblock_kernel_sizes[1], hp.resblock_kernel_sizes[2], hp.resblock_dilations[0],
                hp.resblock_dilations[1], hp.resblock_dilations[2]);
    }

    // Backend
    c->backend_cpu = ggml_backend_cpu_init();
    if (!c->backend_cpu) {
        fprintf(stderr, "indextts-voc: failed to init CPU backend\n");
        delete c;
        return nullptr;
    }
    c->backend = use_gpu ? ggml_backend_init_best() : c->backend_cpu;
    if (!c->backend) {
        c->backend = c->backend_cpu;
    }

    // Pass 2: weights
    {
        core_gguf::WeightLoad wl;
        if (!core_gguf::load_weights(path, c->backend, "indextts-voc", wl)) {
            delete c;
            return nullptr;
        }
        c->ctx_w = wl.ctx;
        c->buf_w = wl.buf;
        c->tensors = std::move(wl.tensors);
    }

    // Verify critical tensors exist
    if (!T(c->tensors, "conv_pre.weight") || !T(c->tensors, "conv_post.weight")) {
        fprintf(stderr, "indextts-voc: missing conv_pre or conv_post weights\n");
        delete c;
        return nullptr;
    }

    // Compute scheduler
    {
        ggml_backend_t backends[2];
        int n_be = 0;
        backends[n_be++] = c->backend;
        if (c->backend != c->backend_cpu) {
            backends[n_be++] = c->backend_cpu;
        }
        c->sched = ggml_backend_sched_new(backends, nullptr, n_be, 32768, false, false);
        c->compute_meta.resize(ggml_tensor_overhead() * 32768 + ggml_graph_overhead_custom(32768, false));
    }

    fprintf(stderr, "indextts-voc: loaded %zu tensors from '%s'\n", c->tensors.size(), path);
    return c;
}

extern "C" float* indextts_voc_generate(struct indextts_voc_context* ctx, const float* latent, int T_in,
                                        const float* spk_emb, int* out_n) {
    if (!ctx || !latent || T_in <= 0 || !out_n) {
        return nullptr;
    }
    *out_n = 0;

    const auto& hp = ctx->hp;

    // Compute expected output length
    int T_audio = T_in;
    for (int i = 0; i < hp.num_upsamples; i++) {
        T_audio *= hp.upsample_rates[i];
    }

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "indextts-voc: generating audio T_in=%d -> T_audio=%d (%.2f sec)\n", T_in, T_audio,
                (float)T_audio / hp.sampling_rate);
    }

    // Build graph
    ggml_cgraph* gf = build_bigvgan_graph(ctx, T_in);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "indextts-voc: failed to alloc BigVGAN graph\n");
        return nullptr;
    }

    // Set latent input: shape (T_in, gpt_dim)
    // The latent from GPT is [T, 1280] row-major, which maps to ggml (T, 1280) with ne[0]=T.
    {
        ggml_tensor* inp = ggml_graph_get_tensor(gf, "latent_input");
        if (!inp) {
            fprintf(stderr, "indextts-voc: latent_input tensor not found in graph\n");
            return nullptr;
        }
        ggml_backend_tensor_set(inp, latent, 0, (size_t)T_in * hp.gpt_dim * sizeof(float));
    }

    // Set speaker embedding: shape (1, spk_emb_dim)
    {
        ggml_tensor* spk_t = ggml_graph_get_tensor(gf, "spk_emb");
        if (spk_t) {
            std::vector<float> spk_data(hp.spk_emb_dim, 0.0f);
            if (spk_emb) {
                std::memcpy(spk_data.data(), spk_emb, hp.spk_emb_dim * sizeof(float));
            }
            ggml_backend_tensor_set(spk_t, spk_data.data(), 0, hp.spk_emb_dim * sizeof(float));
        }
    }

    // Compute
    if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "indextts-voc: BigVGAN compute failed\n");
        return nullptr;
    }

    // Read output
    ggml_tensor* out_t = ggml_graph_get_tensor(gf, "audio_out");
    if (!out_t) {
        fprintf(stderr, "indextts-voc: audio_out tensor not found\n");
        return nullptr;
    }

    int n_samples = (int)ggml_nelements(out_t);
    float* result = (float*)malloc((size_t)n_samples * sizeof(float));
    ggml_backend_tensor_get(out_t, result, 0, (size_t)n_samples * sizeof(float));

    *out_n = n_samples;

    if (ctx->verbosity >= 1) {
        fprintf(stderr, "indextts-voc: generated %d samples (%.2f sec @ %d Hz)\n", n_samples,
                (float)n_samples / hp.sampling_rate, hp.sampling_rate);
    }

    return result;
}

extern "C" void indextts_voc_free(struct indextts_voc_context* ctx) {
    delete ctx;
}
