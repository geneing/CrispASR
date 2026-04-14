// crispasr_aligner.cpp — implementation of crispasr_ctc_align().
//
// Two backends are supported behind one entry point:
//
//   * canary-ctc-aligner   the original NeMo FastConformer + CTC head.
//                          Selected for any aligner_model whose filename
//                          doesn't match the qwen3-fa pattern.
//
//   * qwen3-forced-aligner Qwen/Qwen3-ForcedAligner-0.6B. Same Qwen3-ASR
//                          architecture as the regular qwen3-asr backend
//                          but with a 5000-class lm_head that predicts
//                          per-token timestamps. Selected automatically
//                          when the aligner_model filename contains
//                          "forced-aligner" (case-insensitive).
//
// The dispatch is filename-based for now. A more robust GGUF arch
// detection (reading general.architecture or qwen3asr.llm.lm_head_dim)
// is a small follow-up.

#include "crispasr_aligner.h"

#include "canary_ctc.h"
#include "qwen3_asr.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

// Word tokenizer — split on ASCII whitespace. Same logic as the legacy
// per-CLI tokenise_words() helpers.
std::vector<std::string> tokenise_words(const std::string& text) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : text) {
        if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
            if (!cur.empty()) {
                out.push_back(cur);
                cur.clear();
            }
        } else {
            cur += c;
        }
    }
    if (!cur.empty())
        out.push_back(cur);
    return out;
}

bool path_contains_ci(const std::string& p, const char* needle) {
    std::string lo;
    lo.reserve(p.size());
    for (char c : p)
        lo += (char)std::tolower((unsigned char)c);
    return lo.find(needle) != std::string::npos;
}

// Qwen3-ForcedAligner forced alignment path.
std::vector<crispasr_word> align_with_qwen3_fa(const std::string& model_path, const std::vector<std::string>& words,
                                               const float* samples, int n_samples, int64_t t_offset_cs,
                                               int n_threads) {
    std::vector<crispasr_word> out;
    if (words.empty())
        return out;

    qwen3_asr_context_params cp = qwen3_asr_context_default_params();
    cp.n_threads = n_threads;
    cp.verbosity = 0;
    qwen3_asr_context* ctx = qwen3_asr_init_from_file(model_path.c_str(), cp);
    if (!ctx) {
        fprintf(stderr, "crispasr[aligner-qwen3]: failed to load '%s'\n", model_path.c_str());
        return out;
    }
    if (qwen3_asr_lm_head_dim(ctx) == (int)0 || qwen3_asr_lm_head_dim(ctx) > 10000) {
        fprintf(stderr,
                "crispasr[aligner-qwen3]: model '%s' lm_head dim is %d "
                "(expected ~5000 for forced-aligner)\n",
                model_path.c_str(), qwen3_asr_lm_head_dim(ctx));
        qwen3_asr_free(ctx);
        return out;
    }

    std::vector<const char*> word_ptrs(words.size());
    for (size_t i = 0; i < words.size(); i++)
        word_ptrs[i] = words[i].c_str();

    std::vector<int64_t> start_ms(words.size(), 0);
    std::vector<int64_t> end_ms(words.size(), 0);
    int rc = qwen3_asr_align_words(ctx, samples, n_samples, word_ptrs.data(), (int)words.size(), start_ms.data(),
                                   end_ms.data());
    qwen3_asr_free(ctx);
    if (rc != 0) {
        fprintf(stderr, "crispasr[aligner-qwen3]: align_words rc=%d\n", rc);
        return out;
    }

    out.reserve(words.size());
    for (size_t i = 0; i < words.size(); i++) {
        crispasr_word cw;
        cw.text = words[i];
        // Convert ms -> centiseconds; add the slice offset so the
        // word t0/t1 are absolute against the original audio.
        cw.t0 = t_offset_cs + start_ms[i] / 10;
        cw.t1 = t_offset_cs + end_ms[i] / 10;
        out.push_back(std::move(cw));
    }
    return out;
}

} // namespace

std::vector<crispasr_word> crispasr_ctc_align(const std::string& aligner_model, const std::string& transcript,
                                              const float* samples, int n_samples, int64_t t_offset_cs, int n_threads) {
    std::vector<crispasr_word> out;
    if (aligner_model.empty() || transcript.empty())
        return out;

    // Filename dispatch: route Qwen3 forced-aligner GGUFs through their
    // dedicated forward path (5000-class lm_head + single-pass align).
    // Anything else goes through the canary CTC path below.
    const bool is_qwen3_fa = path_contains_ci(aligner_model, "forced-aligner") ||
                             path_contains_ci(aligner_model, "qwen3-fa") ||
                             path_contains_ci(aligner_model, "qwen3-forced");
    if (is_qwen3_fa) {
        const auto words = tokenise_words(transcript);
        return align_with_qwen3_fa(aligner_model, words, samples, n_samples, t_offset_cs, n_threads);
    }

    // Load the aligner.
    canary_ctc_context_params acp = canary_ctc_context_default_params();
    acp.n_threads = n_threads;
    canary_ctc_context* actx = canary_ctc_init_from_file(aligner_model.c_str(), acp);
    if (!actx) {
        fprintf(stderr, "crispasr[aligner]: failed to load '%s'\n", aligner_model.c_str());
        return out;
    }

    // Compute CTC logits for the whole audio slice.
    float* ctc_logits = nullptr;
    int T_ctc = 0, V_ctc = 0;
    int rc = canary_ctc_compute_logits(actx, samples, n_samples, &ctc_logits, &T_ctc, &V_ctc);
    if (rc != 0) {
        fprintf(stderr, "crispasr[aligner]: compute_logits failed (rc=%d)\n", rc);
        canary_ctc_free(actx);
        return out;
    }

    // Split the transcript into whitespace-delimited words.
    const auto words = tokenise_words(transcript);
    if (words.empty()) {
        free(ctc_logits);
        canary_ctc_free(actx);
        return out;
    }

    std::vector<canary_ctc_word> aligned(words.size());
    std::vector<const char*> word_ptrs(words.size());
    for (size_t i = 0; i < words.size(); i++) {
        word_ptrs[i] = words[i].c_str();
    }

    rc = canary_ctc_align_words(actx, ctc_logits, T_ctc, V_ctc, word_ptrs.data(), (int)words.size(), aligned.data());
    free(ctc_logits);
    canary_ctc_free(actx);

    if (rc != 0) {
        fprintf(stderr, "crispasr[aligner]: align_words failed (rc=%d)\n", rc);
        return out;
    }

    out.reserve(aligned.size());
    for (const auto& w : aligned) {
        crispasr_word cw;
        cw.text = w.text;
        cw.t0 = t_offset_cs + w.t0;
        cw.t1 = t_offset_cs + w.t1;
        out.push_back(std::move(cw));
    }
    return out;
}
