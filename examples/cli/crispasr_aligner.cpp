// crispasr_aligner.cpp — implementation of crispasr_ctc_align().

#include "crispasr_aligner.h"

#include "canary_ctc.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

// Word tokenizer — split on ASCII whitespace. Same logic as the legacy
// per-CLI tokenise_words() helpers.
std::vector<std::string> tokenise_words(const std::string & text) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : text) {
        if (c == ' ' || c == '\n' || c == '\t' || c == '\r') {
            if (!cur.empty()) { out.push_back(cur); cur.clear(); }
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

} // namespace

std::vector<crispasr_word> crispasr_ctc_align(
    const std::string & aligner_model,
    const std::string & transcript,
    const float * samples, int n_samples,
    int64_t t_offset_cs,
    int n_threads)
{
    std::vector<crispasr_word> out;
    if (aligner_model.empty() || transcript.empty()) return out;

    // Load the aligner.
    canary_ctc_context_params acp = canary_ctc_context_default_params();
    acp.n_threads = n_threads;
    canary_ctc_context * actx = canary_ctc_init_from_file(aligner_model.c_str(), acp);
    if (!actx) {
        fprintf(stderr, "crispasr[aligner]: failed to load '%s'\n",
                aligner_model.c_str());
        return out;
    }

    // Compute CTC logits for the whole audio slice.
    float * ctc_logits = nullptr;
    int T_ctc = 0, V_ctc = 0;
    int rc = canary_ctc_compute_logits(actx, samples, n_samples,
                                       &ctc_logits, &T_ctc, &V_ctc);
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

    std::vector<canary_ctc_word>  aligned(words.size());
    std::vector<const char *>     word_ptrs(words.size());
    for (size_t i = 0; i < words.size(); i++) {
        word_ptrs[i] = words[i].c_str();
    }

    rc = canary_ctc_align_words(actx, ctc_logits, T_ctc, V_ctc,
                                word_ptrs.data(), (int)words.size(),
                                aligned.data());
    free(ctc_logits);
    canary_ctc_free(actx);

    if (rc != 0) {
        fprintf(stderr, "crispasr[aligner]: align_words failed (rc=%d)\n", rc);
        return out;
    }

    out.reserve(aligned.size());
    for (const auto & w : aligned) {
        crispasr_word cw;
        cw.text = w.text;
        cw.t0   = t_offset_cs + w.t0;
        cw.t1   = t_offset_cs + w.t1;
        out.push_back(std::move(cw));
    }
    return out;
}
