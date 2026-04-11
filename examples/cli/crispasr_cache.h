// crispasr_cache.h — shared cache directory + curl/wget download helper.
//
// Three places in the unified CLI need to download a small companion file
// from HuggingFace on first use and cache it under ~/.cache/crispasr/:
//
//   * crispasr_model_mgr — `-m auto` model resolution
//   * crispasr_lid       — whisper-tiny LID model auto-download
//   * crispasr_vad       — Silero VAD model auto-download
//
// This header centralises the directory layout, the existence/zombie
// check, and the curl-then-wget fallback so each consumer is a one-liner
// over `crispasr_cache::ensure_cached_file(...)`. No new build deps —
// the helpers shell out to whatever curl/wget the user already has on
// $PATH, identical to the original per-module copies.

#pragma once

#include <string>

namespace crispasr_cache {

// Return ~/.cache/crispasr (creating it if missing). Falls back to
// /tmp/.cache/crispasr when $HOME isn't set.
std::string dir();

// True iff `path` exists AND is non-zero bytes. Treats 0-byte zombies
// (left behind by an interrupted earlier download) as missing so the
// next attempt retries the fetch instead of handing a corrupted file
// to a model loader.
bool file_present(const std::string & path);

// Fetch `url` into `dest` via curl, falling back to wget on failure.
// Returns true iff the file is present and non-empty after the
// download. `quiet=true` suppresses progress bars; failure messages
// always go to stderr.
bool fetch(const std::string & url, const std::string & dest, bool quiet);

// Composite helper: if `dest` (= dir() + "/" + filename) already
// satisfies file_present(), return its path immediately. Otherwise
// invoke fetch() to populate it. Returns the absolute path on success
// or an empty string on failure.
std::string ensure_cached_file(const std::string & filename,
                               const std::string & url,
                               bool quiet,
                               const char * pretty_label = "crispasr");

} // namespace crispasr_cache
