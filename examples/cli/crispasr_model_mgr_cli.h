// crispasr_model_mgr_cli.h — CLI-side model resolution.
//
// Thin wrapper over `src/crispasr_model_registry.h`. Adds interactive
// TTY prompting — "Download now? [Y/n]" — when the user points at a
// recognised filename that isn't cached yet and didn't pass
// --auto-download. The library itself stays non-interactive.

#pragma once

#include <string>

/// Resolve a user-supplied -m argument to a concrete file path.
///
/// If `model_arg` is "auto" / "default" the backend's canonical GGUF is
/// resolved from the library registry. If `model_arg` points at a file
/// not on disk and matches a registry entry, we either download
/// automatically (when `auto_download`) or prompt on a TTY. When stdin
/// is not a TTY and `auto_download` is false, we pass `model_arg`
/// through unchanged so the caller gets a clear load-time error.
std::string crispasr_resolve_model_cli(const std::string& model_arg, const std::string& backend_name, bool quiet,
                                       const std::string& cache_dir_override = "", bool auto_download = false);
