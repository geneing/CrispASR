// crispasr_model_mgr_cli.cpp — CLI model-resolve with TTY prompting.
//
// Delegates non-interactive resolution to the shared library; layers a
// `Download now?` prompt on top when stdin is a TTY.

#include "crispasr_model_mgr_cli.h"
#include "crispasr_cache.h"
#include "crispasr_model_registry.h"

#include <cstdio>
#include <string>

#if defined(_WIN32)
#include <io.h>
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#endif

std::string crispasr_resolve_model_cli(const std::string& model_arg, const std::string& backend_name, bool quiet,
                                       const std::string& cache_dir_override, bool auto_download) {
    // "auto"/"default" and already-on-disk paths: library handles them.
    if (model_arg == "auto" || model_arg == "default") {
        return crispasr_resolve_model(model_arg, backend_name, quiet, cache_dir_override, auto_download);
    }

    // Concrete path: check existence ourselves so we can interpose the
    // TTY prompt before asking the library to download.
    FILE* f = fopen(model_arg.c_str(), "rb");
    if (f) {
        fclose(f);
        return model_arg;
    }

    // File missing — see whether the registry recognises it.
    CrispasrRegistryEntry match;
    bool have_match = crispasr_registry_lookup_by_filename(model_arg, match);
    if (!have_match && !backend_name.empty())
        have_match = crispasr_registry_lookup(backend_name, match);

    if (!have_match) {
        // Nothing to download — return the arg and let the load layer
        // produce a real error.
        return model_arg;
    }

    fprintf(stderr, "crispasr: model '%s' not found locally.\n", model_arg.c_str());
    fprintf(stderr, "  Available for download: %s (%s)\n", match.filename.c_str(), match.approx_size.c_str());

    bool do_download = false;
    if (auto_download) {
        do_download = true;
        fprintf(stderr, "  Auto-downloading (--auto-download is set)...\n");
    } else if (isatty(fileno(stdin))) {
        fprintf(stderr, "  Download now? [Y/n] ");
        fflush(stderr);
        char c = 'y';
        int ch = fgetc(stdin);
        if (ch != EOF && ch != '\n')
            c = (char)ch;
        do_download = (c == 'y' || c == 'Y' || c == '\n');
    } else {
        fprintf(stderr, "  Use --auto-download or -m auto to download automatically.\n");
    }

    if (do_download) {
        return crispasr_cache::ensure_cached_file(match.filename, match.url, quiet, "crispasr", cache_dir_override);
    }

    return model_arg;
}
