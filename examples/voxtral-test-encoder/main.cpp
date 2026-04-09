// voxtral-test-encoder — differential test for the Voxtral audio encoder + projector.
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
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    char magic[6]; f.read(magic, 6);
    if (memcmp(magic, "\x93NUMPY", 6) != 0) return false;
    uint8_t major, minor; f.read((char*)&major, 1); f.read((char*)&minor, 1);
    uint32_t hdr_len = 0;
    if (major == 1) { uint16_t hl; f.read((char*)&hl, 2); hdr_len = hl; }
    else { f.read((char*)&hdr_len, 4); }
    std::string header(hdr_len, '\0'); f.read(&header[0], hdr_len);
    auto sp = header.find("'shape':"); auto lp = header.find('(', sp); auto rp = header.find(')', lp);
    std::string sh = header.substr(lp+1, rp-lp-1); shape.clear();
    size_t i = 0;
    while (i < sh.size()) { while (i<sh.size()&&(sh[i]==' '||sh[i]==',')) i++; if(i>=sh.size()) break;
        int v=0; while(i<sh.size()&&sh[i]>='0'&&sh[i]<='9'){v=v*10+(sh[i]-'0');i++;} shape.push_back(v); }
    size_t total = 1; for (int s : shape) total *= (size_t)s;
    data.resize(total); f.read((char*)data.data(), total * sizeof(float));
    return (bool)f;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.gguf REF_DIR\n", argv[0]); return 1; }
    // Load mel input
    std::vector<float> mel; std::vector<int> mel_shape;
    std::string ref_dir = argv[2];
    if (!load_npy_f32(ref_dir + "/mel_input.npy", mel, mel_shape)) return 2;
    fprintf(stderr, "mel: "); for (int s : mel_shape) fprintf(stderr, "%d ", s); fprintf(stderr, "\n");
    int n_mels = mel_shape.size() == 3 ? mel_shape[1] : mel_shape[0];
    int T_mel  = mel_shape.size() == 3 ? mel_shape[2] : mel_shape[1];

    // Load reference projector output
    std::vector<float> ref; std::vector<int> ref_shape;
    if (!load_npy_f32(ref_dir + "/proj2_out.npy", ref, ref_shape)) return 3;
    fprintf(stderr, "ref proj2: "); for (int s : ref_shape) fprintf(stderr, "%d ", s); fprintf(stderr, "\n");

    auto cp = voxtral_context_default_params(); cp.n_threads = 4;
    auto * ctx = voxtral_init_from_file(argv[1], cp);
    if (!ctx) { fprintf(stderr, "init failed\n"); return 4; }

    int N=0, pdim=0;
    fprintf(stderr, "running encoder ...\n");
    float * out = voxtral_run_encoder(ctx, mel.data(), n_mels, T_mel, &N, &pdim);
    if (!out) { fprintf(stderr, "encoder failed\n"); voxtral_free(ctx); return 5; }
    fprintf(stderr, "C++ encoder: N=%d pdim=%d\n", N, pdim);

    if ((size_t)N * pdim != ref.size()) {
        fprintf(stderr, "size mismatch: cpp=%d ref=%zu\n", N*pdim, ref.size());
        free(out); voxtral_free(ctx); return 6;
    }

    // Per-row cosine similarity
    double cos_sum = 0.0, cos_min = 1.0; int cos_min_i = -1;
    float max_abs = 0.0f;
    for (int i = 0; i < N; i++) {
        const float * a = out + (size_t)i * pdim;
        const float * b = ref.data() + (size_t)i * pdim;
        double dot=0,na=0,nb=0;
        for (int k = 0; k < pdim; k++) {
            dot += (double)a[k]*b[k]; na += (double)a[k]*a[k]; nb += (double)b[k]*b[k];
            float ad = std::fabs(a[k]-b[k]); if (ad > max_abs) max_abs = ad;
        }
        double cs = dot / (std::sqrt(na)*std::sqrt(nb)+1e-12);
        cos_sum += cs;
        if (cs < cos_min) { cos_min = cs; cos_min_i = i; }
    }
    fprintf(stderr, "\nDIFF vs proj2_out:\n");
    fprintf(stderr, "  max abs diff: %.4e\n", max_abs);
    fprintf(stderr, "  per-row cos sim: mean=%.6f min=%.6f (row %d)\n",
            cos_sum / N, cos_min, cos_min_i);
    int verdict = (cos_min > 0.99) ? 0 : 1;
    fprintf(stderr, "  verdict: %s\n", verdict == 0 ? "PASS (cos>0.99)" : "FAIL");

    free(out); voxtral_free(ctx);
    return verdict;
}
