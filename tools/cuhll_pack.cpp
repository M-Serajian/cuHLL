// tools/cuhll_pack — offline FASTA → .cb2 converter.
//
// Reads each input FASTA via the same read_fasta_concat() cuhll uses
// internally, then writes one .cb2 per input to the chosen output directory.
// Output basename is the input's stem (no directory, no extension), with
// ".cb2" suffix.
//
// Usage:
//   cuhll_pack [-o <outdir>] <fasta1> [fasta2 ...]
//   cuhll_pack [-o <outdir>] --list <manifest>

#include "cuHLL/cb2.hpp"
#include "cuHLL/fasta.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::string stem_of(const std::string& path) {
    fs::path p(path);
    return p.stem().string();
}

void usage(const char* prog) {
    std::fprintf(stderr,
        "usage: %s [-o <outdir>] <fasta1> [fasta2 ...]\n"
        "       %s [-o <outdir>] --list <manifest>\n",
        prog, prog);
}

} // namespace

int main(int argc, char** argv) {
    std::string outdir = ".";
    std::vector<std::string> inputs;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-o" || a == "--outdir") {
            if (i + 1 >= argc) { usage(argv[0]); return 2; }
            outdir = argv[++i];
        } else if (a == "--list") {
            if (i + 1 >= argc) { usage(argv[0]); return 2; }
            std::ifstream mf(argv[++i]);
            if (!mf) { std::fprintf(stderr, "cannot read manifest\n"); return 2; }
            std::string line;
            while (std::getline(mf, line)) {
                while (!line.empty() && (line.back() == '\r' || line.back() == '\n'
                                         || line.back() == ' ' || line.back() == '\t'))
                    line.pop_back();
                if (line.empty() || line.front() == '#') continue;
                inputs.push_back(line);
            }
        } else if (a == "-h" || a == "--help") {
            usage(argv[0]); return 0;
        } else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "unknown flag: %s\n", a.c_str()); usage(argv[0]); return 2;
        } else {
            inputs.push_back(a);
        }
    }

    if (inputs.empty()) { usage(argv[0]); return 2; }

    std::error_code ec;
    fs::create_directories(outdir, ec);

    std::size_t total_in_bytes = 0;
    std::size_t total_out_bytes = 0;
    for (const auto& in : inputs) {
        try {
            std::string seq = cuhll::read_fasta_concat(in);
            const std::string base = stem_of(in);
            const std::string out = outdir + "/" + base + ".cb2";
            const std::size_t bytes = cuhll::write_cb2(out, seq.data(), seq.size());
            total_in_bytes  += seq.size();
            total_out_bytes += bytes;
            std::printf("%s -> %s  bases=%zu  out_bytes=%zu\n",
                        in.c_str(), out.c_str(), seq.size(), bytes);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[pack] ERROR on %s: %s\n", in.c_str(), e.what());
            return 1;
        }
    }
    std::printf("[pack] summary: %zu inputs  in_bases=%zu  out_bytes=%zu  ratio=%.3f\n",
                inputs.size(), total_in_bytes, total_out_bytes,
                total_in_bytes ? double(total_out_bytes) / double(total_in_bytes) : 0.0);
    return 0;
}
