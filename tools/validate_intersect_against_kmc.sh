#!/usr/bin/env bash
# tools/validate_intersect_against_kmc.sh — Layer-2 cross-implementation
# correctness check for HLL set intersection.
#
# Builds two FASTAs (or accepts user-provided ones), runs KMC3 on each
# to get exact distinct canonical k-mer sets, computes the exact
# intersection size with `kmc_tools simple ... intersect`, then runs
# cuHLL's Python intersect_estimate on the same inputs and asserts the
# relative error is within the inclusion-exclusion tolerance.
#
# Inputs (env vars):
#   CUHLL_KMC_BIN          required — path to kmc binary
#                          (load via `module load kmc/3.2.1` on HiPerGator).
#   CUHLL_KMC_TOOLS_BIN    required — path to kmc_tools binary
#                          (lives next to kmc; usually $(dirname $CUHLL_KMC_BIN)/kmc_tools).
#   CUHLL_TEST_FASTA       optional — FASTA for set A. Synthesised if unset.
#   CUHLL_TEST_FASTA2      optional — FASTA for set B. Synthesised if unset.
#   CUHLL_K                optional — k-mer length (default 31).
#   CUHLL_P                optional — HLL precision (default 16, NOT 14:
#                          intersection's inclusion-exclusion error compounds,
#                          and p=14 is too tight to validate cleanly on
#                          tiny synthetic sets).
#   CUHLL_KMC_INTERSECT_TOL  optional — relative-error tolerance (default 0.10
#                          = 10%). HLL intersection is much noisier than
#                          single-set cardinality; 5-10% is realistic at p=16.
#
# Usage:
#   module load kmc/3.2.1
#   CUHLL_KMC_BIN=$(which kmc) \
#   CUHLL_KMC_TOOLS_BIN=$(which kmc_tools) \
#       tools/validate_intersect_against_kmc.sh

set -euo pipefail

: "${CUHLL_KMC_BIN:?CUHLL_KMC_BIN must point at the kmc executable}"
: "${CUHLL_KMC_TOOLS_BIN:?CUHLL_KMC_TOOLS_BIN must point at the kmc_tools executable}"

K=${CUHLL_K:-31}
P=${CUHLL_P:-16}
TOL=${CUHLL_KMC_INTERSECT_TOL:-0.10}

WORK=$(mktemp -d -t cuhll_kmc_intersect.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
echo "[layer2/intersect] scratch: $WORK"

# --- Resolve / synthesise the two FASTAs --------------------------------------

resolve_fasta() {
    local var_name="$1"
    local out_var="$2"
    local synth_seed="$3"
    local user_path="${!var_name:-}"
    if [[ -n "$user_path" && -f "$user_path" ]]; then
        printf -v "$out_var" '%s' "$user_path"
        echo "[layer2/intersect] using $var_name = $user_path"
    else
        local path="$WORK/synth_${var_name}.fa"
        python3 - "$path" "$synth_seed" "$K" <<'PY'
import random, sys
out_path, seed, k = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
# Two synthetic FASTAs with controlled overlap. Set A = only_a + shared,
# Set B = only_b + shared. Use the seed to deterministically split a
# shared random pool: even-indexed seeds produce A, odd produce B; both
# share a third stripe of k-mers drawn from a fixed "shared" seed.
SEED_SHARED = 0xC07FEE
ONLY_N      = 30_000
SHARED_N    = 20_000
RECORD_LEN  = 80
sep = "N" * k
def draw_kmers(rng, n, taken):
    out = []
    while len(out) < n:
        kmer = "".join(rng.choices("ACGT", k=k))
        canon = min(kmer, kmer.translate(str.maketrans("ACGT","TGCA"))[::-1])
        if canon in taken:
            continue
        taken.add(canon)
        out.append(kmer)
    return out
taken = set()
rng_s = random.Random(SEED_SHARED)
shared = draw_kmers(rng_s, SHARED_N, taken)
rng_o = random.Random(seed)
only  = draw_kmers(rng_o, ONLY_N, taken)
seq = sep + sep.join(only + shared) + sep
with open(out_path, "w") as f:
    f.write(">synth\n")
    for i in range(0, len(seq), RECORD_LEN):
        f.write(seq[i:i + RECORD_LEN] + "\n")
PY
        printf -v "$out_var" '%s' "$path"
        echo "[layer2/intersect] synthesised $var_name at $path"
    fi
}

resolve_fasta CUHLL_TEST_FASTA  FASTA_A 1
resolve_fasta CUHLL_TEST_FASTA2 FASTA_B 2

# --- KMC: exact intersection size ---------------------------------------------

mkdir -p "$WORK/kmc_tmp_a" "$WORK/kmc_tmp_b"

# Build per-FASTA databases. -ci1 keeps every k-mer; -fm = multi-FASTA.
"$CUHLL_KMC_BIN" -k"$K" -ci1 -fm -t"${SLURM_CPUS_PER_TASK:-4}" \
    "$FASTA_A" "$WORK/db_a" "$WORK/kmc_tmp_a" >"$WORK/kmc_a.log" 2>&1
"$CUHLL_KMC_BIN" -k"$K" -ci1 -fm -t"${SLURM_CPUS_PER_TASK:-4}" \
    "$FASTA_B" "$WORK/db_b" "$WORK/kmc_tmp_b" >"$WORK/kmc_b.log" 2>&1

# Intersect the two DBs.
"$CUHLL_KMC_TOOLS_BIN" simple "$WORK/db_a" "$WORK/db_b" \
    intersect "$WORK/db_int" >"$WORK/kmc_intersect.log" 2>&1

# Dump the intersection DB to a flat file and count lines = exact count.
"$CUHLL_KMC_TOOLS_BIN" transform "$WORK/db_int" \
    dump "$WORK/dump_int.txt" >"$WORK/kmc_dump.log" 2>&1
EXACT_INTERSECT=$(wc -l < "$WORK/dump_int.txt")

EXACT_A=$(awk '/No\. of unique k-mers/ { print $NF; exit }' "$WORK/kmc_a.log")
EXACT_B=$(awk '/No\. of unique k-mers/ { print $NF; exit }' "$WORK/kmc_b.log")
echo "[layer2/intersect] KMC exact: |A|=$EXACT_A  |B|=$EXACT_B  |A ∩ B|=$EXACT_INTERSECT"

if [[ "$EXACT_INTERSECT" -eq 0 ]]; then
    echo "[layer2/intersect] FAIL: KMC reports zero intersection — synthetic" \
         "data wasn't built with overlap, validation impossible." >&2
    exit 1
fi

# --- cuHLL: HLL-estimated intersection ----------------------------------------

EST_INTERSECT=$(python3 - "$FASTA_A" "$FASTA_B" "$K" "$P" <<'PY'
import sys, cuhll
fa, fb, k, p = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
a = cuhll.sketch(fa, k=k, precision=p)
b = cuhll.sketch(fb, k=k, precision=p)
print(cuhll.intersect_estimate(a, b))
PY
)

# --- Compare ------------------------------------------------------------------

REL_ERR=$(python3 -c "
exact = $EXACT_INTERSECT; est = $EST_INTERSECT
print(abs(est - exact) / exact)
")

echo "[layer2/intersect] cuHLL estimate: |A ∩ B| ≈ $EST_INTERSECT  (truth $EXACT_INTERSECT, rel err $REL_ERR, tol $TOL)"

WITHIN=$(python3 -c "print(int($REL_ERR <= $TOL))")
if [[ "$WITHIN" -ne 1 ]]; then
    echo "[layer2/intersect] FAIL: relative error $REL_ERR exceeds tolerance $TOL." >&2
    echo "                  Bump CUHLL_P (currently $P) or CUHLL_KMC_INTERSECT_TOL." >&2
    exit 1
fi

echo "[layer2/intersect] PASS"
