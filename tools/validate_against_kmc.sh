#!/usr/bin/env bash
# tools/validate_against_kmc.sh — Layer 2 cross-implementation correctness
# check. Runs cuHLL and KMC3 on the same FASTA and asserts the HLL estimate
# is within tolerance of KMC's exact distinct-k-mer count.
#
# Inputs (env vars):
#   CUHLL_BIN          required — path to the cuhll executable
#   CUHLL_KMC_BIN      required — path to the kmc executable
#   CUHLL_TEST_FASTA   optional — if set and the file exists, use it;
#                                 otherwise synthesise a small ACGT FASTA
#   CUHLL_K            optional — k-mer length (default 31)
#   CUHLL_P            optional — HLL precision (default 14)
#   CUHLL_KMC_TOL      optional — relative-error tolerance (default 0.05 = 5%)
#
# Wiring: tests/CMakeLists.txt registers this as the `test_against_kmc`
# CTest entry and sets CUHLL_BIN / CUHLL_KMC_BIN automatically. To run
# standalone:
#   CUHLL_BIN=build/bin/cuhll CUHLL_KMC_BIN=$(which kmc) \
#       tools/validate_against_kmc.sh

set -euo pipefail

: "${CUHLL_BIN:?CUHLL_BIN must point at the cuhll executable}"
: "${CUHLL_KMC_BIN:?CUHLL_KMC_BIN must point at the kmc executable}"

K=${CUHLL_K:-31}
P=${CUHLL_P:-14}
TOL=${CUHLL_KMC_TOL:-0.05}

WORK=$(mktemp -d -t cuhll_kmc.XXXXXX)
trap 'rm -rf "$WORK"' EXIT

# Resolve the FASTA: user-provided wins, otherwise synthesise.
if [[ -n "${CUHLL_TEST_FASTA:-}" && -f "${CUHLL_TEST_FASTA}" ]]; then
    FASTA="$CUHLL_TEST_FASTA"
    echo "[layer2] using user-provided FASTA: $FASTA"
else
    FASTA="$WORK/synth.fa"
    if ! command -v python3 >/dev/null 2>&1; then
        echo "[layer2] FAIL: python3 is needed to synthesise the test FASTA" >&2
        echo "         set CUHLL_TEST_FASTA to a real FASTA to skip synthesis" >&2
        exit 1
    fi
    python3 - "$FASTA" <<'PY'
import random, sys
out_path = sys.argv[1]
random.seed(0xC0FFEE)
n_records   = 4
record_len  = 500_000
with open(out_path, "w") as f:
    for r in range(n_records):
        f.write(f">rec{r}\n")
        seq = "".join(random.choices("ACGT", k=record_len))
        for i in range(0, len(seq), 80):
            f.write(seq[i:i+80] + "\n")
PY
    echo "[layer2] synthesised FASTA at $FASTA (4 records, 500 kbp each)"
fi

# KMC3: count distinct canonical k-mers exactly. KMC writes its banner
# and stats to stdout/stderr; we capture them in kmc.log and parse the
# "No. of unique k-mers" line. If KMC itself fails, dump the log so the
# CTest output makes the failure mode obvious.
if ! "$CUHLL_KMC_BIN" -k"$K" -ci1 -fm -t"${SLURM_CPUS_PER_TASK:-4}" \
        "$FASTA" "$WORK/kmc_db" "$WORK" >"$WORK/kmc.log" 2>&1; then
    echo "[layer2] FAIL: kmc exited non-zero" >&2
    cat "$WORK/kmc.log" >&2
    exit 1
fi
EXACT=$(awk '/No\. of unique k-mers/ { print $NF; exit }' "$WORK/kmc.log")
if [[ -z "${EXACT:-}" ]]; then
    echo "[layer2] FAIL: could not parse KMC unique-k-mer count" >&2
    cat "$WORK/kmc.log" >&2
    exit 1
fi

# cuHLL: HLL estimate. The default mode prints one integer (the union
# cardinality) on stdout when there is no --output-dir.
EST=$("$CUHLL_BIN" --k "$K" --precision "$P" "$FASTA" 2>/dev/null | tail -1)
if [[ -z "${EST:-}" ]]; then
    echo "[layer2] FAIL: cuhll produced no output" >&2
    exit 1
fi

REL_ERR=$(awk -v e="$EST" -v x="$EXACT" \
    'BEGIN { d = (e>x) ? (e-x) : (x-e); printf "%.6f", d/x }')

printf "[layer2] k=%s p=%s  kmc_exact=%s  cuhll_est=%s  rel_err=%s  tolerance=%s\n" \
    "$K" "$P" "$EXACT" "$EST" "$REL_ERR" "$TOL"

PASS=$(awk -v r="$REL_ERR" -v t="$TOL" 'BEGIN { print (r<=t ? 1 : 0) }')
if [[ "$PASS" == "1" ]]; then
    echo "[PASS] test_against_kmc"
    exit 0
fi
echo "[FAIL] test_against_kmc: relative error $REL_ERR exceeds tolerance $TOL" >&2
exit 1
