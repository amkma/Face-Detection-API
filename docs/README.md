# Evidence notebook -- measurements taken through the real gate

Running `python bench/eye_bench.py --out <file>` followed by
`python bench/eye_stats.py <file>` (both under `~/venvs/face/bin/python`,
`mediapipe 1.2.2`, model sha256 `64184e22...`) gives the tables below.
The CELEBA row is a *false-reject cost*: CelebA has no eye-state label, so
a rejection there may be correct.

## Thresholds 0.18 / 0.20 (shipped)

    celebA            n=3049   p1  0.085  p5  0.177  med 0.309  p95 0.387
                               rejected 161/3049 = 5.28 %
    CEW closed        n= 400   p1  0.005  p5  0.007  med 0.034  p95 0.116
                               accepted   5/ 400 = 1.25 %

Unmeasurable (no face / same frame below corner floor): CelebA 21/3070,
CEW 0/400.

## Candidate thresholds

    t       CEW closed rejected   CelebA rejected
    0.10    91.75 %               1.41 %
    0.12    95.25 %               2.10 %
    0.15    97.50 %               3.64 %
    0.18    98.75 %               5.28 %
    0.20    99.00 %               6.76 %
    0.22    99.50 %               9.35 %
    0.25    99.75 %               17.71 %

## Where the closed-set misses sit

Five CEW faces pass `t=0.20`:  their minimum-eye EAR is
0.20--0.31. Four of them are the four `ear>=0.22` rows above; the fifth
is the `0.20--0.22` band. One face alone carries more than half of the
residual mass -- so the residual is a handful of faces, not a
distribution, and must not be read as the start of a tail.

## Open side is not graded

No eye-state label exists for CelebA in any shard fetched. The
`open_eyes` split of `MichalMlodawski/closed-open-eyes` is the same real,
unlabelled images merged from the 200k CelebA set; treating it as an
"open" class would manufacture the target. A download attempt of shards
54--132 (outside the `data/dataset_*.parquet` set) returned `MISS` for
every shard: those rows are **not** in the repository under those names
and must not be counted from a local cache.
