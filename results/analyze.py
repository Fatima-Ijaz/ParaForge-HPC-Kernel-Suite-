"""Parse raw result files, compute medians, write plots (results/fig_*.png) and summary.json."""
import json, os, re, statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "axes.grid": True, "grid.alpha": .3})
OUT = {}

def parse(fn):
    rows = []
    if not os.path.exists(os.path.join(D, fn)):
        return rows
    for line in open(os.path.join(D, fn)):
        t = line.split()
        if not t: continue
        tag = t[0] if t[0].isupper() else ""
        if tag: t = t[1:]
        r = {"tag": tag, "kind": t[0]}
        for x in t[1:]:
            if "=" in x:
                k, v = x.split("=", 1)
                try: v = float(v)
                except ValueError: pass
                r[k] = v
            else: r["status"] = x
        rows.append(r)
    return rows

def med(rows, key="time", **f):
    v = [r[key] for r in rows if all(r.get(k) == x for k, x in f.items())]
    return st.median(v) if v else None

def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(D, name)); plt.close(fig)

# ---------------- CS1 ----------------
r1 = parse("cs1_raw.txt")
if r1:
    base = [r for r in r1 if r["tag"] == ""]
    cs1 = {"serial": {}, "strong": {}, "weak": [], "K": {}}
    P = sorted({int(r["p"]) for r in base if r["kind"] == "parallel"})
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3))
    for i, N in enumerate([1024, 4096]):
        Ts = med(base, kind="serial", N=N); cs1["serial"][N] = Ts
        for d in [1, 2]:
            for m in ["b", "nb"]:
                rows = []
                for p in P:
                    tp = med(base, kind="parallel", N=N, p=p, decomp=d, mode=m)
                    tc = med(base, "comm", kind="parallel", N=N, p=p, decomp=d, mode=m)
                    rows.append({"p": p, "T": tp, "comm": tc, "S": Ts / tp, "E": Ts / tp / p})
                cs1["strong"][f"{N}_{d}D_{m}"] = rows
                ax[i].plot(P, [x["S"] for x in rows], "o-", label=f"{d}-D {'non-blocking' if m=='nb' else 'blocking'}")
        ax[i].plot(P, P, "k:", lw=.8, label="ideal")
        ax[i].axvspan(2, max(P), color="red", alpha=.06)
        ax[i].set(title=f"CS1 strong scaling N={N}, T=500", xlabel="MPI ranks p", ylabel="speedup vs serial")
        ax[i].set_xticks(P); ax[i].set_xticklabels(P)
    ax[0].legend(fontsize=7)
    fig.text(0.5, 0.005, "median of 3 runs; shaded = oversubscribed (p > 2 physical cores)", ha="center", fontsize=7)
    save(fig, "fig_cs1_strong.png")

    wk = [r for r in r1 if r["tag"] == "WEAK"]
    T1 = med(wk, p=1)
    for p in sorted({int(r["p"]) for r in wk}):
        T = med(wk, p=p); cs1["weak"].append({"p": p, "N": int(med(wk, "N", p=p)), "T": T, "comm": med(wk, "comm", p=p), "Ew": T1 / T})
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.plot([w["p"] for w in cs1["weak"]], [w["Ew"] for w in cs1["weak"]], "o-")
    ax.axhline(1, color="k", ls=":", lw=.8); ax.axvspan(2, 8, color="red", alpha=.06)
    ax.set(title="CS1 weak scaling (1024² cells/rank, 2-D nb)", xlabel="MPI ranks p", ylabel="weak efficiency T1/Tp", ylim=(0, 1.15))
    ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels([1, 2, 4, 8])
    save(fig, "fig_cs1_weak.png")

    kt = [r for r in r1 if r["tag"] == "KTEST"]
    for N in [1024, 4096]:
        for p in [2, 4]:
            for K in [0, 1, 50]:
                cs1["K"][f"{N}_{p}_{K}"] = {"T": med(kt, N=N, p=p, K=K), "allreduce": med(kt, "allreduce", N=N, p=p, K=K)}
    OUT["cs1"] = cs1

# ---------------- CS2 ----------------
r2 = parse("cs2_raw.txt")
if r2:
    cs2 = {"serial": {}, "ver": {}, "race": [], "fs": {}}
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3))
    for i, k in enumerate([8, 12]):
        for seq in ["u", "s"]:
            Ts = med(r2, kind="serial", k=k, seq=seq); cs2["serial"][f"{k}{seq}"] = Ts
            for v in "ABCR":
                rows = []
                for t in [1, 2, 4]:
                    T = med(r2, kind="parallel", k=k, seq=seq, ver=v, threads=t)
                    if T is None: continue
                    rows.append({"t": t, "T": T, "S": Ts / T, "E": Ts / T / t,
                                 "lost": med(r2, "lost", kind="parallel", k=k, seq=seq, ver=v, threads=t),
                                 "Mk": med(r2, "Mkmers_per_s", kind="parallel", k=k, seq=seq, ver=v, threads=t),
                                 "MB": med(r2, "extra_MB", kind="parallel", k=k, seq=seq, ver=v, threads=t)})
                cs2["ver"][f"{k}{seq}{v}"] = rows
                if seq == "u" and v in "BCR":
                    ax[i].plot([x["t"] for x in rows], [x["S"] for x in rows], "o-", label={"B": "B atomic", "C": "C private", "R": "R array reduction"}[v])
        ax[i].plot([1, 2, 4], [1, 2, 4], "k:", lw=.8, label="ideal"); ax[i].axvspan(2, 4, color="red", alpha=.06)
        ax[i].set(title=f"CS2 k={k}, L=10⁸ uniform", xlabel="threads", ylabel="speedup vs serial", xticks=[1, 2, 4])
    ax[0].legend(fontsize=7)
    fig.text(0.5, 0.005, "median of 3 runs; shaded = oversubscribed (> 2 physical cores)", ha="center", fontsize=7)
    save(fig, "fig_cs2_speedup.png")
    for k in [2, 4, 6, 8, 12]:
        l = med(r2, "lost", kind="parallel", k=k, seq="u", ver="A", threads=2)
        if l is not None: cs2["race"].append({"k": k, "lost": l, "expected": med(r2, "expected", kind="parallel", k=k, seq="u", ver="A", threads=2)})
    for t in [1, 2]:
        for pad in [0, 16, 32]:
            cs2["fs"][f"{t}_{pad}"] = med(r2, kind="parallel", k=1, ver="F", pad=pad, threads=t)
        cs2["fs"][f"{t}_C"] = med(r2, kind="parallel", k=1, ver="C", threads=t)
    OUT["cs2"] = cs2

# ---------------- CS3 ----------------
r3 = parse("cs3_raw.txt")
if r3:
    base = [r for r in r3 if r["tag"] == ""]
    cs3 = {}
    Ts = med(base, kind="serial"); cs3["serial"] = Ts
    cs3["serial_row"] = [r for r in base if r["kind"] == "serial"][0]
    cs3["strong"] = [{"t": t, "T": med(base, kind="parallel", mode="good", threads=t), } for t in [1, 2, 4]]
    for x in cs3["strong"]: x["S"] = Ts / x["T"]; x["E"] = x["S"] / x["t"]
    cs3["shared"] = {t: med(base, kind="parallel", mode="shared", threads=t) for t in [1, 2]}
    cs3["good_1e7"] = med([r for r in r3 if r["tag"] == "REPRO"], threads=2)
    tr = [r for r in r3 if r["tag"] == "TRIAL"]
    for m in ["good", "sameseed"]:
        prices = [r["price"] for r in tr if r["mode"] == m]
        cs3["trial_" + m] = {"emp_std": st.stdev(prices), "mean_se": st.mean([r["se"] for r in tr if r["mode"] == m]),
                             "mean_price": st.mean(prices), "n": len(prices)}
    fx = [r for r in r3 if r["tag"] == "FIXED"]
    cs3["fixed"] = {"T": med(fx), "se": med(fx, "se"), "M": fx[0]["M"]}
    p1 = [r for r in base if r["kind"] == "parallel" and r["mode"] == "good" and r["threads"] == 1]
    cs3["p1_se"] = p1[0]["se"]; cs3["p1_price"] = p1[0]["price"]
    rp = [r for r in r3 if r["tag"] == "REPRO"]
    cs3["repro"] = [{"t": int(r["threads"]), "price": r["price"]} for r in rp]
    mp = [r for r in r3 if r["tag"] == "MPI"]
    cs3["mpi"] = {t: med(mp, ranks=t) for t in [1, 2]}
    cs3["mpi_price2"] = [r["price"] for r in mp if r["ranks"] == 2][0]
    cs3["omp_price2"] = [r["price"] for r in base if r["kind"] == "parallel" and r["mode"] == "good" and r["threads"] == 2][0]
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.plot([x["t"] for x in cs3["strong"]], [x["S"] for x in cs3["strong"]], "o-", label="OpenMP, own streams")
    ax.plot([1, 2], [Ts / cs3["mpi"][1], Ts / cs3["mpi"][2]], "s--", label="MPI (bonus)")
    ax.plot([1, 2], [Ts * 0.1 / cs3["shared"][1], Ts * 0.1 / cs3["shared"][2]], "x-", label="shared RNG + lock")
    ax.plot([1, 2, 4], [1, 2, 4], "k:", lw=.8, label="ideal"); ax.axvspan(2, 4, color="red", alpha=.06)
    ax.set(title="CS3 Monte Carlo, M=10⁸", xlabel="threads / ranks", ylabel="speedup vs serial", xticks=[1, 2, 4])
    ax.legend(fontsize=6.5)
    save(fig, "fig_cs3_speedup.png")
    OUT["cs3"] = cs3

# ---------------- CS4 ----------------
r4 = parse("cs4_raw.txt")
if r4:
    cs4 = {}
    base = [r for r in r4 if r["tag"] == ""]
    sr = [r for r in base if r["kind"] == "serial"]
    Ts = med(sr); cs4["serial"] = {"T": Ts, "iters": sr[0]["iters"], "n": sr[0]["n"], "m": sr[0]["m"], "dangling": sr[0]["dangling"], "Meps": med(sr, "Medges_per_s")}
    cs4["strong"] = {}
    for mode in ["pull", "push"]:
        rows = []
        for t in [1, 2, 4]:
            f = dict(kind="parallel", mode=mode, sched="static", chunk=0, threads=t)
            T = med(base, **f)
            rows.append({"t": t, "T": T, "S": Ts / T, "E": Ts / T / t, "R_time": med(base, "R_time", **f), "R_edges": med(base, "R_edges_static", **f),
                         "Meps": med(base, "Medges_per_s", **f), "iters": med(base, "iters", **f), "maxdiff": med(base, "ref_maxdiff", **f)})
        cs4["strong"][mode] = rows
    cs4["pushplain"] = med(base, kind="parallel", mode="pushplain", threads=1)
    cs4["sched"] = {}
    for mode in ["pull", "push"]:
        for sc, c in [("static", 0), ("static", 64), ("static", 1024), ("dynamic", 1), ("dynamic", 16), ("dynamic", 64), ("dynamic", 256), ("dynamic", 1024),
                      ("dynamic", 4096), ("dynamic", 16384), ("guided", 0), ("guided", 64), ("guided", 1024)]:
            f = dict(kind="parallel", mode=mode, sched=sc, chunk=c, threads=2)
            cs4["sched"][f"{mode}_{sc}_{c}"] = {"T": med(base, **f), "R": med(base, "R_time", **f)}
    cs4["imb"] = {int(r["threads"]): r["R_edges_static"] for r in r4 if r["tag"] == "IMB"}
    cs4["imb"][2] = cs4["strong"]["pull"][1]["R_edges"]; cs4["imb"][4] = cs4["strong"]["pull"][2]["R_edges"]
    OUT["cs4"] = cs4
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3))
    for mode in ["pull", "push"]:
        ax[0].plot([1, 2, 4], [x["S"] for x in cs4["strong"][mode]], "o-", label=f"{mode}, static")
    best = min((v["T"], k) for k, v in cs4["sched"].items() if k.startswith("pull"))
    ax[0].plot([2], [Ts / best[0]], "r*", ms=9, label="pull, best schedule")
    ax[0].plot([1, 2, 4], [1, 2, 4], "k:", lw=.8, label="ideal"); ax[0].axvspan(2, 4, color="red", alpha=.06)
    ax[0].set(title="CS4 PageRank, R-MAT 2²⁰ v / 16.7M e", xlabel="threads", ylabel="speedup vs serial", xticks=[1, 2, 4]); ax[0].legend(fontsize=7)
    dg = [l.split() for l in open(os.path.join(D, "cs4_degrees.txt")).readlines()[1:]]
    deg = [int(x[0]) for x in dg]; ci = [int(x[1]) for x in dg]; co = [int(x[2]) for x in dg]
    ax[1].loglog([d for d, c in zip(deg, ci) if c and d], [c for d, c in zip(deg, ci) if c and d], ".", ms=2, label="in-degree")
    ax[1].loglog([d for d, c in zip(deg, co) if c and d], [c for d, c in zip(deg, co) if c and d], ".", ms=2, label="out-degree")
    ax[1].set(title="Degree distribution", xlabel="degree", ylabel="number of vertices"); ax[1].legend(fontsize=7)
    fig.text(0.5, 0.005, "median of 3 runs; shaded = oversubscribed", ha="center", fontsize=7)
    save(fig, "fig_cs4.png")
    cs4["maxdeg_in"] = max(d for d, c in zip(deg, ci) if c)

# ---------------- CS5 ----------------
r5 = parse("cs5_raw.txt")
if r5:
    cs5 = {"serial": {}, "strong": {}, "os": {}}
    base = [r for r in r5 if r["tag"] == ""]
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3))
    for dist in ["u", "z"]:
        Ts = med(base, kind="serial", dist=dist); cs5["serial"][dist] = Ts
        rows = []
        for p in sorted({int(r["p"]) for r in base if r["kind"] == "parallel"}):
            f = dict(kind="parallel", dist=dist, p=p)
            row = {"p": p, "T": med(base, **f)}
            for ph in ["local_sort", "sampling", "alltoallv", "merge", "a2av_call", "max_over_mean"]: row[ph] = med(base, ph, **f)
            row["S"] = Ts / row["T"]; row["E"] = row["S"] / p
            rows.append(row)
        cs5["strong"][dist] = rows
        ax[0].plot([x["p"] for x in rows], [x["S"] for x in rows], "o-", label="uniform" if dist == "u" else "Zipf skewed")
    ax[0].plot([1, 2, 4], [1, 2, 4], "k:", lw=.8, label="ideal"); ax[0].axvspan(2, 4, color="red", alpha=.06)
    ax[0].set(title="CS5 sample sort, N=10⁸", xlabel="MPI ranks p", ylabel="speedup vs std::sort", xticks=[1, 2, 4]); ax[0].legend(fontsize=7)
    rows = cs5["strong"]["u"]; xs = range(len(rows)); bottom = [0] * len(rows)
    for ph, col in [("local_sort", "C0"), ("sampling", "C1"), ("alltoallv", "C2"), ("merge", "C3")]:
        vals = [x[ph] for x in rows]
        ax[1].bar(xs, vals, bottom=bottom, color=col, label=ph); bottom = [b + v for b, v in zip(bottom, vals)]
    ax[1].set_xticks(list(xs)); ax[1].set_xticklabels([f"p={x['p']}" for x in rows])
    ax[1].set(title="Phase breakdown (uniform)", ylabel="time (s, max over ranks)"); ax[1].legend(fontsize=7)
    fig.text(0.5, 0.005, "median of 3 runs; p=4 oversubscribed", ha="center", fontsize=7)
    save(fig, "fig_cs5.png")
    for r in r5:
        if r["tag"] == "OS": cs5["os"][f"{r['dist']}_{int(r['p'])}_{int(r['os'])}"] = {"mom": r["max_over_mean"], "counts": r["counts"], "alltoallv": r["alltoallv"]}
    pp = [tuple(map(float, l.split())) for l in open(os.path.join(D, "cs5_pingpong.txt"))]
    big = [x for x in pp if x[0] >= 65536]
    n = len(big); sx = sum(x[0] for x in big); sy = sum(x[1] for x in big)
    sxx = sum(x[0] ** 2 for x in big); sxy = sum(x[0] * x[1] for x in big)
    beta = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    cs5["alpha"] = min(x[1] for x in pp)          # latency from the smallest message
    cs5["beta"] = beta; cs5["pingpong"] = pp
    OUT["cs5"] = cs5

json.dump(OUT, open(os.path.join(D, "summary.json"), "w"), indent=1, default=str)
print("wrote summary.json and figures")
