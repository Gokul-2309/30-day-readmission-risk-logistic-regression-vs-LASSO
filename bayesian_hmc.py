"""
Project 2 (NumPy version): Bayesian hierarchical logistic regression for 30-day readmission.

Same data, cleaning and model as the PyMC version, but the sampler (Hamiltonian Monte Carlo)
is written by hand with NumPy. It needs only numpy, scipy, pandas, scikit-learn and matplotlib,
so it runs in a normal Anaconda install. Keep this file in the same folder as
readmission_glm_lasso.py (the cleaning function is imported from there).

Model
    readmit_i ~ Bernoulli( sigmoid( alpha[specialty_i] + x_i . beta ) )
    alpha[s]  = mu_a + sigma_a * z[s]          (partial pooling across specialties)
    z[s] ~ Normal(0, 1)   mu_a ~ Normal(-2.3, 1)   sigma_a ~ HalfNormal(0.5)   beta_j ~ Normal(0, 1)

Run:
    python bayesian_hmc.py --data data/diabetic_data.csv --out outputs_bayes
Quick test (a few minutes):
    python bayesian_hmc.py --data data/diabetic_data.csv --out outputs_bayes --n 5000 --draws 300 --tune 300 --chains 2
"""
import argparse
import os
import time
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.special import expit
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from readmission_glm_lasso import NUMERIC, load_and_clean

RANDOM_STATE = 42
GROUP_COL = "specialty_group"
CAT_FIXED = ["discharge_group", "diabetesMed", "diag_group"]

# priors
MU_A_MEAN, MU_A_SD = -2.3, 1.0   # overall baseline log-odds (about 9% risk)
SIGMA_A_SD = 0.5                 # HalfNormal scale for the spread between specialties
BETA_SD = 1.0                    # Normal(0, 1) on standardised predictors


# ----------------------------------------------------------------------------------------
# 1. Data preparation
# ----------------------------------------------------------------------------------------
def prepare_data(df, seed=RANDOM_STATE):
    groups = sorted(df[GROUP_COL].unique())
    g_all = df[GROUP_COL].map({g: i for i, g in enumerate(groups)}).values.astype(int)

    X = pd.get_dummies(df[NUMERIC + CAT_FIXED], columns=CAT_FIXED, drop_first=True).astype(float)
    y = df["readmit_30"].values.astype(float)

    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)

    # centre every column with TRAIN means (so alpha = log-odds for an average patient),
    # and divide the numeric columns by their TRAIN standard deviation
    mean = X.iloc[tr].mean()
    sd = pd.Series(1.0, index=X.columns)
    sd[NUMERIC] = X.iloc[tr][NUMERIC].std().replace(0, 1)
    Xs = (X - mean) / sd

    return dict(
        groups=groups, features=list(X.columns),
        X_tr=Xs.iloc[tr].values, X_te=Xs.iloc[te].values,
        y_tr=y[tr], y_te=y[te], g_tr=g_all[tr], g_te=g_all[te],
    )


# ----------------------------------------------------------------------------------------
# 2. The model: log posterior and its gradient
#    parameter vector q = [mu_a, log(sigma_a), z_1..z_G, beta_1..beta_p]
# ----------------------------------------------------------------------------------------
class HierLogit:
    def __init__(self, X, y, g, n_groups):
        self.X = np.ascontiguousarray(X)
        self.XT = np.ascontiguousarray(X.T)
        self.y = y
        self.g = g
        self.G = n_groups
        self.p = X.shape[1]
        self.dim = 2 + n_groups + self.p

    def logp_grad(self, q):
        G = self.G
        mu_a, s = q[0], q[1]
        z = q[2:2 + G]
        beta = q[2 + G:]
        sigma = np.exp(s)

        with np.errstate(all="ignore"):
            eta = mu_a + sigma * z[self.g] + self.X @ beta
            loglik = self.y @ eta - np.logaddexp(0.0, eta).sum()
            r = self.y - expit(eta)          # y - p, the core of every gradient below

            logp = (
                loglik
                - 0.5 * ((mu_a - MU_A_MEAN) / MU_A_SD) ** 2
                - 0.5 * (z @ z)
                - 0.5 * (beta @ beta) / BETA_SD ** 2
                - 0.5 * (sigma / SIGMA_A_SD) ** 2 + s    # half-normal prior + log-Jacobian of sigma=exp(s)
            )

            grad = np.empty_like(q)
            grad[0] = r.sum() - (mu_a - MU_A_MEAN) / MU_A_SD ** 2
            grad[1] = sigma * (r @ z[self.g]) - (sigma / SIGMA_A_SD) ** 2 + 1.0
            grad[2:2 + G] = sigma * np.bincount(self.g, weights=r, minlength=G) - z
            grad[2 + G:] = self.XT @ r - beta / BETA_SD ** 2
        return logp, grad


# ----------------------------------------------------------------------------------------
# 3. Hamiltonian Monte Carlo
# ----------------------------------------------------------------------------------------
def hmc_step(model, q, logp, grad, eps, n_steps, chol, rng):
    """One HMC transition. `chol` is the Cholesky factor of the estimated posterior
    covariance (the 'mass matrix' is its inverse). Returns the new state, the acceptance
    probability and whether the trajectory diverged."""
    w = rng.standard_normal(q.size)
    p = solve_triangular(chol, w, trans="T", lower=True)     # momentum ~ Normal(0, Sigma^-1)
    H0 = -logp + 0.5 * (w @ w)

    q_new = q.copy()
    p_new = p + 0.5 * eps * grad
    logp_new, grad_new = logp, grad
    ok = True
    for i in range(n_steps):
        q_new = q_new + eps * (chol @ (chol.T @ p_new))
        logp_new, grad_new = model.logp_grad(q_new)
        if not np.isfinite(logp_new) or not np.all(np.isfinite(grad_new)):
            ok = False
            break
        if i < n_steps - 1:
            p_new = p_new + eps * grad_new

    if not ok:
        return q, logp, grad, 0.0, True

    p_new = p_new + 0.5 * eps * grad_new
    kin = chol.T @ p_new
    H1 = -logp_new + 0.5 * (kin @ kin)
    dH = H1 - H0
    accept_prob = 1.0 if dH <= 0 else float(np.exp(-dH))
    divergent = bool(dH > 1000)
    if rng.random() < accept_prob:
        return q_new, logp_new, grad_new, accept_prob, divergent
    return q, logp, grad, accept_prob, divergent


def find_reasonable_eps(model, q, logp, grad, chol, rng, eps=0.05):
    """Double or halve the step size until a single step has about 50% acceptance."""
    def acc(e):
        _, _, _, a, _ = hmc_step(model, q, logp, grad, e, 1, chol, rng)
        return a

    direction = 1 if acc(eps) > 0.5 else -1
    for _ in range(40):
        eps_new = eps * (2.0 ** direction)
        a = acc(eps_new)
        eps = eps_new
        if (direction == 1 and a < 0.5) or (direction == -1 and a > 0.5):
            break
    return float(np.clip(eps, 1e-4, 5.0))


class DualAveraging:
    """Step-size adaptation (Hoffman and Gelman, 2014) towards a target acceptance rate."""
    def __init__(self, eps, delta=0.8, gamma=0.05, t0=10, kappa=0.75):
        self.delta, self.gamma, self.t0, self.kappa = delta, gamma, t0, kappa
        self.restart(eps)

    def restart(self, eps):
        self.mu = np.log(10.0 * eps)
        self.m = 0
        self.Hbar = 0.0
        self.logeps_bar = np.log(eps)
        self.eps = eps

    def update(self, accept):
        self.m += 1
        w = 1.0 / (self.m + self.t0)
        self.Hbar = (1 - w) * self.Hbar + w * (self.delta - accept)
        logeps = self.mu - np.sqrt(self.m) / self.gamma * self.Hbar
        eta = self.m ** (-self.kappa)
        self.logeps_bar = eta * logeps + (1 - eta) * self.logeps_bar
        self.eps = float(np.exp(logeps))

    @property
    def final_eps(self):
        return float(np.exp(self.logeps_bar))


def window_ends(T):
    """Iterations at which the covariance estimate is updated during warm-up."""
    if T < 100:
        return [], 0, 0
    init = int(0.15 * T)
    term = int(0.10 * T)
    win = max(25, int(0.05 * T))
    ends, start = [], init
    while start + win <= T - term:
        end = start + win
        if (T - term) - end < 2 * win:
            end = T - term
        ends.append(end)
        start = end
        win *= 2
    return ends, init, term


def estimate_chol(draws):
    n, dim = draws.shape
    cov = np.cov(draws, rowvar=False)
    lam = dim / (n + dim)                       # shrink towards the diagonal when draws are few
    cov = (1 - lam) * cov + lam * np.diag(np.diag(cov))
    cov = cov + 1e-6 * np.eye(dim)
    jitter = 0.0
    for _ in range(10):
        try:
            return np.linalg.cholesky(cov + jitter * np.eye(dim))
        except np.linalg.LinAlgError:
            jitter = 1e-6 if jitter == 0 else jitter * 10
    return np.eye(dim)


def run_chain(model, q0, n_tune, n_draws, n_leapfrog, rng, label, target_accept=0.95):
    dim = q0.size
    chol = np.eye(dim)
    q = q0.copy()
    logp, grad = model.logp_grad(q)

    eps0 = find_reasonable_eps(model, q, logp, grad, chol, rng)
    da = DualAveraging(eps0, delta=target_accept)
    ends, init_buf, term_buf = window_ends(n_tune)
    buf = []

    draws = np.empty((n_draws, dim))
    acc_post, n_div = [], 0
    recent = []
    t0 = time.time()
    total = n_tune + n_draws

    for it in range(total):
        warm = it < n_tune
        eps = da.eps if warm else da.final_eps
        jitter = 1.0 + 0.1 * (2.0 * rng.random() - 1.0)
        L = int(rng.integers(max(1, n_leapfrog // 2), n_leapfrog + n_leapfrog // 2 + 1))
        q, logp, grad, a, div = hmc_step(model, q, logp, grad, eps * jitter, L, chol, rng)
        recent.append(a)

        if warm:
            da.update(a)
            if ends and init_buf <= it < n_tune - term_buf:
                buf.append(q.copy())
            if (it + 1) in ends:
                chol = estimate_chol(np.array(buf))
                buf = []
                eps_new = find_reasonable_eps(model, q, logp, grad, chol, rng, eps=da.eps)
                da.restart(eps_new)
        else:
            draws[it - n_tune] = q
            acc_post.append(a)
            n_div += int(div)

        if (it + 1) % 250 == 0 or it + 1 == total:
            phase = "warm-up" if warm else "sampling"
            print(f"  {label} iter {it + 1}/{total} ({phase})  step size {eps:.4f}  "
                  f"recent acceptance {np.mean(recent[-250:]):.2f}  elapsed {time.time() - t0:.0f}s", flush=True)

    return draws, float(np.mean(acc_post)) if acc_post else np.nan, n_div, da.final_eps


# ----------------------------------------------------------------------------------------
# 4. Diagnostics: split R-hat and effective sample size
# ----------------------------------------------------------------------------------------
def _split(x):
    c, n = x.shape
    half = n // 2
    return np.concatenate([x[:, :half], x[:, half:2 * half]], axis=0)


def split_rhat(x):
    xs = _split(x)
    m, n = xs.shape
    W = xs.var(axis=1, ddof=1).mean()
    B = n * xs.mean(axis=1).var(ddof=1)
    if W <= 0:
        return np.nan
    var_plus = (n - 1) / n * W + B / n
    return float(np.sqrt(var_plus / W))


def effective_sample_size(x):
    xs = _split(x)
    m, n = xs.shape
    xc = xs - xs.mean(axis=1, keepdims=True)
    f = np.fft.rfft(xc, n=2 * n, axis=1)
    acov = np.fft.irfft(f * np.conj(f), n=2 * n, axis=1)[:, :n] / n
    W = acov[:, 0].mean() * n / (n - 1)
    B = n * xs.mean(axis=1).var(ddof=1)
    var_plus = (n - 1) / n * W + B / n
    if var_plus <= 0:
        return np.nan
    rho = 1.0 - (W - acov.mean(axis=0)) / var_plus
    rho[0] = 1.0
    pairs = rho[: (n // 2) * 2].reshape(-1, 2).sum(axis=1)
    total, prev = 0.0, np.inf
    for P in pairs:
        if P < 0:
            break
        P = min(P, prev)
        prev = P
        total += P
    tau = max(-1.0 + 2.0 * total, 1.0 / np.log10(m * n))
    return float(m * n / tau)


# ----------------------------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------------------------
def main(args):
    os.makedirs(args.out, exist_ok=True)
    rng_master = np.random.default_rng(args.seed)

    df, _ = load_and_clean(args.data)
    if args.n and args.n < len(df):
        df = df.sample(args.n, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"Patients used: {len(df)}, readmission rate: {df['readmit_30'].mean():.3f}", flush=True)

    d = prepare_data(df)
    groups, features = d["groups"], d["features"]
    G, p = len(groups), len(features)
    model = HierLogit(d["X_tr"], d["y_tr"], d["g_tr"], G)
    print(f"Training patients: {len(d['y_tr'])}, specialties: {G}, patient-level predictors: {p}", flush=True)

    # ---------- sampling ----------
    all_chains, accs, divs, epss = [], [], 0, []
    t_start = time.time()
    for c in range(args.chains):
        rng = np.random.default_rng(rng_master.integers(1 << 31))
        q0 = np.concatenate([
            [MU_A_MEAN + 0.1 * rng.standard_normal(), np.log(0.3) + 0.1 * rng.standard_normal()],
            0.1 * rng.standard_normal(G + p),
        ])
        draws, acc, nd, eps = run_chain(model, q0, args.tune, args.draws, args.leapfrog, rng,
                                        f"chain {c + 1}/{args.chains}", args.target_accept)
        all_chains.append(draws)
        accs.append(acc)
        divs += nd
        epss.append(eps)
    Q = np.stack(all_chains)                               # chains x draws x dim
    print(f"Sampling finished in {(time.time() - t_start) / 60:.1f} minutes", flush=True)

    # ---------- transform to the parameters we care about ----------
    mu_a = Q[:, :, 0]
    sigma_a = np.exp(Q[:, :, 1])
    z = Q[:, :, 2:2 + G]
    beta = Q[:, :, 2 + G:]
    alpha = mu_a[:, :, None] + sigma_a[:, :, None] * z

    names, arrays = ["mu_a", "sigma_a"], [mu_a, sigma_a]
    for i, gname in enumerate(groups):
        names.append(f"alpha[{gname}]")
        arrays.append(alpha[:, :, i])
    for j, fname in enumerate(features):
        names.append(f"beta[{fname}]")
        arrays.append(beta[:, :, j])

    rows = []
    for nm, arr in zip(names, arrays):
        flat = arr.ravel()
        rows.append({
            "parameter": nm, "mean": flat.mean(), "sd": flat.std(ddof=1),
            "q3": np.percentile(flat, 3), "q97": np.percentile(flat, 97),
            "r_hat": split_rhat(arr), "ess": effective_sample_size(arr),
        })
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(args.out, "posterior_summary.csv"), index=False)

    rhat_max = float(np.nanmax(summ["r_hat"]))
    ess_min = float(np.nanmin(summ["ess"]))
    diag = (
        f"max r_hat: {rhat_max:.3f} (want < 1.01)\n"
        f"min effective sample size: {ess_min:.0f} (want > 400)\n"
        f"divergences: {divs} (want 0)\n"
        f"mean acceptance rate per chain: {', '.join(f'{a:.2f}' for a in accs)} (at least the target {args.target_accept})\n"
    )
    total_draws = args.chains * args.draws
    if rhat_max >= 1.05 or ess_min <= 100 or divs > 0.01 * total_draws:
        verdict = ("Verdict: NOT reliable yet. Run longer (--draws 2000 --tune 2000), raise --target-accept to 0.97 "
                   "or try --leapfrog 30, and share this file with me.")
    elif rhat_max < 1.01 and ess_min > 400 and divs == 0:
        verdict = "Verdict: sampling looks good."
    else:
        why = []
        if rhat_max >= 1.01 or ess_min <= 400:
            why.append("R-hat or effective sample size is a little short of the targets")
        if divs > 0:
            why.append(f"there were {divs} divergent steps out of {total_draws} draws (a handful is usually harmless)")
        verdict = ("Verdict: usable, but " + " and ".join(why) + ". "
                   "For final numbers run longer: --draws 2000 --tune 2000 --target-accept 0.97.")
    print(diag + verdict, flush=True)
    with open(os.path.join(args.out, "diagnostics.txt"), "w") as f:
        f.write(diag + verdict + "\n")

    # ---------- trace plot ----------
    fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for ax, nm, arr in zip(axs, ["mu_a", "sigma_a"], [mu_a, sigma_a]):
        for c in range(arr.shape[0]):
            ax.plot(arr[c], lw=0.6, alpha=0.8)
        ax.set_ylabel(nm)
    axs[-1].set_xlabel("Draw (each colour = one chain; should look like a fuzzy caterpillar)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "trace.png"), dpi=130)
    plt.close(fig)

    # ---------- flatten draws ----------
    alpha_s = alpha.reshape(-1, G)
    beta_s = beta.reshape(-1, p)
    mu_a_s = mu_a.ravel()
    sigma_a_s = sigma_a.ravel()

    # ---------- specialty table ----------
    g_tr, y_tr = d["g_tr"], d["y_tr"]
    tab = pd.DataFrame({"group": groups})
    tab["n_train"] = [int((g_tr == i).sum()) for i in range(G)]
    tab["raw_readmit_rate"] = [float(y_tr[g_tr == i].mean()) if (g_tr == i).any() else np.nan for i in range(G)]
    tab["alpha_mean"] = alpha_s.mean(axis=0)
    tab["alpha_low_3pct"] = np.percentile(alpha_s, 3, axis=0)
    tab["alpha_high_97pct"] = np.percentile(alpha_s, 97, axis=0)
    tab["baseline_risk_at_avg_patient"] = expit(alpha_s).mean(axis=0)
    tab.sort_values("alpha_mean").to_csv(os.path.join(args.out, "specialty_baselines.csv"), index=False)

    # ---------- test-set performance (posterior predictive average) ----------
    sub = np.linspace(0, len(alpha_s) - 1, min(300, len(alpha_s))).astype(int)
    eta_te = alpha_s[sub][:, d["g_te"]] + beta_s[sub] @ d["X_te"].T
    p_te = expit(eta_te).mean(axis=0)
    res = pd.DataFrame({
        "model": ["Bayesian hierarchical logistic"],
        "test_AUC": [roc_auc_score(d["y_te"], p_te)],
        "test_Brier": [brier_score_loss(d["y_te"], p_te)],
    })
    res.to_csv(os.path.join(args.out, "model_performance.csv"), index=False)
    print(res.to_string(index=False))
    print(f"sigma_a (spread between specialties): mean {sigma_a_s.mean():.3f}, 94% interval "
          f"[{np.percentile(sigma_a_s, 3):.3f}, {np.percentile(sigma_a_s, 97):.3f}]", flush=True)

    # ---------- plot 1: specialty baselines ----------
    order = np.argsort(tab["alpha_mean"].values)
    plt.figure(figsize=(7, 4.5))
    for k, i in enumerate(order):
        plt.plot([tab.loc[i, "alpha_low_3pct"], tab.loc[i, "alpha_high_97pct"]], [k, k], color="tab:blue", lw=2)
        plt.plot(tab.loc[i, "alpha_mean"], k, "o", color="tab:blue")
    plt.axvline(float(mu_a_s.mean()), color="k", ls="--", lw=0.8, label="overall mean baseline")
    plt.yticks(np.arange(G), [f"{groups[i]} (n={tab.loc[i, 'n_train']})" for i in order])
    plt.xlabel("Baseline log-odds, average patient (94% interval)")
    plt.title("Specialty baselines (partial pooling)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "specialty_baselines.png"), dpi=140, bbox_inches="tight")
    plt.close()

    # ---------- plot 2: patient-level odds ratios ----------
    orr = np.exp(beta_s)
    med = np.median(orr, axis=0)
    lo = np.percentile(orr, 3, axis=0)
    hi = np.percentile(orr, 97, axis=0)
    feat = np.array(features)
    top = np.argsort(-np.abs(np.log(med)))[:12][::-1]
    plt.figure(figsize=(7.5, 5))
    yp = np.arange(len(top))
    plt.errorbar(med[top], yp, xerr=[med[top] - lo[top], hi[top] - med[top]], fmt="o", capsize=3)
    plt.axvline(1, color="k", ls="--", lw=0.8)
    plt.yticks(yp, feat[top])
    plt.xlabel("Odds ratio (posterior median, 94% interval)\nNumeric variables are per 1 SD")
    plt.title("Bayesian odds ratios: strongest patient-level factors")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "bayes_odds_ratios.png"), dpi=140, bbox_inches="tight")
    plt.close()
    pd.DataFrame({"feature": feat, "or_median": med, "or_low_3pct": lo, "or_high_97pct": hi}) \
        .sort_values("or_median", ascending=False) \
        .to_csv(os.path.join(args.out, "bayes_odds_ratios.csv"), index=False)

    # ---------- plot 3: how much do specialties differ? ----------
    plt.figure(figsize=(5.5, 4))
    plt.hist(sigma_a_s, bins=40, color="tab:blue", alpha=0.8)
    plt.xlabel("sigma_a: spread of specialty baselines (log-odds)")
    plt.ylabel("Posterior draws")
    plt.title("Posterior of between-specialty spread")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "sigma_posterior.png"), dpi=140)
    plt.close()

    print(f"Done. Files saved in {args.out}/", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/diabetic_data.csv")
    ap.add_argument("--out", default="outputs_bayes")
    ap.add_argument("--n", type=int, default=20000, help="patients to sample (0 = use all)")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--leapfrog", type=int, default=15, help="average leapfrog steps per iteration")
    ap.add_argument("--target-accept", dest="target_accept", type=float, default=0.95,
                    help="higher (0.97) = smaller steps, fewer divergences, slower")
    ap.add_argument("--seed", type=int, default=1)
    main(ap.parse_known_args()[0])
