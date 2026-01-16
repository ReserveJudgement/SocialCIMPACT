"""
Multiplayer Elo / Bradley–Terry with

    • one global latent skill μ_i,
    • per‑type deviation δ_{i,t},
    • Optional K‑scaling by match size and optional regularization,
    • joint maximum‑likelihood (L‑BFGS‑B),

**including indirect pairwise comparisons** that are built from games where two players faced the *same*
opponent set in separate matches.
"""

from __future__ import annotations
import itertools
import numpy as np
from scipy.optimize import minimize
import pandas as pd
from typing import Any, Dict, List, Tuple, Optional


# ----------------------------------------------------------------------
# 1. Helper – actual pairwise win fraction A_i from raw scores
# ----------------------------------------------------------------------
def actual_fraction(scores: np.ndarray) -> np.ndarray:
    """
    Convert a vector of raw scores for one match into the *actual*
    pair‑wise win fractions

        A_i = (higher + 0.5·equal) / (n-1)

    Parameters
    ----------
    scores : ndarray, shape (n,)
        Raw numeric scores; larger means better performance.

    Returns
    -------
    A : ndarray, shape (n,)
        Fraction of pair‑wise wins for each participant.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)

    if n < 2:
        return np.zeros(1)

    higher = (scores[:, None] > scores[None, :]).sum(axis=1).astype(float)
    equal  = ((scores[:, None] == scores[None, :]).sum(axis=1) - 1).astype(float)

    A = (higher + 0.5 * equal) / (n - 1.0)
    return A


# ----------------------------------------------------------------------
# 2. Build indirect pairwise observations
# ----------------------------------------------------------------------
def add_indirect_observations(
    data: Dict[str, Any],
    base_K: float = 1.0,
    weight_scheme: str = "constant",
) -> None:
    """
    Populate ``data['indirect']`` with extra binary comparisons that can be
    inferred when two players faced **exactly the same opponent set** in
    different matches.

    Each entry is a tuple

        (i, type_i, score_i, j, type_j, score_j, weight)

    where the outcome between i and j is derived from the two raw scores.
    The weight follows the same K‑scaling rule that is used for direct
    matches.
    """
    indirect: List[Tuple[int, int, float, int, int, float, float]] = []
    # key → list of (player_id, match_type, raw_score)
    opp_dict: Dict[frozenset, List[Tuple[int, int, float]]] = {}

    N          = data["N"]
    types      = data["type"]
    players_ls = data["players"]
    scores_ls  = data["scores"]

    for g in range(N):
        t       = types[g]
        players = players_ls[g]
        scores  = scores_ls[g]

        # For each participant record the set of his opponents
        for idx, pid in enumerate(players):
            opp_set = frozenset(np.delete(players, idx))   # all others
            if len(opp_set) == 0:
                continue
            opp_dict.setdefault(opp_set, []).append((pid, t, scores[idx]))

    # Build indirect observations from every unordered pair that shares a key
    for opp_set, entries in opp_dict.items():
        n_opp = len(opp_set)
        if n_opp == 0:
            continue

        # weight according to the chosen scaling rule (same as direct matches)
        if   weight_scheme == "sqrt":
            w = base_K / np.sqrt(n_opp)
        elif weight_scheme == "linear":
            w = base_K / float(n_opp)
        else:               # constant
            w = base_K

        for (i, ti, si), (j, tj, sj) in itertools.combinations(entries, 2):
            indirect.append((i, ti, si, j, tj, sj, w))

    data["indirect"] = indirect


# ----------------------------------------------------------------------
# 3. Core likelihood & gradient – now also consumes indirect obs.
# ----------------------------------------------------------------------
def _loglik_and_grad(
    theta: np.ndarray,
    data: Dict[str, Any],
    base_K: float = 1.0,
    weight_scheme: str = "constant",
    lam_mu: float = 0.0,
    lam_delta: float = 0.0,
) -> Tuple[float, np.ndarray]:
    """
    Return (negative log‑likelihood, gradient).  The function is written
    for ``scipy.optimize.minimize(..., jac=True)``.

    In addition to the usual multiplayer matches it also processes the
    list ``data['indirect']`` that contains binary comparisons derived
    from identical opponent sets.
    """
    P = data["num_players"]
    G = data["num_types"]

    mu    = theta[:P]                     # (P,)
    delta = theta[P:].reshape(P, G)       # (P,G)

    d_const = np.log(10.0) / 400.0        # logistic scaling
    eps = 1e-15

    loss      = 0.0
    grad_mu   = np.zeros_like(mu)
    grad_delta= np.zeros_like(delta)

    N          = data["N"]
    types      = data["type"]
    players_ls = data["players"]
    scores_ls  = data["scores"]

    # ------------------------------------------------------------------
    # 3a) Direct multiplayer matches
    # ------------------------------------------------------------------
    for g in range(N):
        t       = types[g]
        players = players_ls[g]
        scores  = scores_ls[g]
        n = len(players)

        if n < 2:
            continue

        # weight that implements the K‑scaling
        if   weight_scheme == "sqrt":
            w = base_K / np.sqrt(n - 1.0)
        elif weight_scheme == "linear":
            w = base_K / float(n - 1)
        else:               # constant
            w = base_K

        r = mu[players] + delta[players, t]

        diff = r[:, None] - r[None, :]
        p    = 1.0 / (1.0 + np.exp(-d_const * diff))
        np.fill_diagonal(p, 0.0)

        E = p.sum(axis=1) / (n - 1.0)
        A = actual_fraction(scores)

        loss -= w * np.sum(A * np.log(E + eps) +
                           (1.0 - A) * np.log(1.0 - E + eps))

        f = A / (E + eps) - (1.0 - A) / (1.0 - E + eps)
        C = d_const * p * (1.0 - p) / (n - 1.0)

        grad_vec = ((f[:, None] - f[None, :]) * C).sum(axis=1)

        grad_mu[players]       -= w * grad_vec
        grad_delta[players, t] -= w * grad_vec

    # ------------------------------------------------------------------
    # 3b) Indirect binary observations (i vs j)
    # ------------------------------------------------------------------
    for i, ti, si, j, tj, sj, w in data.get("indirect", []):
        ri = mu[i] + delta[i, ti]
        rj = mu[j] + delta[j, tj]

        diff = ri - rj
        p = 1.0 / (1.0 + np.exp(-d_const * diff))

        # outcome derived from the two raw scores
        if   si > sj: y = 1.0
        elif si == sj: y = 0.5
        else:          y = 0.0

        loss -= w * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

        # derivative of log‑likelihood wrt p
        f = y / (p + eps) - (1 - y) / (1 - p + eps)

        dp_dr_i = d_const * p * (1 - p)   # ∂p/∂ri ; note ∂p/∂rj = –dp_dr_i

        grad_mu[i]       -= w * f * dp_dr_i
        grad_delta[i, ti] -= w * f * dp_dr_i

        grad_mu[j]       += w * f * dp_dr_i   # opposite sign for the opponent
        grad_delta[j, tj] += w * f * dp_dr_i

    # ------------------------------------------------------------------
    # 4) L2 regularisation (optional)
    # ------------------------------------------------------------------
    loss += 0.5 * lam_mu   * np.sum(mu**2) + \
            0.5 * lam_delta* np.sum(delta**2)

    grad_mu    += lam_mu   * mu
    grad_delta += lam_delta* delta

    grad = np.concatenate([grad_mu, grad_delta.ravel()])
    return loss, grad


# ----------------------------------------------------------------------
# 5. Public fitting routine
# ----------------------------------------------------------------------
def fit_elo_multigame(
    data: Dict[str, Any],
    base_K: float = 1.0,
    weight_scheme: str = "constant",
    lam_mu: float = 0.0,
    lam_delta: float = 0.0,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Joint maximum‑likelihood estimate of global skill μ and per‑type
    deviation δ.  The function automatically incorporates any indirect
    observations that are present in ``data['indirect']``.
    """
    P = data["num_players"]
    G = data["num_types"]

    mu0    = np.full(P, 1500.0)
    delta0 = np.zeros((P, G))
    theta0 = np.concatenate([mu0, delta0.ravel()])

    args = (data, base_K, weight_scheme, lam_mu, lam_delta)

    result = minimize(
        fun=_loglik_and_grad,
        x0=theta0,
        args=args,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter, "ftol": tol, "disp": False},
    )

    theta_opt = result.x
    mu_hat    = theta_opt[:P]
    delta_hat = theta_opt[P:].reshape(P, G)

    return mu_hat, delta_hat, result


# ----------------------------------------------------------------------
# 6. Diagnostics
# ----------------------------------------------------------------------
def player_log_loss(
    mu: np.ndarray,
    delta: np.ndarray,
    data: Dict[str, Any],
    base_K: float = 1.0,
    weight_scheme: str = "constant",
    per_pair: bool = False,
    per_pair_raw: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a player‑wise average negative log‑likelihood.

    Parameters
    ----------
    mu, delta : rating vectors returned by ``fit_elo_multigame``.
    data      : tournament dictionary (same keys as before).
    base_K,
    weight_scheme : K‑scaling rule used during fitting.
    per_pair   : if True return the **pairwise** average (divide by the
                 *weighted* number of opponent encounters).  This is the
                 quantity that should be comparable to the global loss and
                 will typically lie in the range [0, ~1] for a well‑fitted model.
    per_pair_raw : if True return the average divided by the **raw**
                   count of opponents (the behaviour that caused the
                   inflation).  Kept only for backward compatibility.

    Returns
    -------
    avg_loss   : ndarray (P,) – the requested average loss for each player.
    denom_used : ndarray (P,) – the denominator that was used in the division
                 (useful for sanity‑checking).
    """
    P = data["num_players"]
    d_const = np.log(10.0) / 400.0
    eps = 1e-15

    total_ll        = np.zeros(P)
    weight_sum      = np.zeros(P)   # denominator for per‑match average
    pair_weight_sum = np.zeros(P)   # denominator for *weighted* per‑pair average
    raw_pair_cnt    = np.zeros(P)   # denominator for the old “raw” version

    N          = data["N"]
    types      = data["type"]
    players_ls = data["players"]
    scores_ls  = data["scores"]

    for g in range(N):
        t       = types[g]
        players = players_ls[g]
        scores  = scores_ls[g]
        n = len(players)

        if n < 2:
            continue

        # ---- match weight (same K‑scaling as used during fitting) ----
        if   weight_scheme == "sqrt":
            w = base_K / np.sqrt(n - 1.0)
        elif weight_scheme == "linear":
            w = base_K / float(n - 1)
        else:               # constant
            w = base_K

        r = mu[players] + delta[players, t]

        diff = r[:, None] - r[None, :]
        p    = 1.0 / (1.0 + np.exp(-d_const * diff))
        np.fill_diagonal(p, 0.0)

        E = p.sum(axis=1) / (n - 1.0)
        A = actual_fraction(scores)

        ll_i = -(A * np.log(E + eps) +
                 (1.0 - A) * np.log(1.0 - E + eps))

        # ----- accumulate the three possible denominators -----
        total_ll[players]       += w * ll_i          # weighted loss
        weight_sum[players]     += w                # one weight per match
        pair_weight_sum[players] += w * (n-1)       # one weight per player comparison!
        raw_pair_cnt[players]   += (n - 1)           # raw number of opponent encounters

    if per_pair_raw:
        avg = np.where(raw_pair_cnt > 0,
                       total_ll / raw_pair_cnt, np.nan)
        denom = raw_pair_cnt
    elif per_pair:          # weighted pairwise average – the *correct* one
        avg = np.where(pair_weight_sum > 0,
                       total_ll / pair_weight_sum, np.nan)
        denom = pair_weight_sum
    else:                   # per‑match average (the original behaviour before we added `per_pair`)
        avg = np.where(weight_sum > 0,
                       total_ll / weight_sum, np.nan)
        denom = weight_sum

    return avg, denom


def effective_pairwise_counts(
    data: Dict[str, Any],
    weight_scheme: str = "constant",
) -> np.ndarray:
    """
    Weighted count of opponent‑encounters contributed by each player.
    The same scaling rule that is used for the likelihood can be
    chosen (default = linear).
    """
    P = data["num_players"]
    N = data["N"]
    players_ls = data["players"]

    counts = np.zeros(P)

    for g in range(N):
        n = len(players_ls[g])
        if n < 2:
            continue

        if   weight_scheme == "sqrt":
            w = 1.0 / np.sqrt(n - 1.0)
        elif weight_scheme == "linear":
            w = 1.0 / float(n - 1)
        else:
            w = 1.0

        counts[players_ls[g]] += w * (n - 1)

    return counts


def _pairwise_win_matrix(ratings: np.ndarray) -> np.ndarray:
    """
    Compute the Bradley–Terry win‑probability matrix from a 1‑D rating array.
    Uses the same logistic function that appears in the model:

        p_ij = 1 / (1 + 10^{(R_j - R_i)/400})

    Parameters
    ----------
    ratings : ndarray, shape (P,)
        Rating for each player (global or type‑specific).

    Returns
    -------
    P : ndarray, shape (P,P)
        Pairwise win probabilities; diagonal entries are set to NaN.
    """
    d_const = np.log(10.0) / 400.0
    diff = ratings[:, None] - ratings[None, :]          # R_i – R_j
    prob = 1.0 / (1.0 + np.exp(-d_const * diff))
    np.fill_diagonal(prob, np.nan)                     # we do not care about self‑vs‑self
    return prob


def prepare_data(logs):
    data = []
    for game in logs:
        agents = game["Participants"]
        for agent in list(agents.keys()):
            data.append({"game_id": game["Game_ID"],
                         "game": game["Game"],
                         "scenario": game["Scenario"],
                         "num_players": len(list(agents.keys())),
                         "agent": agent,
                         "score": game["Scores"][agent]})
    #data = pd.DataFrame(data)
    #data["z_score"] = data.groupby(["game", "num_players"])["score"].transform(lambda x: (x - x.mean()) / x.std())
    return data


def run_elo(data):
    df = pd.DataFrame(data)

    P = len(df["agent"].unique().tolist())  # number of distinct players
    G = len(df["game"].unique().tolist())  # game types
    N = len(df["game_id"].unique().tolist())  # total matches

    print(f"{P} players, {G} games, {N} matches")

    # no game-size scaling
    k = 1
    w = "constant"

    # no regularization
    lam_mu = 0 #1e-4
    lam_delta = 0 #5e-3

    data = {
        "num_players": P,
        "num_types": G,
        "N": N,
        "type": [],
        "players": [],
        "scores": [],
    }

    games = sorted(df["game"].unique().tolist())
    game_key = {g: x for x, g in enumerate(games)}
    agents = sorted(df["agent"].unique().tolist())
    agent_key = {m: x for x, m in enumerate(agents)}

    for g in df["game_id"].unique().tolist():
        df_game = df[df["game_id"] == g]
        t = game_key[df_game["game"].unique().tolist()[0]]
        #n_players = df["num_players"].unique().tolist()[0]
        player_names = sorted(df_game["agent"].unique().tolist())
        players = [agent_key[x] for x in player_names]
        scores = [df_game[df_game["agent"] == x]["score"].item() for x in player_names]
        data["type"].append(t)
        data["players"].append(players)
        data["scores"].append(scores)


    # --------------------------------------------------------------
    # Build indirect observations
    # --------------------------------------------------------------
    add_indirect_observations(data,
                              base_K=k,  # 1.0,
                              weight_scheme=w)  # "constant")

    # --------------------------------------------------------------
    # Fit the model
    # --------------------------------------------------------------
    mu_hat, delta_hat, res = fit_elo_multigame(
        data,
        base_K=k,
        weight_scheme=w,
        lam_mu=lam_mu,
        lam_delta=lam_delta,
        max_iter=300
    )
    print(f"Optimization success: {res.success}   final loss = {res.fun:.2f}")

    report = {"agent": agents, "global_elo":mu_hat.tolist()}
    for i, game in enumerate(games):
        report[game] = delta_hat[:, i].tolist()
    #report = pd.DataFrame(report)
    #report.sort_values(by="global_elo", ascending=False, inplace=True)
    return report
