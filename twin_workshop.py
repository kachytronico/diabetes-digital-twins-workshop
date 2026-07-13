"""
twin_workshop.py
================
Shared, tested machinery for the "From Population Models to Personalized
Digital Twins" workshop (AI in Diabetes summer school, Girona).

This module is the *complete* reference implementation. The student notebooks
import the non-pedagogical parts from here so a typo in a blank can never strand
the rest of the session. Notebook A re-implements the architecture inline (with
blanks); Notebook B imports the architecture from here and only blanks the
transfer/personalization steps.

Faithful to the author's codebase:
  data_processing.py / vae_model.py / vae_personalize.py / evaluate_holdout.py
"""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")  # route tf.keras -> Keras 2

import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (
    Dense, Dropout, LeakyReLU, Flatten, Lambda, LSTM, Reshape,
    Concatenate, TimeDistributed,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HORIZON      = 18       # BG steps predicted ahead
LATENT_DIM   = 5        # VAE latent dimensionality
HIDDEN       = 64       # dense / LSTM width in the decoder

KL_WEIGHT    = 3e-3
MONO_WEIGHT  = 5e-2
LR           = 1e-4

MIN_GAIN_MGDL = 0.3     # floor on the per-step control weight magnitude
CONTROL_SCALE = 1.0

INSULIN_UP   = 1.20     # counterfactual: +20% insulin
CARBS_UP     = 1.20     # counterfactual: +20% carbs
K_INS_100    = 40.0     # expected mg/dL swing per 100% insulin change
K_CARB_100   = 25.0     # expected mg/dL swing per 100% carb change

IN_SHAPE     = (1, 1)            # insulin / carbs : one scalar each
OUT_SHAPE    = (1, HORIZON)      # bg trajectory   : (1, 18)

SAMPLES_PER_DAY = 288
HOLDOUT_DAYS    = 3

# Data format -> (folder, extension)
FORMAT_MAP = {
    "csv":   ("patients_csv",  ".csv"),
    "excel": ("patients_xlsx", ".xlsx"),
    "pkl":   ("patients_pkl",  ".pkl"),
}
PRETRAINED = os.path.join("models", "ad_best_decoder.h5")


# ---------------------------------------------------------------------------
# Data loading  (mirrors preprocess_patient_data in vae_personalize.py)
# ---------------------------------------------------------------------------
class _NumpyCompatUnpickler(pickle.Unpickler):
    """Load pickles written with numpy>=2.0 under an older numpy (<2.0)."""
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def read_pickle_compat(path):
    with open(path, "rb") as f:
        return _NumpyCompatUnpickler(f).load()


def list_patients(fmt="csv"):
    """Discover available patient IDs by scanning the format's folder."""
    folder, ext = FORMAT_MAP[fmt]
    files = glob.glob(os.path.join(folder, f"*{ext}"))
    return sorted(os.path.splitext(os.path.basename(f))[0] for f in files)


def load_patient(patient_id, fmt="csv"):
    """Load one patient (BG / PI / RA) from the folder matching `fmt`."""
    if fmt not in FORMAT_MAP:
        raise ValueError(f"fmt must be one of {list(FORMAT_MAP)} (got {fmt!r})")
    folder, ext = FORMAT_MAP[fmt]
    path = os.path.join(folder, f"{patient_id}{ext}")
    if fmt == "csv":
        df = pd.read_csv(path)
    elif fmt == "excel":
        df = pd.read_excel(path)
    else:
        df = pd.DataFrame(read_pickle_compat(path))
    if "ID" in df.columns and df["ID"].nunique() > 1:
        df = df[df["ID"] == patient_id]
    return df[["BG", "PI", "RA"]].dropna().reset_index(drop=True)


def make_windows(df, horizon=HORIZON):
    """
    Per-patient scaling + sliding windows, in the shapes the model expects.
      insulin : (N, 1, 1)
      carbs   : (N, 1, 1)
      bg      : (N, 1, horizon)
    """
    scaler_bg, scaler_insulin, scaler_carbs = MinMaxScaler(), MinMaxScaler(), MinMaxScaler()
    bg      = scaler_bg.fit_transform(df[["BG"]]).astype("float32").ravel()
    insulin = scaler_insulin.fit_transform(df[["PI"]]).astype("float32").ravel()
    carbs   = scaler_carbs.fit_transform(df[["RA"]]).astype("float32").ravel()

    ins_list, carb_list, bg_list = [], [], []
    for i in range(len(df) - horizon):
        ins_list.append(np.array(insulin[i], dtype="float32").reshape(1, 1))
        carb_list.append(np.array(carbs[i],  dtype="float32").reshape(1, 1))
        bg_list.append(bg[i + 1: i + 1 + horizon].reshape(1, horizon))

    insulin_arr = np.stack(ins_list,  axis=0)
    carbs_arr   = np.stack(carb_list, axis=0)
    bg_arr      = np.stack(bg_list,   axis=0)
    scalers = {"bg": scaler_bg, "insulin": scaler_insulin, "carbs": scaler_carbs}
    return insulin_arr, carbs_arr, bg_arr, scalers


def holdout_split(insulin_arr, carbs_arr, bg_arr, holdout_days=HOLDOUT_DAYS):
    """Last `holdout_days` of samples held out (mirrors vae_personalize.py)."""
    holdout_n = holdout_days * SAMPLES_PER_DAY
    if len(bg_arr) <= holdout_n:
        holdout_n = max(1, len(bg_arr) // 4)
    tr = slice(0, len(bg_arr) - holdout_n)
    ho = slice(len(bg_arr) - holdout_n, len(bg_arr))
    train = (insulin_arr[tr], carbs_arr[tr], bg_arr[tr])
    hold  = (insulin_arr[ho], carbs_arr[ho], bg_arr[ho])
    return train, hold


# ---------------------------------------------------------------------------
# Architecture  (mirrors vae_model.py)
# ---------------------------------------------------------------------------
def define_encoder():
    """Conditional encoder  q(z | bg, insulin, carbs)."""
    init = RandomNormal(stddev=0.02)
    in_bg      = Input(shape=OUT_SHAPE, dtype=tf.float32, name="in_bg")
    in_insulin = Input(shape=IN_SHAPE,  dtype=tf.float32, name="in_insulin")
    in_carbs   = Input(shape=IN_SHAPE,  dtype=tf.float32, name="in_carbs")

    n = HORIZON
    bg_r  = Reshape((1, n))(Dense(n, kernel_initializer=init)(in_bg))
    ins_r = Reshape((1, n))(Dense(n, kernel_initializer=init)(in_insulin))
    car_r = Reshape((1, n))(Dense(n, kernel_initializer=init)(in_carbs))
    merged = Concatenate(axis=1)([ins_r, car_r, bg_r])              # (3, 18)

    e = LSTM(100, return_sequences=False, kernel_initializer=init)(merged)
    z_mean    = Dense(LATENT_DIM, name="z_mean")(e)
    z_log_var = Dense(LATENT_DIM, name="z_log_var")(e)

    def sampling(args):
        zm, zv = args
        eps = tf.random.normal(shape=(tf.shape(zm)[0], tf.shape(zm)[1]))
        return zm + tf.exp(0.5 * zv) * eps

    z = Lambda(sampling, output_shape=(LATENT_DIM,), name="z")([z_mean, z_log_var])
    return Model([in_bg, in_insulin, in_carbs], [z_mean, z_log_var, z], name="encoder")


def define_decoder():
    """Two-branch decoder: base trajectory from z + sign-constrained control."""
    init = RandomNormal(stddev=0.02)
    seq_len, hidden = HORIZON, HIDDEN
    in_lat  = Input(shape=(LATENT_DIM,), dtype=tf.float32, name="in_lat")
    in_ins  = Input(shape=IN_SHAPE,      dtype=tf.float32, name="in_ins")
    in_carb = Input(shape=IN_SHAPE,      dtype=tf.float32, name="in_carb")

    # Base branch (trajectory shape from z)
    h = Dense(hidden, kernel_initializer=init)(in_lat)
    h = LeakyReLU(0.2)(h)
    h = Dense(seq_len * hidden, kernel_initializer=init)(h)
    h = LeakyReLU(0.2)(h)
    h = Reshape((seq_len, hidden))(h)
    h = Dropout(0.3)(h)
    base = TimeDistributed(Dense(1, activation=None, kernel_initializer=init))(h)
    base = Flatten()(base)
    base = Reshape((1, seq_len))(base)

    # Control branch (insulin / carb heads)
    c = Concatenate()([Flatten()(in_ins), Flatten()(in_carb)])
    c = Dense(64, kernel_initializer=init)(c)
    c = LeakyReLU(0.2)(c)
    w_raw = Dense(seq_len * 2, kernel_initializer=init)(c)
    w_raw = Reshape((seq_len, 2))(w_raw)
    w_raw = LSTM(16, return_sequences=True)(w_raw)
    w_raw = Dense(2)(w_raw)
    w_ins_raw  = w_raw[..., 0]
    w_carb_raw = w_raw[..., 1]

    # Sign enforcement -> physiological monotonicity
    w_ins  = -(K.softplus(w_ins_raw)  + MIN_GAIN_MGDL)   # always negative
    w_carb =  (K.softplus(w_carb_raw) + MIN_GAIN_MGDL)   # always positive
    w_ins  = Reshape((1, seq_len))(w_ins)
    w_carb = Reshape((1, seq_len))(w_carb)

    ins  = Reshape((1, 1))(Flatten()(in_ins))
    carb = Reshape((1, 1))(Flatten()(in_carb))
    control = CONTROL_SCALE * (ins * w_ins + carb * w_carb)

    out = base + control
    return Model([in_lat, in_ins, in_carb], out, name="decoder_controlled")


class VAETrainer(tf.keras.Model):
    """Recon (MSE) + KL + soft counterfactual monotonicity penalty."""
    def __init__(self, encoder, decoder,
                 kl_weight=KL_WEIGHT, mono_weight=MONO_WEIGHT,
                 insulin_up=INSULIN_UP, carbs_up=CARBS_UP,
                 k_ins_100=K_INS_100, k_carb_100=K_CARB_100, **kwargs):
        super().__init__(**kwargs)
        self.encoder, self.decoder = encoder, decoder
        self.kl_weight, self.mono_weight = kl_weight, mono_weight
        self.insulin_up, self.carbs_up = insulin_up, carbs_up
        self.k_ins_100, self.k_carb_100 = k_ins_100, k_carb_100

    def train_step(self, data):
        x = data[0] if isinstance(data, tuple) else data
        bg, ins, carb = x
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder([bg, ins, carb], training=True)
            y0 = self.decoder([z, ins, carb], training=True)
            recon = K.mean(K.square(bg - y0))
            kl = -0.5 * K.mean(
                K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1))
            y_ins_up  = self.decoder([z, ins * self.insulin_up, carb], training=True)
            y_carb_up = self.decoder([z, ins, carb * self.carbs_up], training=True)
            mu_base   = K.mean(y0,        axis=[1, 2])
            mu_ins_up = K.mean(y_ins_up,  axis=[1, 2])
            mu_c_up   = K.mean(y_carb_up, axis=[1, 2])
            margin_ins  = (self.insulin_up - 1.0) * self.k_ins_100
            margin_carb = (self.carbs_up  - 1.0) * self.k_carb_100
            mono_ins  = K.relu(mu_ins_up - (mu_base - margin_ins))
            mono_carb = K.relu((mu_base + margin_carb) - mu_c_up)
            mono = K.mean(mono_ins + mono_carb)
            total = recon + self.kl_weight * kl + self.mono_weight * mono
        grads = tape.gradient(total, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": total, "recon": recon, "kl": kl, "mono": mono}


def build_trainer(encoder, decoder, lr=LR, **kw):
    vae = VAETrainer(encoder, decoder, **kw)
    vae.compile(optimizer=Adam(lr))
    return vae


# ---------------------------------------------------------------------------
# Anderson-Darling distance  (mirrors vae_model.py)
# ---------------------------------------------------------------------------
def _inv_cdf_at_u(samples, u):
    s = np.sort(np.ravel(samples).astype(float))
    if s.size == 0:
        return np.full_like(u, np.nan, dtype=float)
    idx = u * (len(s) - 1)
    lo, hi = np.floor(idx).astype(int), np.ceil(idx).astype(int)
    w = idx - lo
    return (1.0 - w) * s[lo] + w * s[hi]


def anderson_darling_distance(x, y, nq=2048, eps=1e-8):
    u = np.linspace(eps, 1.0 - eps, nq)
    qx, qy = _inv_cdf_at_u(x, u), _inv_cdf_at_u(y, u)
    w = 1.0 / (u * (1.0 - u))
    return float(np.mean(((qx - qy) ** 2) * w))


# ---------------------------------------------------------------------------
# Personalization loop  (mirrors VAEModel.pretrain)
# ---------------------------------------------------------------------------
def personalize(vae, encoder, decoder, dataset, save_dir=None,
                n_epochs=10, batch_size=32, use_early_stopping=True,
                patience=3, min_delta=0.0, restore_best=True, verbose=1):
    """
    Fine-tune the VAE with Wasserstein-distance early stopping.

    restore_best=True restores the best-score weights into the in-memory encoder
    and decoder before returning, so the objects you evaluate are the best
    checkpoint -- matching evaluate_holdout.py, which loads the best saved model.
    Without this, the in-memory model is the LAST epoch, which (for a model that
    starts from a strong population prior) can be well past its best score and will
    evaluate far worse than its early-stopping number implies.
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    train_ins, train_carbs, train_bg = dataset
    n = train_bg.shape[0]
    val_n = max(1, int(0.1 * n))
    tr, va = slice(0, n - val_n), slice(n - val_n, n)
    tr_ins, tr_carb, tr_bg = train_ins[tr], train_carbs[tr], train_bg[tr]
    va_ins, va_carb, va_bg = train_ins[va], train_carbs[va], train_bg[va]

    best_score, wait = np.inf, 0
    best_enc_w, best_dec_w = None, None

    if use_early_stopping:
        # Precompute a FIXED set of OU latent draws once, outside the epoch loop,
        # and reuse the same draws every epoch. If the seed varied per epoch,
        # score differences would reflect which random z happened to be sampled
        # that epoch as much as any real change in the decoder -- since
        # best_score starts at inf, epoch 1 would then trivially "win" and
        # patience could fire on pure latent-draw noise rather than genuine
        # non-improvement, restoring an early, barely fine-tuned checkpoint.
        # Averaging over several fixed draws further reduces that noise floor.
        n_val = va_bg.shape[0]
        N_SCORE_DRAWS = 5
        z_ou_draws = [
            tf.convert_to_tensor(generate_latent_ou(n_val, LATENT_DIM, seed=1000 + d),
                                  dtype=tf.float32)
            for d in range(N_SCORE_DRAWS)
        ]
        ins  = tf.convert_to_tensor(va_ins,  dtype=tf.float32)
        carb = tf.convert_to_tensor(va_carb, dtype=tf.float32)

    for epoch in range(n_epochs):
        hist = vae.fit([tr_bg, tr_ins, tr_carb], tr_bg,
                       epochs=1, batch_size=batch_size, verbose=verbose)
        if verbose:
            h = hist.history
            print(f"Epoch {epoch+1}/{n_epochs} | loss={h['loss'][0]:.6f} "
                  f"recon={h['recon'][0]:.6f} kl={h['kl'][0]:.6f} mono={h['mono'][0]:.6f}")
        if use_early_stopping:
            # Generative score: sample z from the same free-running OU prior used
            # by simulate() at evaluation time, NOT from the encoder's posterior
            # (see note above), evaluated on the SAME fixed draws every epoch.
            scores = []
            for z_ou in z_ou_draws:
                y_pred = decoder([z_ou, ins, carb], training=False).numpy()
                scores.append(wasserstein_1d(np.asarray(va_bg).reshape(-1), y_pred.reshape(-1)))
            score = float(np.mean(scores))
            if verbose:
                print(f"[W1] epoch {epoch+1}: {score:.6f} (best: {best_score:.6f})")
            if score < (best_score - float(min_delta)):
                best_score, wait = score, 0
                best_enc_w = encoder.get_weights()       # snapshot best in memory
                best_dec_w = decoder.get_weights()
                if save_dir:
                    encoder.save(os.path.join(save_dir, "ad_best_encoder.h5"))
                    decoder.save(os.path.join(save_dir, "ad_best_decoder.h5"))
            else:
                wait += 1
                if wait > int(patience):
                    if verbose:
                        print(f"[W1] Early stopping at epoch {epoch+1}. Best W1={best_score:.6f}")
                    break

    if restore_best and best_enc_w is not None:
        encoder.set_weights(best_enc_w)               # evaluate the BEST, not the last
        decoder.set_weights(best_dec_w)
        if verbose:
            print(f"[restore] in-memory model set to best-score weights (W1={best_score:.6f})")
    return best_score


# ---------------------------------------------------------------------------
# Simulation + metrics  (mirrors evaluate_holdout.py)
# ---------------------------------------------------------------------------
def generate_latent_ou(n_steps, latent_dim, theta=0.1, mean=0.0,
                       sigma=1.0, dt=1.0, seed=None):
    rng = np.random.default_rng(seed)
    z = np.zeros((n_steps, latent_dim), dtype="float32")
    z[0] = rng.normal(0.0, 1.0, size=latent_dim)
    for t in range(1, n_steps):
        dW = rng.normal(0.0, np.sqrt(dt), size=latent_dim)
        z[t] = z[t-1] + theta * (mean - z[t-1]) * dt + sigma * dW
    return z


def to_mgdl(seq_scaled, scaler_bg):
    s = seq_scaled.reshape(seq_scaled.shape[0], -1)
    return np.abs(scaler_bg.inverse_transform(s.reshape(-1, 1)).reshape(s.shape))


def simulate(decoder, ins, carb, scaler_bg, n_steps=None, seed=None):
    """Generate BG sequences (mg/dL), shape (n, H), via OU latents."""
    n = ins.shape[0] if n_steps is None else min(n_steps, ins.shape[0])
    ins, carb = ins[:n], carb[:n]
    z = generate_latent_ou(n, LATENT_DIM, seed=seed)
    gen = decoder.predict([z, ins, carb], verbose=0).reshape(n, -1)
    return to_mgdl(gen, scaler_bg)


def moving_avg_trajectory(decoder, ins, carb, scaler_bg, seed=None):
    """
    Overlap-and-average trajectory for qualitative visualisation.

    Each timestep t predicts HORIZON steps ahead via simulate().  The
    predictions are placed on a diagonal and averaged column-wise (NaN for
    empty cells), producing a single smooth 1-D BG curve of length
    n + HORIZON - 1.  This mirrors the original paper's simulation pipeline
    (twin_workshop2.py: generate_bg → instance_data → nanmean).

    Parameters
    ----------
    decoder   : Keras model — population or personalised decoder
    ins       : (n, 1, 1) scaled insulin array (holdout or slice)
    carb      : (n, 1, 1) scaled carbs array
    scaler_bg : fitted MinMaxScaler for BG
    seed      : int — OU latent seed (default 0, matches simulate())

    Returns
    -------
    trajectory : (n + HORIZON - 1,) float array in mg/dL
    """
    gen = simulate(decoder, ins, carb, scaler_bg, seed=seed)   # (n, H) mg/dL
    n, H = gen.shape
    mat = np.full((n, n + H), np.nan, dtype=np.float64)
    for i in range(n):
        mat[i, i:i + H] = gen[i]
    return np.nanmean(mat, axis=0)                             # (n + H - 1,)


def simulate_mc(decoder, ins, carb, scaler_bg, n_draws=20, base_seed=0):
    """
    Many OU draws per condition.

    Returns
    -------
    pooled   : (n_draws * n, H) — all draws concatenated (for KDE / ECDF plots)
    per_draw : DataFrame        — one row of state_metrics per draw (for boxplots)
    draws    : (n_draws, n, H)  — stacked individual draws (for CRPS)
    """
    seqs, per_draw = [], []
    for d in range(n_draws):
        G = simulate(decoder, ins, carb, scaler_bg, seed=base_seed + d)
        seqs.append(G)
        per_draw.append(state_metrics(G))
    draws = np.stack(seqs, axis=0)                 # (n_draws, n, H)
    return np.concatenate(seqs, axis=0), pd.DataFrame(per_draw), draws


def crps_ensemble(draws, obs):
    """
    Mean Continuous Ranked Probability Score of an ensemble forecast.

    Uses the energy-score identity (exact for finite ensembles):
        CRPS = E[|X - y|] - 0.5 * E[|X - X'|]

    where X, X' are independent draws from the forecast ensemble and y is the
    scalar observation. Averaged over all n * H timesteps.

    Parameters
    ----------
    draws : (n_draws, n, H)  — stacked trajectories from simulate_mc
    obs   : (n, H)           — real BG in mg/dL (e.g. G_real)

    Notes
    -----
    The spread term creates an (n_draws, n_draws, n, H) intermediate array.
    With n_draws=20 and n~864 this is ~50 M floats — fine for CPU.
    Keep n_draws <= 50 for live sessions to avoid slowdown.

    Returns
    -------
    float  — mean CRPS in mg/dL (lower is better; 0 = perfect deterministic hit)
    """
    T = min(draws.shape[1], obs.shape[0])
    d  = draws[:, :T, :]           # (n_draws, T, H)
    y  = obs[:T, :]                # (T, H)
    accuracy = float(np.mean(np.abs(d - y[None, :, :])))
    spread   = float(np.mean(np.abs(d[:, None, :, :] - d[None, :, :, :])))
    return accuracy - 0.5 * spread


def quantile_mae(real, gen, qs=(0.05, 0.25, 0.50, 0.75, 0.95)):
    return float(np.mean([abs(np.quantile(gen, q) - np.quantile(real, q)) for q in qs]))


def wasserstein_1d(real, gen, nq=1024):
    u = np.linspace(1e-8, 1 - 1e-8, nq)
    return float(np.mean(np.abs(np.quantile(real, u) - np.quantile(gen, u))))


def tailmass_l1(real, gen):
    bins = [lambda x: x < 54, lambda x: (x >= 54) & (x < 70),
            lambda x: (x >= 70) & (x <= 180), lambda x: (x > 180) & (x <= 250),
            lambda x: x > 250]
    return float(sum(abs(np.mean(b(gen)) - np.mean(b(real))) for b in bins))


def _acf_vec(x, lags):
    x = x - x.mean(); v = x.var()
    if v <= 1e-12:
        return np.zeros(lags)
    n = len(x)
    return np.array([np.dot(x[:-k], x[k:]) / ((n-k)*v) if k < n else 0.0
                     for k in range(1, lags + 1)])


def acf_rmse(G_real, G_gen, lags=12):
    L = max(1, min(lags, G_real.shape[1] - 1))
    ar = np.mean([_acf_vec(s, L) for s in G_real], axis=0)
    ag = np.mean([_acf_vec(s, L) for s in G_gen], axis=0)
    return float(np.sqrt(np.mean((ar - ag) ** 2)))


def roughness(G):
    return float(np.mean(np.mean(np.abs(np.diff(G, axis=1)), axis=1)))


def episodes(G, thr, above, dt=5.0):
    counts, durs = [], []
    for seq in G:
        m = (seq > thr if above else seq < thr).astype(int)
        starts = np.where((m[1:] == 1) & (m[:-1] == 0))[0] + 1
        if m[0] == 1:
            starts = np.insert(starts, 0, 0)
        ends = np.where((m[1:] == 0) & (m[:-1] == 1))[0] + 1
        if m[-1] == 1:
            ends = np.append(ends, len(m))
        counts.append(len(starts)); durs += list((ends - starts) * dt)
    return float(np.mean(counts)), float(np.median(durs) if durs else 0.0)


def state_metrics(G):
    s = pd.Series(G.reshape(-1))
    pct = lambda lo, hi: float(((s >= lo) & (s <= hi)).mean() * 100)
    n_hypo, _ = episodes(G, 70, above=False)
    n_hyper, _ = episodes(G, 180, above=True)
    return {
        "Mean BG": float(s.mean()),
        "CV %": float(s.std(ddof=0) / s.mean() * 100),
        "TIR 70-180": pct(70, 180),
        "Hypo <70": pct(0, 69),
        "Hyper >180": pct(181, 400),
        "Roughness": roughness(G),
        "# Hypo epi": n_hypo,
        "# Hyper epi": n_hyper,
    }


def distance_metrics(G_real, G_gen):
    rp, gp = G_real.reshape(-1), G_gen.reshape(-1)
    return {
        "Quantile MAE": quantile_mae(rp, gp),
        "Wasserstein": wasserstein_1d(rp, gp),
        "Tailmass L1": tailmass_l1(rp, gp),
        "ACF RMSE": acf_rmse(G_real, G_gen),
    }


# ---------------------------------------------------------------------------
# Complete clinical glycemic metrics  (ATTD/ADA consensus bands)
# ---------------------------------------------------------------------------
def glycemic_metrics(G):
    """
    Full consensus glycemic panel. The mutually-exclusive TBR/TIR/TAR bands
    sum to 100%. TITR 70-140 is the tight target range, a *subset* of TIR
    70-180, reported additionally. Bands are contiguous half-open intervals so
    continuous glucose values don't fall in cracks.
    """
    x = np.asarray(G).reshape(-1).astype(float)
    n = max(len(x), 1)
    mean = float(x.mean())
    frac = lambda mask: float(mask.mean() * 100)
    return {
        "Mean BG":      mean,
        "GMI %":        float(3.31 + 0.02392 * mean),       # glucose management indicator
        "CV %":         float(x.std(ddof=0) / mean * 100) if mean else float("nan"),
        "TBR <54":      frac(x < 54),                        # level-2 hypo
        "TBR 54-69":    frac((x >= 54) & (x < 70)),          # level-1 hypo
        "TIR 70-180":   frac((x >= 70) & (x <= 180)),        # target range
        "TITR 70-140":  frac((x >= 70) & (x <= 140)),        # tight target (subset of TIR)
        "TAR 180-250":  frac((x > 180) & (x <= 250)),        # level-1 hyper
        "TAR >250":     frac(x > 250),                       # level-2 hyper
    }


# ---------------------------------------------------------------------------
# Glucodensity  (Matabuena et al. — full glucose distribution as the object)
# ---------------------------------------------------------------------------
_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 2.x renames trapz


def glucodensity(values, grid=None, bw=None):
    """Normalized glucose density over a fixed mg/dL grid (integrates to 1)."""
    from scipy.stats import gaussian_kde
    if grid is None:
        grid = np.linspace(20, 400, 400)
    v = np.asarray(values).reshape(-1)
    kde = gaussian_kde(v) if bw is None else gaussian_kde(v, bw_method=bw)
    d = kde(grid)
    area = _trapz(d, grid)
    return grid, (d / area if area > 0 else d)


def wasserstein2(real, gen, nq=1024):
    """2-Wasserstein distance via quantile L2 — the glucodensity-native metric."""
    u = np.linspace(1e-8, 1 - 1e-8, nq)
    return float(np.sqrt(np.mean((np.quantile(real, u) - np.quantile(gen, u)) ** 2)))


def glucodensity_l2(real, gen, grid=None):
    """L2 distance between two glucodensities."""
    if grid is None:
        grid = np.linspace(20, 400, 400)
    _, dr = glucodensity(real, grid)
    _, dg = glucodensity(gen, grid)
    return float(np.sqrt(_trapz((dr - dg) ** 2, grid)))


def glucodensity_distance(G_real, G_gen):
    """Glucodensity comparison: 2-Wasserstein (primary) + density L2."""
    rp, gp = G_real.reshape(-1), G_gen.reshape(-1)
    return {
        "Glucodensity W2": wasserstein2(rp, gp),
        "Glucodensity L2": glucodensity_l2(rp, gp),
    }


# ---------------------------------------------------------------------------
# Setup verification
# ---------------------------------------------------------------------------
def verify_setup(fmt="csv"):
    """Print a readiness checklist so students know they can start."""
    ok = True
    print("Workshop setup check")
    print("-" * 40)

    import tensorflow as _tf
    using_legacy = os.environ.get("TF_USE_LEGACY_KERAS") == "1"
    print(f"  TensorFlow {_tf.__version__}  | legacy Keras: {using_legacy}")
    if not using_legacy:
        print("    ! set TF_USE_LEGACY_KERAS=1 BEFORE importing tensorflow")
        ok = False

    ids = list_patients(fmt)
    if ids:
        print(f"  data folder: {len(ids)} patients found ({fmt})")
    else:
        print(f"    ! no patient files in '{FORMAT_MAP[fmt][0]}/'")
        ok = False

    if os.path.exists(PRETRAINED):
        try:
            dec = load_model(PRETRAINED, compile=False)
            z = np.zeros((2, LATENT_DIM), "float32")
            i = np.zeros((2, 1, 1), "float32")
            _ = dec.predict([z, i, i], verbose=0)
            print(f"  pretrained decoder loads + forward pass OK")
        except Exception as e:
            print(f"    ! decoder failed to load: {type(e).__name__}: {e}")
            ok = False
    else:
        print(f"    ! missing {PRETRAINED}")
        ok = False

    print("-" * 40)
    print("READY" if ok else "NOT READY -- resolve the ! items above")
    return ok