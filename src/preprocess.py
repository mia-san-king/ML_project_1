from pathlib import Path

import numpy as np

SENTINEL_DEFAULT = (77, 99, 777, 999, 7777, 9999)
SENTINEL_DK_REFUSED = (7, 9)
DAYS_NONE_TO_ZERO = ("PHYSHLTH", "MENTHLTH", "POORHLTH", "CHILDREN")
NEVER_TREAT_7_9_AS_MISSING = ("_AGEG5YR", "_AGE65YR", "_AGE80", "_AGE_G", "_BMI5")

# admin / weights, leakage, BMI twins, >90% nan modules (EDA §13)
DROP_COLS = """
_STATE FMONTH IDATE IMONTH IDAY IYEAR DISPCODE SEQNO _PSU
CTELENUM PVTRESD1 COLGHOUS STATERES CELLFON3 LADULT NUMADULT NUMMEN NUMWOMEN
CTELNUM1 CELLFON2 CADULT PVTRESD2 CCLGHOUS CSTATE LANDLINE HHADULT
QSTVER QSTLANG MSCODE _STSTR _STRWT _RAWRAKE _WT2RAKE _CHISPNC
_CRACE1 _CPRACE _CLLCPWT _DUALUSE _DUALCOR _LLCPWT
HAREHAB1 STREHAB1 CVDASPRN ASPUNSAF RLIVPAIN RDUCHART
WEIGHT2 HEIGHT3 HTIN4 HTM4 WTKG3 _BMI5CAT _RFBMI5
NUMPHON2 INSULIN BLDSUGAR FEETCHK2 DOCTDIAB CHKHEMO3 FEETCHK EYEEXAM DIABEYE DIABEDU
CRGVREL1 CRGVLNG1 CRGVHRS1 CRGVPRB1 CRGVPERS CRGVHOUS CRGVMST2
VIDFCLT2 VIREDIF3 VIPRFVS2 VINOCRE2 VIEYEXM2 VIINSUR2 VICTRCT4 VIGLUMA2 VIMACDG2
CDHOUSE CDASSIST CDHELP CDSOCIAL CDDISCUS WTCHSALT LONGWTCH DRADVISE
ASTHMAGE ASATTACK ASERVIST ASDRVIST ASRCHKUP ASACTLIM ASYMPTOM ASNOSLEP ASTHMED3 ASINHALR
RDUCSTRK ARTTODAY ARTHWGT ARTHEXER ARTHEDU TETANUS HPVADVC2 HPVADSHT SHINGLE2
HADMAM HOWLONG HADPAP2 LASTPAP2 HPVTEST HPLSTTST HADHYST2 PROFEXAM LENGEXAM
LSTBLDS3 HADSGCO1 LASTSIG3 PCPSAAD2 PCPSADI1 PCPSARE1 PSATEST1 PSATIME PCPSARS1
PCPSADE1 PCDMDECN SCNTPAID SCNTWRK1 SCNTLPAD SCNTLWK1 CASTHNO2 EMTSUPRT LSATISFY
ADPLEASR ADDOWN ADSLEEP ADENERGY ADEAT1 ADFAIL ADTHINK ADMOVE MISTMNT ADANXEV
""".split()


def load_feature_names(data_path):
    """Return 1D array of 321 column names (no Id)."""
    path = Path(data_path) / "x_train.csv"
    with open(path) as f:
        header = f.readline().strip().split(",")
    return np.array(header[1:])


def drop_features(x, names, drop_cols=None):
    """Return (x_kept, names_kept). Same columns for any x with this names."""
    if drop_cols is None:
        drop_cols = DROP_COLS
    keep = ~np.isin(names, np.asarray(drop_cols))
    return x[:, keep], names[keep]


def apply_sentinels(x, names):
    """Replace BRFSS codes. Returns a new array; does not drop columns."""
    x = np.array(x, dtype=float, copy=True)
    names = np.asarray(names)
    for j, name in enumerate(names):
        col = x[:, j]
        if name in DAYS_NONE_TO_ZERO:
            col = np.where(col == 88, 0.0, col)
            col = np.where(np.isin(col, [77, 99]), np.nan, col)
        elif name == "_AGEG5YR":
            col = np.where(col == 14, np.nan, col)
        else:
            col = np.where(np.isin(col, SENTINEL_DEFAULT), np.nan, col)
            if name not in NEVER_TREAT_7_9_AS_MISSING:
                col = np.where(np.isin(col, SENTINEL_DK_REFUSED), np.nan, col)
        x[:, j] = col
    return x


def fit_medians(x):
    """1D medians, one per column. All-nan columns become 0."""
    med = np.nanmedian(x, axis=0)
    return np.nan_to_num(med, nan=0.0)


def apply_medians(x, medians):
    x = np.array(x, dtype=float, copy=True)
    return np.where(np.isnan(x), medians, x)


def fit_standardize(x_imputed):
    """Return (mean, std) per column. std==0 is replaced by 1."""
    mean = np.mean(x_imputed, axis=0)
    std = np.std(x_imputed, axis=0)
    std = np.where(std == 0, 1.0, std)
    return mean, std


def apply_standardize(x_imputed, mean, std):
    return (x_imputed - mean) / std


def fit_preprocess(x_train, names, drop_cols=None):
    """Learn drop mask + impute + scale on TRAIN only.

    Returns (x_train_clean, stats).
    """
    if drop_cols is None:
        drop_cols = DROP_COLS
    x, names_kept = drop_features(x_train, names, drop_cols)
    x = apply_sentinels(x, names_kept)
    medians = fit_medians(x)
    x = apply_medians(x, medians)
    mean, std = fit_standardize(x)
    x = apply_standardize(x, mean, std)
    stats = {
        "names_kept": names_kept,
        "medians": medians,
        "mean": mean,
        "std": std,
        "drop_cols": list(drop_cols),
    }
    return x, stats


def transform_preprocess(x, names, stats):
    """Apply a fitted pipeline to val or test."""
    x, names_kept = drop_features(x, names, stats["drop_cols"])
    if not np.array_equal(names_kept, stats["names_kept"]):
        raise ValueError("column names after drop do not match the train fit.")
    x = apply_sentinels(x, names_kept)
    x = apply_medians(x, stats["medians"])
    x = apply_standardize(x, stats["mean"], stats["std"])
    return x


def add_bias(x):
    """Prepend a column of 1s. Shape (N, D) -> (N, D+1)."""
    ones = np.ones((x.shape[0], 1))
    return np.hstack([ones, x])


def train_val_split(x, y, ids, val_ratio=0.2, seed=0):
    """Shuffle then split. Returns x_tr, y_tr, ids_tr, x_val, y_val, ids_val."""
    n = x.shape[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = int(n * val_ratio)
    val_i, tr_i = perm[:n_val], perm[n_val:]
    return x[tr_i], y[tr_i], ids[tr_i], x[val_i], y[val_i], ids[val_i]
