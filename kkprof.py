"""Inference helpers for the Kazakh profanity classifier.

prep_v1: original (lowercase, drop links, collapse long repeats).
prep_v2: v1 + masking handling + Kazakh-letter folding. prep_v3: v1 + masking handling only (training adds folded copies). A bundle stores which one it was trained with
         ("prep": "v1" | "v2"); old bundles without the key are treated as v1.
"""
import re, unicodedata, joblib

_L = "а-яәіңғүұқөһa-z"      # letters (after lowercasing; ё is replaced before use)
_FOLD = str.maketrans({"қ": "к", "ғ": "г", "ә": "а", "ү": "у", "ұ": "у",
                       "і": "и", "ң": "н", "ө": "о", "һ": "х"})
_URL = re.compile(r"https?://\S+")
_MAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")           # needs a dot, so 'с@ка' is NOT treated as an email
_MASK = re.compile(rf"(?<=[{_L}])\*+(?=[{_L}])")           # бл*ть -> блть
_LEET_O = re.compile(rf"(?<=[{_L}])0(?=[{_L}])")           # қ0тақ -> қотақ
_LEET_A = re.compile(rf"(?<=[{_L}])@(?=[{_L}])")           # с@ка -> сака
_REP = re.compile(r"(.)\1{2,}")


def prep_v1(s):
    s = unicodedata.normalize("NFKC", str(s)).lower().replace("ё", "е")
    s = re.sub(r"https?://\S+|\S+@\S+", " ", s)
    s = re.sub(r"(.)\1{2,}", r"\1\1", s)
    return re.sub(r"\s+", " ", s).strip()


def _clean(s):
    s = unicodedata.normalize("NFKC", str(s)).lower().replace("ё", "е")
    s = _MAIL.sub(" ", _URL.sub(" ", s))
    s = _MASK.sub("", s)                 # masking asterisks inside a word are removed
    s = _LEET_O.sub("о", s)              # digit/symbol look-alikes inside a word
    s = _LEET_A.sub("а", s)
    s = _REP.sub(r"\1\1", s)             # 'ааааа' -> 'аа'
    return re.sub(r"\s+", " ", s).strip()


def fold(s):
    return s.translate(_FOLD)            # Kazakh-specific letters -> Russian counterparts


def prep_v2(s):                          # masking handling + folding (rejected on validation)
    return fold(_clean(s))


def prep_v3(s):                          # masking handling, NO folding; trained on original + folded copies
    return _clean(s)


PREPS = {"v1": prep_v1, "v2": prep_v2, "v3": prep_v3}
prep = prep_v1                            # backward-compatible alias


def get_prep(bundle):
    return PREPS[bundle.get("prep", "v1")]


def load(path="models/tfidf_lr.joblib"):
    return joblib.load(path)


def predict(text, bundle=None):
    b = bundle or load()
    p = float(b["clf"].predict_proba(b["vec"].transform([get_prep(b)(text)]))[0, 1])
    return {"label": int(p >= b["thr"]), "prob": round(p, 4), "threshold": round(b["thr"], 4)}