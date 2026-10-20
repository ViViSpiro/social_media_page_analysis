#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализ факторов, влияющих на реакцию аудитории страницы ВКонтакте.

Что делает скрипт:
  1. Забирает посты со стены страницы через официальный VK API (wall.get).
  2. Считает по каждому посту признаки: час, день недели, пауза с прошлого
     поста, длина текста, хештеги, эмодзи, вложения, возраст поста и т.д.
  3. Считает метрики реакции: лайки, комментарии, репосты, суммарный engagement.
  4. Ищет, что сильнее всего коррелирует с реакцией:
     - разбивка по часу публикации и дню недели (+ тест Краскела-Уоллиса),
     - разбивка по интервалу между постами,
     - корреляции Спирмена по числовым признакам,
     - эффект ключевых слов (Mann-Whitney U),
     - важность признаков по Random Forest (permutation importance).
  5. Сохраняет отчёты (CSV) и графики (PNG).

Как получить токен (нужен пользовательский, scope=wall, или сервисный):
  https://oauth.vk.com/authorize?client_id=ВАШ_ID&display=page
    &redirect_uri=https://oauth.vk.com/blank.html&scope=wall,offline
    &response_type=token&v=5.199
  Токен появится в адресной строке после &access_token=...

Запуск:
  python vk_analysis.py --screen-name khoruanna --token ВАШ_ТОКЕН
  python vk_analysis.py --screen-name khoruanna --demo        # демо без токена
"""

import argparse
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

try:
    from scipy import stats
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.inspection import permutation_importance
except ImportError:
    sys.exit("Установите зависимости: pip install -r requirements.txt")

API_URL = "https://api.vk.com/method/"
API_VERSION = "5.199"

WEEKDAYS_RU = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]

HASHTAG_RE = re.compile(r"#[\wа-яё]+", re.IGNORECASE)
MENTION_RE = re.compile(r"\[[^\]]+\]|@[\w\.]+")
LINK_RE = re.compile(r"https?://\S+")
TOKEN_RE = re.compile(r"[а-яёa-z]{3,}", re.IGNORECASE)
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U0001F1E6-\U0001F1FF"
    "\U00002600-\U000027BF\U00002190-\U000021FF"
    "\U00002B00-\U00002BFF\U0000FE00-\U0000FE0F\U00002700-\U00002EFF]"
)

RU_STOPWORDS = {
    "это", "как", "все", "уже", "для", "что", "его", "еще", "если", "или",
    "был", "была", "были", "быть", "есть", "так", "также", "только", "себя",
    "свой", "свою", "свое", "весь", "вся", "под", "над", "про", "при", "без",
    "кто", "чем", "чём", "них", "нам", "вам", "вас", "нас", "они", "она",
    "оно", "мы", "вы", "он", "ее", "её", "ему", "ней", "нём", "том", "тем",
    "тот", "та", "те", "этот", "эта", "эти", "это", "когда", "где", "чтобы",
    "будет", "будут", "может", "можно", "нужно", "очень", "самый", "более",
    "менее", "весь", "всей", "одна", "один", "два", "три", "через", "между",
    "после", "перед", "потом", "сюда", "туда", "здесь", "там", "сегодня",
    "вчера", "завтра", "день", "года", "году", "годы", "раз", "еще", "уже",
}


# ----------------------------------------------------------------------------
# VK API
# ----------------------------------------------------------------------------
def vk_api(method, token, **params):
    params["access_token"] = token
    params["v"] = API_VERSION
    resp = requests.get(API_URL + method, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        err = data["error"]
        raise RuntimeError(f"VK API {method}: {err.get('error_msg')} (code {err.get('error_code')})")
    return data["response"]


def resolve_owner(screen_name, token):
    res = vk_api("utils.resolveScreenName", token, screen_name=screen_name)
    if not res:
        raise SystemExit(f"Не удалось найти '{screen_name}'. Проверьте адрес сообщества.")
    return res


def fetch_posts(owner_id, token, count=100, max_posts=1000, pause=0.34):
    posts, offset = [], 0
    while len(posts) < max_posts:
        batch = vk_api(
            "wall.get", token,
            owner_id=owner_id, count=count, offset=offset, filter="owner",
        )
        items = batch.get("items", [])
        if not items:
            break
        posts.extend(items)
        offset += count
        if offset >= batch.get("count", 0):
            break
        time.sleep(pause)
    return posts[:max_posts]


def get_object_title(resolved, token):
    try:
        if resolved["type"] == "group":
            info = vk_api("groups.getById", token, group_id=resolved["object_id"], fields="name")
            groups = info.get("groups", []) if isinstance(info, dict) else info
            return groups[0].get("name") if groups else "?"
        info = vk_api("users.get", token, user_ids=resolved["object_id"], fields="screen_name")
        u = info[0]
        name = f"{u.get('first_name', '')} {u.get('last_name', '')}".strip()
        return name or u.get("screen_name") or "?"
    except Exception as e:
        return f"(имя недоступно: {e})"


def check_comments_access(raw_posts, owner_id, token):
    """Возвращает текст предупреждения, если комментарии недоступны через API, иначе None."""
    counts = [p.get("comments", {}).get("count", 0) for p in raw_posts]
    if any(c > 0 for c in counts):
        return None
    pid = raw_posts[0]["id"]
    try:
        res = vk_api("wall.getComments", token, owner_id=owner_id, post_id=pid, count=1)
        total = res.get("count", 0)
        if total > 0:
            return (f"wall.get вернул comments=0, но wall.getComments показывает {total}. "
                    "Поле comments.count ненадёжно — нужен пользовательский токен.")
    except Exception as e:
        msg = str(e)
        if "1051" in msg:
            return None
        if "212" in msg or "denied" in msg.lower():
            return ("НЕТ ДОСТУПА К КОММЕНТАРИЯМ (wall.getComments: error 212 "
                    "'Access to post comments denied'). Токен не может читать комментарии этой "
                    "стены. Используйте пользовательский токен аккаунта, под которым комментарии "
                    "видны (владелец/друг), со scope=wall.")
        return f"Проверка комментариев не удалась: {e}"
    return None


# ----------------------------------------------------------------------------
# Признаки
# ----------------------------------------------------------------------------
def parse_post(p, tz):
    text = p.get("text") or ""
    atts = p.get("attachments") or []
    types = [a.get("type") for a in atts]
    dt = datetime.fromtimestamp(p["date"], tz)
    likes = p.get("likes", {}).get("count", 0)
    comments = p.get("comments", {}).get("count", 0)
    reposts = p.get("reposts", {}).get("count", 0)
    views = p.get("views", {}).get("count", np.nan)
    words = TOKEN_RE.findall(text.lower())
    return {
        "id": p["id"],
        "date": dt,
        "text": text,
        "hour": dt.hour,
        "weekday": dt.weekday(),
        "weekday_name": WEEKDAYS_RU[dt.weekday()],
        "text_len": len(text),
        "word_count": len(words),
        "n_hashtags": len(HASHTAG_RE.findall(text)),
        "n_mentions": len(MENTION_RE.findall(text)),
        "n_links": len(LINK_RE.findall(text)),
        "n_emojis": len(EMOJI_RE.findall(text)),
        "n_attachments": len(atts),
        "has_photo": int("photo" in types),
        "has_video": int("video" in types),
        "has_link": int("link" in types),
        "has_poll": int("poll" in types),
        "has_doc": int("doc" in types),
        "has_audio": int(("audio" in types) or ("audio_message" in types)),
        "is_pinned": int(p.get("is_pinned", 0)),
        "marked_as_ads": int(p.get("marked_as_ads", 0)),
        "likes": likes,
        "comments": comments,
        "reposts": reposts,
        "views": views,
        "engagement": likes + comments + reposts,
    }


def build_dataframe(raw_posts, tz, exclude_ads=True):
    df = pd.DataFrame([parse_post(p, tz) for p in raw_posts])
    if df.empty:
        raise SystemExit("Не удалось получить ни одного поста.")
    if exclude_ads and "marked_as_ads" in df:
        df = df[df["marked_as_ads"] == 0].copy()
    df = df.sort_values("date").reset_index(drop=True)
    df["interval_hours"] = df["date"].diff().dt.total_seconds() / 3600
    now = datetime.now(tz)
    df["age_days"] = (now - df["date"]).dt.total_seconds() / 86400
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["engagement_per_view"] = df["engagement"] / df["views"]
    return df


# ----------------------------------------------------------------------------
# Анализ
# ----------------------------------------------------------------------------
TARGETS = ["likes", "comments", "reposts", "engagement"]
NUMERIC_FEATURES = [
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "interval_hours",
    "age_days", "text_len", "word_count", "n_hashtags", "n_mentions",
    "n_links", "n_emojis", "n_attachments", "has_photo", "has_video",
    "has_link", "has_poll", "has_doc", "has_audio",
]


def group_by(df, column, targets=TARGETS):
    agg = df.groupby(column).agg(
        posts=("id", "count"),
        **{f"{t}_mean": (t, "mean") for t in targets},
        **{f"{t}_median": (t, "median") for t in targets},
    )
    return agg.sort_values("engagement_mean", ascending=False)


def kruskal_report(df, column, target="engagement"):
    groups = [g[target].values for _, g in df.groupby(column) if len(g) >= 3]
    if len(groups) < 2:
        return None
    stat, p = stats.kruskal(*groups)
    return {"factor": column, "target": target, "H": round(stat, 2), "p_value": p,
            "significant": p < 0.05}


def correlations(df, features=NUMERIC_FEATURES, targets=TARGETS):
    rows = []
    for f in features:
        for t in targets:
            sub = df[[f, t]].dropna()
            if len(sub) < 10 or sub[f].nunique() < 2 or sub[t].nunique() < 2:
                continue
            rho, p = stats.spearmanr(sub[f], sub[t])
            rows.append({"feature": f, "target": t, "spearman_rho": round(rho, 3),
                         "p_value": p, "significant": p < 0.05})
    res = pd.DataFrame(rows)
    if not res.empty:
        res = res.sort_values(["target", "spearman_rho"], key=lambda s: s.abs() if s.name == "spearman_rho" else s, ascending=[True, False])
    return res


def keyword_effects(df, min_posts=10, max_df_ratio=0.9, top_n=30, target="engagement"):
    if df[target].nunique() <= 1:
        return pd.DataFrame()
    texts = df["text"].fillna("")
    cv = CountVectorizer(
        token_pattern=r"(?u)\b\w{3,}\b", stop_words=list(RU_STOPWORDS),
        min_df=min_posts, binary=True,
    )
    X = cv.fit_transform(texts)
    vocab = cv.get_feature_names_out()
    y = df[target].values
    n = len(df)
    rows = []
    for i, word in enumerate(vocab):
        mask = X[:, i].toarray().ravel().astype(bool)
        n_present = int(mask.sum())
        if n_present >= n * max_df_ratio:
            continue
        a, b = y[mask], y[~mask]
        if len(b) < 3:
            continue
        try:
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            continue
        rows.append({
            "keyword": word,
            "posts_with": n_present,
            f"{target}_with": round(a.mean(), 2),
            f"{target}_without": round(b.mean(), 2),
            "lift_pct": round((a.mean() - b.mean()) / (b.mean() + 1e-9) * 100, 1),
            "p_value": p,
        })
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res = res.sort_values("lift_pct", ascending=False)
    top = pd.concat([res.head(top_n), res.tail(top_n)]).drop_duplicates("keyword")
    return top


def rf_importance(df, target="engagement", features=NUMERIC_FEATURES):
    data = df.dropna(subset=[target]).copy()
    if len(data) < 30 or data[target].nunique() <= 1:
        return pd.DataFrame()
    X = data[features].apply(pd.to_numeric, errors="coerce").fillna(0).values
    y = np.log1p(data[target].values)
    model = RandomForestRegressor(n_estimators=400, random_state=0, n_jobs=-1)
    model.fit(X, y)
    perm = permutation_importance(model, X, y, n_repeats=10, random_state=0, n_jobs=-1)
    res = pd.DataFrame({
        "feature": features,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values("importance_mean", ascending=False)
    res["importance_mean"] = res["importance_mean"].round(4)
    res["r2"] = round(model.score(X, y), 3)
    return res


# ----------------------------------------------------------------------------
# Графики
# ----------------------------------------------------------------------------
def make_plots(df, rf_tables, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, name), dpi=130)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    df.groupby("hour")[["likes", "comments", "reposts"]].mean().plot(kind="bar", ax=ax)
    ax.set_title("Средняя реакция по часу публикации")
    ax.set_xlabel("Час суток")
    save(fig, "engagement_by_hour.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    wd = df.groupby("weekday")[["likes", "comments", "reposts"]].mean()
    wd.index = [WEEKDAYS_RU[i] for i in wd.index]
    wd.plot(kind="bar", ax=ax)
    ax.set_title("Средняя реакция по дню недели")
    ax.set_xlabel("День недели")
    save(fig, "engagement_by_weekday.png")

    fig, ax = plt.subplots(figsize=(12, 4))
    d = df.sort_values("date")
    ax.plot(d["date"], d["engagement"], "o-", ms=2, lw=0.5, alpha=0.6)
    ax.set_title("Динамика engagement по постам")
    save(fig, "engagement_over_time.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    sub = df.dropna(subset=["interval_hours"])
    if not sub.empty:
        ax.scatter(sub["interval_hours"].clip(upper=200), sub["engagement"], alpha=0.4, s=12)
        ax.set_xlabel("Часов с прошлого поста (обрезано до 200)")
        ax.set_ylabel("Engagement")
        ax.set_title("Реакция vs интервал между постами")
    save(fig, "engagement_vs_interval.png")

    for target, table in rf_tables.items():
        if table.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 6))
        t = table.sort_values("importance_mean")
        ax.barh(t["feature"], t["importance_mean"], xerr=t["importance_std"])
        ax.set_title(f"Важность признаков (Random Forest) — {target}")
        save(fig, f"rf_importance_{target}.png")


# ----------------------------------------------------------------------------
# Демо-данные (если нет токена)
# ----------------------------------------------------------------------------
def generate_demo_posts(n=500, seed=42):
    rng = np.random.default_rng(seed)
    t = datetime(2023, 1, 1, 10, 0, tzinfo=timezone(timedelta(hours=3)))
    hour_bonus = {h: 1.0 for h in range(24)}
    for h in range(18, 23):
        hour_bonus[h] = 1.9
    for h in range(9, 12):
        hour_bonus[h] = 1.4
    wd_bonus = [1.0, 1.25, 1.2, 1.15, 1.05, 0.85, 0.9]
    keywords = ["конкурс", "розыгрыш", "подборка", "советы", "новости"]
    posts = []
    for i in range(n):
        t = t + timedelta(hours=float(rng.choice([8, 12, 18, 24, 36, 48, 72])))
        h, wd = t.hour, t.weekday()
        has_photo = rng.random() < 0.7
        kw = rng.choice(keywords) if rng.random() < 0.5 else None
        base = 40 * hour_bonus[h] * wd_bonus[wd] * (1.35 if has_photo else 1.0)
        kw_boost = 1.8 if kw in ("конкурс", "розыгрыш") else 1.1
        noise = rng.lognormal(0, 0.5)
        likes = max(0, int(rng.poisson(base * noise)))
        comments = max(0, int(rng.poisson(base * noise * 0.12 * kw_boost)))
        reposts = max(0, int(rng.poisson(base * noise * 0.05 * kw_boost)))
        text = (f"{kw.capitalize() if kw else 'Пост'} номер {i}. " * rng.integers(2, 8))
        atts = [{"type": "photo"}] if has_photo else []
        posts.append({
            "id": i + 1,
            "date": int(t.timestamp()),
            "text": text + " #хэштег",
            "likes": {"count": likes},
            "comments": {"count": comments},
            "reposts": {"count": reposts},
            "views": {"count": int(base * noise * 30)},
            "attachments": atts,
            "marked_as_ads": 0,
        })
    return posts


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Анализ реакции аудитории VK-сообщества")
    ap.add_argument("--screen-name", default="khoruanna")
    ap.add_argument("--token", default=None)
    ap.add_argument("--max-posts", type=int, default=1000)
    ap.add_argument("--tz-offset", type=int, default=3, help="часовой сдвиг от UTC (МСК=3)")
    ap.add_argument("--min-keyword-posts", type=int, default=10)
    ap.add_argument("--outdir", default="vk_report")
    ap.add_argument("--demo", action="store_true", help="сгенерировать данные без токена")
    ap.add_argument("--owner-id", type=int, default=None,
                    help="Числовой owner_id страницы (если resolveScreenName недоступен)")
    args = ap.parse_args()

    tz = timezone(timedelta(hours=args.tz_offset))
    os.makedirs(args.outdir, exist_ok=True)
    lines = []

    def log(*a):
        text = " ".join(str(x) for x in a)
        print(text)
        lines.append(text)

    if args.demo:
        log("РЕЖИМ ДЕМО: используются синтетические данные.\n")
        raw = generate_demo_posts()
        owner_id = -1
    else:
        token = args.token or os.environ.get("VK_TOKEN")
        if not token:
            sys.exit("Нужен токен: --token ВАШ_ТОКЕН или переменная окружения VK_TOKEN "
                     "(либо запустите с --demo).")
        if args.owner_id is not None:
            owner_id = args.owner_id
            log(f"Используем owner_id={owner_id} (--owner-id; resolveScreenName пропущен)")
        else:
            resolved = resolve_owner(args.screen_name, token)
            owner_id = -resolved["object_id"] if resolved["type"] == "group" else resolved["object_id"]
            kind = "Сообщество" if resolved["type"] == "group" else "Личная страница"
            log(f"{kind}: {get_object_title(resolved, token)} (id={abs(owner_id)})")
        log(f"Загружаю до {args.max_posts} постов...")
        raw = fetch_posts(owner_id, token, max_posts=args.max_posts)

    df = build_dataframe(raw, tz)
    log(f"Постов в анализе: {len(df)}; период: "
        f"{df['date'].min():%Y-%m-%d} .. {df['date'].max():%Y-%m-%d}\n")
    log("Средние метрики:", {t: round(df[t].mean(), 1) for t in TARGETS}, "\n")

    if not args.demo and raw:
        warn = check_comments_access(raw, owner_id, token)
        if warn:
            log("ВНИМАНИЕ: " + warn + "\n")
    const_targets = [t for t in TARGETS if df[t].nunique() <= 1]
    if const_targets:
        log(f"ВНИМАНИЕ: у метрик {const_targets} нет вариативности — "
            "корреляции, модель и ключевые слова по ним не считаются.\n")

    by_hour = group_by(df, "hour")
    by_weekday = group_by(df, "weekday")
    log("=== Лучшие часы публикации (по среднему engagement) ===")
    log(by_hour[["posts", "likes_mean", "comments_mean", "reposts_mean", "engagement_mean"]]
        .head(8).to_string())

    log("\n=== Лучшие дни недели ===")
    bwd = by_weekday.copy()
    bwd.index = [WEEKDAYS_RU[int(i)] for i in bwd.index]
    log(bwd[["posts", "likes_mean", "comments_mean", "reposts_mean", "engagement_mean"]].to_string())

    log("\n=== Тесты значимости различий (Краскела-Уоллиса) ===")
    kr = [kruskal_report(df, "hour"), kruskal_report(df, "weekday")]
    for r in kr:
        if r:
            log(f"  {r['factor']:>8} -> p={r['p_value']:.4g} "
                f"({'значимо' if r['significant'] else 'не значимо'})")

    corr = correlations(df)
    log("\n=== Топ корреляций Спирмена с engagement ===")
    if not corr.empty:
        top_corr = corr[corr["target"] == "engagement"].head(10)
        log(top_corr.to_string(index=False))

    log("\n=== Эффект ключевых слов (engagement, Mann-Whitney) ===")
    kw = keyword_effects(df, min_posts=args.min_keyword_posts)
    if not kw.empty:
        log(kw.head(15).to_string(index=False))

    log("\n=== Важность признаков (Random Forest) ===")
    rf_tables = {}
    for target in TARGETS:
        table = rf_importance(df, target)
        rf_tables[target] = table
        if table.empty:
            log(f"\n-- {target}: пропущено (мало данных или метрика постоянна) --")
        else:
            log(f"\n-- {target} (R2={table['r2'].iloc[0]}) --")
            log(table.head(10).to_string(index=False))

    # Сохранение отчётов
    df.to_csv(os.path.join(args.outdir, "posts.csv"), index=False, encoding="utf-8-sig")
    by_hour.to_csv(os.path.join(args.outdir, "by_hour.csv"), encoding="utf-8-sig")
    by_weekday.to_csv(os.path.join(args.outdir, "by_weekday.csv"), encoding="utf-8-sig")
    if not corr.empty:
        corr.to_csv(os.path.join(args.outdir, "correlations.csv"), index=False, encoding="utf-8-sig")
    if not kw.empty:
        kw.to_csv(os.path.join(args.outdir, "keyword_effects.csv"), index=False, encoding="utf-8-sig")
    for target, table in rf_tables.items():
        if not table.empty:
            table.to_csv(os.path.join(args.outdir, f"rf_importance_{target}.csv"),
                         index=False, encoding="utf-8-sig")

    try:
        make_plots(df, rf_tables, args.outdir)
        log(f"\nГрафики сохранены в '{args.outdir}'.")
    except Exception as e:
        log(f"Не удалось построить графики: {e}")

    with open(os.path.join(args.outdir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"\nГотово. Отчёты в папке '{args.outdir}'.")


if __name__ == "__main__":
    main()
