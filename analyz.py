import json
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DATA_PATH = Path("data/botsv1.json") 
OUT_DIR = Path("output")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_botsv1_json(path: Path) -> pd.DataFrame:
    
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    # raw может быть списком: [{ "result": {...}}, ...]
    rows = []
    for item in raw:
        if isinstance(item, dict) and "result" in item and isinstance(item["result"], dict):
            rows.append(item["result"])
        elif isinstance(item, dict):
            # на случай если "result" отсутствует и поля на верхнем уровне
            rows.append(item)
        else:
            # если элемент не словарь — пропускаем
            continue

    df = pd.DataFrame(rows)
    return df


def normalize_time(df: pd.DataFrame) -> pd.DataFrame:
    # В botsv1 часто _time хранится как строка времени
    if "_time" in df.columns:
        df["_time"] = pd.to_datetime(df["_time"], errors="coerce")
    return df


def split_logs(df: pd.DataFrame):
    """
    Делим на WinEventLog и DNS по полю "source" / "sourcetype" / "eventtype" (что найдём).
    """
    # Windows (в примерах видно source = "WinEventLog:Security")
    is_win = False
    if "source" in df.columns:
        is_win = df["source"].astype(str).str.contains("WinEventLog", na=False)
    elif "sourcetype" in df.columns:
        is_win = df["sourcetype"].astype(str).str.contains("WinEventLog", na=False)

    win_df = df[is_win].copy()

    # DNS: ищем по sourcetype/source/eventtype
    is_dns = False
    if "sourcetype" in df.columns:
        is_dns = df["sourcetype"].astype(str).str.contains("dns", case=False, na=False)
    elif "source" in df.columns:
        is_dns = df["source"].astype(str).str.contains("dns", case=False, na=False)
    else:
        # иногда DNS можно вычислить по наличию query/domain поля, но это запасной вариант
        possible_dns_cols = {"query", "query_name", "domain", "dest_dns", "dns_query"}
        is_dns = df.columns.to_series().isin(possible_dns_cols).any()

    dns_df = df[is_dns].copy()

    return win_df, dns_df


def suspicious_wineventlog(win_df: pd.DataFrame) -> pd.DataFrame:
   
    # В botsv1 встречается EventCode и signature_id
    event_col = None
    for c in ["EventCode", "signature_id", "EventID", "event_id"]:
        if c in win_df.columns:
            event_col = c
            break

    if event_col is None or win_df.empty:
        return pd.DataFrame(columns=["type", "key", "count"])

    win_df[event_col] = win_df[event_col].astype(str)

    suspicious_ids = {
        # логины
        "4625": "Failed logon (4625)",
        "4624": "Successful logon (4624) - check anomalies",
        "4648": "Logon with explicit creds (4648)",
        "4672": "Special privileges assigned (4672)",
        # управление пользователями/группами
        "4720": "User account created (4720)",
        "4722": "User enabled (4722)",
        "4723": "Password change attempt (4723)",
        "4724": "Password reset attempt (4724)",
        "4728": "Added to privileged group (4728)",
        "4732": "Added to local group (4732)",
        "4756": "Added to universal group (4756)",
        # процессы/службы/планировщик
        "4688": "Process created (4688)",
        "4697": "Service installed (4697)",
        "7045": "Service created (7045)",
        "4698": "Scheduled task created (4698)",
        "4699": "Scheduled task deleted (4699)",
        # логи/аудит
        "1102": "Audit log cleared (1102)",
        "4719": "Audit policy changed (4719)",
        "4703": "User right adjusted (4703)"
    }

    # Оставляем те события, которые есть в списке
    flagged = win_df[win_df[event_col].isin(suspicious_ids.keys())].copy()
    if flagged.empty:
        # если в данных нет ни одного из выбранных — вернем топ по частоте EventCode как "подозрительные по частоте"
        top = (
            win_df[event_col]
            .value_counts()
            .head(10)
            .reset_index()
            .rename(columns={"index": "key", event_col: "count"})
        )
        top["type"] = "WinEventLog (top by frequency)"
        return top[["type", "key", "count"]]

    # Группируем
    cnt = flagged[event_col].value_counts().reset_index()
    cnt.columns = ["key", "count"]
    cnt["type"] = "WinEventLog"
    # делаем человекочитаемое описание (key будет "4625 — Failed logon")
    cnt["key"] = cnt["key"].map(lambda x: suspicious_ids.get(str(x), str(x)))
    return cnt[["type", "key", "count"]]


def extract_domain(q: str) -> str:
    
    if not isinstance(q, str) or not q:
        return ""
    q = q.strip(".").lower()
    parts = q.split(".")
    if len(parts) <= 2:
        return q
    # грубо: last2 (example.com)
    return ".".join(parts[-2:])


def suspicious_dns(dns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ищем подозрительное в DNS
    """
    if dns_df.empty:
        return pd.DataFrame(columns=["type", "key", "count"])

    # Пытаемся найти колонку с DNS именем запроса
    query_col = None
    for c in ["query", "query_name", "dns_query", "dest_dns", "Domain", "domain", "QueryName"]:
        if c in dns_df.columns:
            query_col = c
            break

    # Если нет явной колонки — пробуем вытащить из "Message"/"_raw"
    if query_col is None:
        for c in ["Message", "_raw", "body"]:
            if c in dns_df.columns:
                query_col = c
                break

    if query_col is None:
        return pd.DataFrame(columns=["type", "key", "count"])

    s = dns_df[query_col].astype(str)

    domain_like = s.str.extract(r"([a-zA-Z0-9\-]{1,63}(?:\.[a-zA-Z0-9\-]{1,63})+\.[a-zA-Z]{2,})", expand=False)
    qname = domain_like.fillna(s).str.strip().str.lower()

    dns_work = dns_df.copy()
    dns_work["qname"] = qname
    dns_work["domain"] = dns_work["qname"].map(extract_domain)
    dns_work["qname_len"] = dns_work["qname"].str.len()
    dns_work["dots"] = dns_work["qname"].str.count(r"\.")
    dns_work["digits"] = dns_work["qname"].str.count(r"\d")

    # эвристики "нестандартного поддомена"
    # 1) очень длинное имя
    # 2) много уровней (точек)
    # 3) много цифр
    suspicious_q = dns_work[
        (dns_work["qname_len"] >= 40) |
        (dns_work["dots"] >= 5) |
        (dns_work["digits"] >= 10)
    ].copy()

    # частые обращения к доменам
    domain_counts = dns_work["domain"].value_counts().reset_index()
    domain_counts.columns = ["domain", "count_domains"]

    # редкие домены: встречаются 1-2 раза (можно расширить)
    rare_domains = set(domain_counts[domain_counts["count_domains"] <= 2]["domain"].tolist())

    rare_hits = dns_work[dns_work["domain"].isin(rare_domains)]
    rare_top = rare_hits["domain"].value_counts().head(10).reset_index()
    rare_top.columns = ["key", "count"]
    rare_top["type"] = "DNS (rare domains)"

    weird_top = suspicious_q["qname"].value_counts().head(10).reset_index()
    weird_top.columns = ["key", "count"]
    weird_top["type"] = "DNS (weird subdomains)"

    out = pd.concat([rare_top, weird_top], ignore_index=True)
    return out[["type", "key", "count"]]


def save_tables(win_top: pd.DataFrame, dns_top: pd.DataFrame, combined_top: pd.DataFrame):
    win_top.to_csv(OUT_DIR / "win_suspicious_top.csv", index=False, encoding="utf-8")
    dns_top.to_csv(OUT_DIR / "dns_suspicious_top.csv", index=False, encoding="utf-8")
    combined_top.to_csv(OUT_DIR / "combined_top.csv", index=False, encoding="utf-8")


def plot_top10(df: pd.DataFrame, title: str, filename: str):
    if df.empty:
        return

    top10 = df.sort_values("count", ascending=False).head(10).copy()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top10, x="count", y="key")
    plt.title(title)
    plt.xlabel("Count")
    plt.ylabel("Event / Query")
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=200)
    plt.close()


def main():
    df = load_botsv1_json(DATA_PATH)
    df = normalize_time(df)

    win_df, dns_df = split_logs(df)

    win_susp = suspicious_wineventlog(win_df)
    dns_susp = suspicious_dns(dns_df)

    # объединённая визуализация: топ по суммарной частоте
    combined = pd.concat([win_susp, dns_susp], ignore_index=True)
    combined_top = (
        combined.groupby(["type", "key"], as_index=False)["count"].sum()
        .sort_values("count", ascending=False)
    )

    save_tables(win_susp, dns_susp, combined_top)

    plot_top10(win_susp, "Top-10 suspicious WinEventLog events", "win_top10.png")
    plot_top10(dns_susp, "Top-10 suspicious DNS events", "dns_top10.png")
    plot_top10(combined_top.assign(key=combined_top["type"] + " | " + combined_top["key"]),
               "Top-10 suspicious events (combined)", "combined_top10.png")

    print("Done! Files saved to ./output")


if __name__ == "__main__":
    main()
