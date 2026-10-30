#!/usr/bin/env python3
"""Robô de dados — análise de apostas de futebol.

Descarrega os ficheiros do football-data.co.uk, estima os ratings de cada equipa
(ponderação temporal + regularização para a média da liga) e grava:
  dados.json     -> próximos jogos (λ, odds, mercado), ratings atuais, resultados recentes
  backtest.json  -> previsões "walk-forward" de todos os jogos das épocas carregadas

As probabilidades (Poisson + Dixon-Coles) são calculadas na página, a partir de λ,
para que os parâmetros (ρ, α, limiares) possam ser alterados sem voltar a correr o robô.

Uso:  python atualizar.py <pasta_saida>
Teste local:  FD_LOCAL=<pasta com mmz4281/... e fixtures.csv> FD_HOJE=2026-10-30 python atualizar.py saida
Só usa a biblioteca padrão do Python.
"""
import csv, io, json, os, sys, time, datetime as dt, urllib.request

BASE = "https://www.football-data.co.uk/"
AQUI = os.path.dirname(os.path.abspath(__file__))


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ leitura
def obter(caminho):
    """Devolve o texto do ficheiro (local em modo de teste, ou da Web)."""
    local = os.environ.get("FD_LOCAL")
    if local:
        p = os.path.join(local, caminho)
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        with open(p, "rb") as fh:
            raw = fh.read()
    else:
        ultimo = None
        for tentativa in range(3):
            try:
                req = urllib.request.Request(BASE + caminho, headers={"User-Agent": "Mozilla/5.0 (analise-apostas; uso pessoal)"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    raw = r.read()
                break
            except Exception as e:  # noqa: BLE001
                ultimo = e
                time.sleep(3 * (tentativa + 1))
        else:
            raise ultimo
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def ler_csv(texto):
    linhas = list(csv.reader(io.StringIO(texto)))
    if not linhas:
        return []
    cab = [c.strip().lstrip("﻿") for c in linhas[0]]
    out = []
    for l in linhas[1:]:
        if not l or all(not x.strip() for x in l):
            continue
        out.append({cab[i]: (l[i].strip() if i < len(l) else "") for i in range(len(cab)) if cab[i]})
    return out


def num(x):
    try:
        v = float(str(x).replace(",", "."))
        return v
    except (TypeError, ValueError):
        return None


def odd(r, *cols):
    """Primeira coluna com odd válida (>1)."""
    for c in cols:
        v = num(r.get(c))
        if v is not None and v > 1.0001:
            return v
    return None


def data_de(s):
    s = (s or "").strip()
    for f in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(s, f).date()
        except ValueError:
            pass
    return None


def justas(odds):
    """Probabilidades sem margem (método proporcional). None se faltar alguma odd."""
    if not odds or any(o is None for o in odds):
        return None
    inv = [1 / o for o in odds]
    s = sum(inv)
    return [x / s for x in inv]


def tres(r, p):
    return [odd(r, f"{p}H"), odd(r, f"{p}D"), odd(r, f"{p}A")]


def ou(r, p):
    return [odd(r, f"{p}>2.5"), odd(r, f"{p}<2.5")]


def mercado_1x2(r, fontes):
    for f in fontes:
        j = justas(tres(r, f))
        if j:
            return j, f
    return None, None


def mercado_ou(r, fontes):
    for f in fontes:
        j = justas(ou(r, f))
        if j:
            return j[0], f
    return None, None


def r2(x, n=3):
    return None if x is None else round(x, n)


# ------------------------------------------------------------------ modelo
class Liga:
    """Ratings por razão simples com decaimento exponencial e regularização."""

    def __init__(self, meia_vida, k):
        self.h = meia_vida
        self.k = k
        self.t = None
        self.W = self.SH = self.SA = 0.0
        self.eq = {}  # nome -> [Wh, Wa, marcados, sofridos, n_jogos]

    def decair(self, t):
        if self.t is not None and t > self.t:
            f = 0.5 ** ((t - self.t) / self.h)
            self.W *= f; self.SH *= f; self.SA *= f
            for v in self.eq.values():
                v[0] *= f; v[1] *= f; v[2] *= f; v[3] *= f
        self.t = t if self.t is None else max(self.t, t)

    def rating(self, nome, f=1.0):
        muH, muA = self.SH / self.W, self.SA / self.W
        v = self.eq.get(nome)
        if not v or v[0] + v[1] <= 0:
            return 1.0, 1.0, 0
        Wh, Wa, s, c, n = v
        w = Wh + Wa
        att_b = s / (muH * Wh + muA * Wa) if (muH * Wh + muA * Wa) > 0 else 1.0
        def_b = c / (muA * Wh + muH * Wa) if (muA * Wh + muH * Wa) > 0 else 1.0
        we = w * f  # peso efetivo na data da previsão
        att = (att_b * we + self.k) / (we + self.k)
        dfn = (def_b * we + self.k) / (we + self.k)
        return att, dfn, n

    def prever(self, casa, fora, t=None):
        if self.W <= 0:
            return None
        f = 1.0 if (t is None or self.t is None) else 0.5 ** ((t - self.t) / self.h)
        muH, muA = self.SH / self.W, self.SA / self.W
        aH, dH, nH = self.rating(casa, f)
        aA, dA, nA = self.rating(fora, f)
        return muH * aH * dA, muA * aA * dH, nH, nA

    def juntar(self, casa, fora, gc, gf):
        self.W += 1; self.SH += gc; self.SA += gf
        for nome, sc, ss, onde in ((casa, gc, gf, 0), (fora, gf, gc, 1)):
            v = self.eq.setdefault(nome, [0.0, 0.0, 0.0, 0.0, 0])
            v[onde] += 1; v[2] += sc; v[3] += ss; v[4] += 1


# ------------------------------------------------------------------ principal
def epoca_atual(hoje):
    y = hoje.year % 100
    return f"{y:02d}{(y + 1) % 100:02d}" if hoje.month >= 7 else f"{(y - 1) % 100:02d}{y:02d}"


def epocas(hoje, anteriores):
    a = epoca_atual(hoje)
    y = int(a[:2])
    return [f"{(y - i) % 100:02d}{(y - i + 1) % 100:02d}" for i in range(anteriores + 1)]


def main():
    saida = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(saida, exist_ok=True)
    with open(os.path.join(AQUI, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    hoje = dt.date.fromisoformat(os.environ["FD_HOJE"]) if os.environ.get("FD_HOJE") else dt.date.today()
    eps = epocas(hoje, int(cfg.get("epocas_anteriores", 2)))
    ligas = {k: v for k, v in cfg["ligas"].items() if v.get("ativa")}
    H, K = float(cfg["meia_vida_dias"]), float(cfg["regularizacao_k"])
    log(f"Hoje {hoje} | épocas {eps} | ligas {list(ligas)}")

    estado = {"ficheiros_ok": 0, "ficheiros_falha": [], "avisos": []}
    jogos_por_liga = {}
    for div in ligas:
        lst = []
        for ep in eps:
            try:
                linhas = ler_csv(obter(f"mmz4281/{ep}/{div}.csv"))
            except Exception as e:  # noqa: BLE001
                estado["ficheiros_falha"].append(f"{div}/{ep}: {type(e).__name__}")
                continue
            estado["ficheiros_ok"] += 1
            for r in linhas:
                d = data_de(r.get("Date"))
                gc, gf = num(r.get("FTHG")), num(r.get("FTAG"))
                casa, fora = r.get("HomeTeam", ""), r.get("AwayTeam", "")
                if not d or gc is None or gf is None or not casa or not fora:
                    continue
                m1x2, f1 = mercado_1x2(r, ("PS", "Avg", "BbAv"))
                mo, fo = mercado_ou(r, ("P", "Avg", "BbAv"))
                c1x2, _ = mercado_1x2(r, ("PSC", "AvgC"))
                co, _ = mercado_ou(r, ("PC", "AvgC"))
                lst.append({
                    "div": div, "ep": ep, "d": d, "casa": casa, "fora": fora, "gc": int(gc), "gf": int(gf),
                    "ap": tres(r, "Avg") + ou(r, "Avg"),
                    "m": (m1x2 or [None] * 3) + [mo],
                    "f": (c1x2 or [None] * 3) + [co],
                })
        lst.sort(key=lambda x: (x["d"], x["casa"]))
        jogos_por_liga[div] = lst
        log(f"  {div}: {len(lst)} jogos")

    if estado["ficheiros_ok"] == 0:
        log("ERRO: nenhum ficheiro descarregado — a publicação anterior mantém-se.")
        sys.exit(1)

    # ---- walk-forward por liga
    modelos, bt, resultados = {}, [], []
    lim_res = hoje - dt.timedelta(days=int(cfg.get("dias_resultados", 120)))
    for div, lst in jogos_por_liga.items():
        L = Liga(H, K)
        i = 0
        while i < len(lst):
            d = lst[i]["d"]
            j = i
            while j < len(lst) and lst[j]["d"] == d:
                j += 1
            t = d.toordinal()
            L.decair(t)
            dia = lst[i:j]
            for m in dia:
                p = L.prever(m["casa"], m["fora"])
                if p:
                    lh, la, nH, nA = p
                    bt.append([div, m["ep"], d.isoformat(), r2(lh), r2(la), min(nH, nA)]
                              + [r2(x, 2) for x in m["ap"]] + [r2(x, 4) for x in m["m"]] + [r2(x, 4) for x in m["f"]]
                              + [m["gc"], m["gf"]])
            for m in dia:
                L.juntar(m["casa"], m["fora"], m["gc"], m["gf"])
                if d >= lim_res:
                    resultados.append([div, d.isoformat(), m["casa"], m["fora"], m["gc"], m["gf"]] + [r2(x, 4) for x in m["f"]])
            i = j
        modelos[div] = L

    # ---- ratings atuais (para jogos manuais / taças)
    t_hoje = hoje.toordinal()
    recentes = {}
    for div, lst in jogos_por_liga.items():
        for m in lst:
            if (hoje - m["d"]).days <= 400:
                recentes.setdefault(div, set()).update((m["casa"], m["fora"]))
    ratings = {}
    for div, L in modelos.items():
        if L.W <= 0:
            continue
        f = 0.5 ** ((t_hoje - L.t) / L.h) if L.t else 1.0
        eq = {}
        for nome in sorted(recentes.get(div, [])):
            a, d_, n = L.rating(nome, f)
            eq[nome] = [r2(a, 4), r2(d_, 4), n]
        ratings[div] = {"muH": r2(L.SH / L.W, 4), "muA": r2(L.SA / L.W, 4), "equipas": eq}

    # ---- próximos jogos
    proximos = []
    try:
        fx = ler_csv(obter("fixtures.csv"))
    except Exception as e:  # noqa: BLE001
        fx = []
        estado["avisos"].append(f"fixtures.csv indisponível ({type(e).__name__}).")
    for r in fx:
        div = r.get("Div", "")
        d = data_de(r.get("Date"))
        if div not in ligas or not d or d < hoje:
            continue
        casa, fora = r.get("HomeTeam", ""), r.get("AwayTeam", "")
        L = modelos.get(div)
        p = L.prever(casa, fora, d.toordinal()) if L else None
        m1x2, f1 = mercado_1x2(r, ("PS", "BFE", "Avg"))
        mo, fo = mercado_ou(r, ("P", "BFE", "Avg"))
        proximos.append({
            "id": f"{div}|{d.isoformat()}|{casa}|{fora}", "div": div, "data": d.isoformat(), "hora": r.get("Time", ""),
            "casa": casa, "fora": fora,
            "lh": r2(p[0]) if p else None, "la": r2(p[1]) if p else None,
            "nH": p[2] if p else 0, "nA": p[3] if p else 0,
            "odds": {
                "avg": [r2(x, 2) for x in tres(r, "Avg") + ou(r, "Avg")],
                "b365": [r2(x, 2) for x in tres(r, "B365") + ou(r, "B365")],
                "max": [r2(x, 2) for x in tres(r, "Max") + ou(r, "Max")],
            },
            "merc": ([r2(x, 4) for x in m1x2] if m1x2 else [None] * 3) + [r2(mo, 4)],
            "fonteMerc": f1 or "",
        })
    proximos.sort(key=lambda x: (x["data"], x["hora"], x["div"], x["casa"]))
    if fx and not proximos:
        estado["avisos"].append("O ficheiro de próximos jogos não tem jogos futuros das ligas ativas.")

    gerado = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    dados = {
        "gerado": gerado, "hoje": hoje.isoformat(), "epocas": eps,
        "config": {"meia_vida_dias": H, "regularizacao_k": K},
        "ligas": {k: {"nome": v["nome"], "pais": v["pais"]} for k, v in ligas.items()},
        "estado": estado, "jogos": proximos, "ratings": ratings,
        "resultados": resultados,
    }
    campos = ["div", "ep", "data", "lh", "la", "nmin", "o1", "oX", "o2", "oO", "oU",
              "m1", "mX", "m2", "mO", "f1", "fX", "f2", "fO", "gc", "gf"]
    with open(os.path.join(saida, "dados.json"), "w", encoding="utf-8") as fh:
        json.dump(dados, fh, ensure_ascii=False, separators=(",", ":"))
    with open(os.path.join(saida, "backtest.json"), "w", encoding="utf-8") as fh:
        json.dump({"gerado": gerado, "campos": campos, "linhas": bt}, fh, ensure_ascii=False, separators=(",", ":"))
    log(f"OK: {len(proximos)} próximos jogos, {len(resultados)} resultados recentes, {len(bt)} linhas de backtest.")
    if estado["ficheiros_falha"]:
        log("Falhas:", ", ".join(estado["ficheiros_falha"]))


if __name__ == "__main__":
    main()
