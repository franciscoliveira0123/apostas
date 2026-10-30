#!/usr/bin/env python3
"""Robô de dados v2 — análise de apostas de futebol.

Modelo
  • Ratings de ataque/defesa ajustados à força dos adversários (modelo de Maher/Dixon-Coles,
    estimado por máxima verosimilhança iterativa), com vantagem casa própria de cada liga.
  • Ponderação temporal (meia-vida) e regularização bayesiana (k jogos fictícios) com prior
    específico para equipas promovidas e despromovidas.
  • Variável de entrada = mistura de golos e remates à baliza (proxy de golos esperados):
    os remates à baliza são mais estáveis do que os golos e preveem melhor o futuro.
  • ρ de Dixon-Coles e peso do modelo face ao mercado (α) estimados por máxima verosimilhança
    APENAS com épocas anteriores (validação fora da amostra).
  • Validação por liga: log-loss face ao mercado, ROI e CLV das estratégias "modelo" e "preço".

Saídas
  dados.json     -> próximos jogos (probabilidades, odds, mercado justo), ratings, validação, resultados
  backtest.json  -> previsões walk-forward das épocas avaliadas (fora da amostra)

Uso:  python atualizar.py <pasta_saida>
Teste local:  FD_LOCAL=<pasta> FD_HOJE=AAAA-MM-DD python atualizar.py saida
Só usa a biblioteca padrão do Python.
"""
import csv, io, json, math, os, sys, time, datetime as dt, urllib.request

BASE = "https://www.football-data.co.uk/"
AQUI = os.path.dirname(os.path.abspath(__file__))
MIN_JOGOS = 6


def log(*a):
    print(*a, flush=True)


# ================================================================== leitura
def obter(caminho):
    local = os.environ.get("FD_LOCAL")
    if local:
        with open(os.path.join(local, caminho), "rb") as fh:
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
        return float(str(x).replace(",", "."))
    except (TypeError, ValueError):
        return None


def odd(r, *cols):
    for c in cols:
        v = num(r.get(c))
        if v is not None and v > 1.0001:
            return v
    return None


def cinco(r, *prefixos):
    """[1, X, 2, +2,5, −2,5] com o primeiro prefixo disponível em cada mercado.
    Pinnacle usa PSH/PSD/PSA no 1X2 e P>2.5 nos golos."""
    out = []
    for suf in ("H", "D", "A", ">2.5", "<2.5"):
        cols = []
        for p in prefixos:
            if p in ("PS", "PSC") and suf.startswith((">", "<")):
                cols.append(("P" if p == "PS" else "PC") + suf)
            else:
                cols.append(p + suf)
        out.append(odd(r, *cols))
    return out


def justas(odds):
    if not odds or any(o is None for o in odds):
        return None
    inv = [1 / o for o in odds]
    s = sum(inv)
    return [x / s for x in inv]


def justas4(o5):
    a, b = justas(o5[:3]), justas(o5[3:5])
    return (a or [None] * 3) + [b[0] if b else None]


def data_de(s):
    s = (s or "").strip()
    for f in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(s, f).date()
        except ValueError:
            pass
    return None


def r2(x, n=3):
    return None if x is None else round(x, n)


# ================================================================== probabilidades
FACT = [math.factorial(k) for k in range(11)]


def pmf(l):
    e = math.exp(-l)
    return [e * l ** k / FACT[k] for k in range(11)]


def tau(x, y, lh, la, rho):
    if x == 0 and y == 0:
        return 1 - lh * la * rho
    if x == 0 and y == 1:
        return 1 + lh * rho
    if x == 1 and y == 0:
        return 1 + la * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def dc_probs(lh, la, rho):
    """[P1, PX, P2, P+2,5] com correção de Dixon-Coles."""
    ph, pa = pmf(lh), pmf(la)
    h = d = a = o = s = 0.0
    for i in range(11):
        for j in range(11):
            x = ph[i] * pa[j]
            if i < 2 and j < 2:
                x *= max(tau(i, j, lh, la, rho), 0.0)
            s += x
            if i > j:
                h += x
            elif i == j:
                d += x
            else:
                a += x
            if i + j > 2:
                o += x
    return [h / s, d / s, a / s, o / s]


def cinco_p(p4):
    if p4 is None or any(v is None for v in p4[:3]):
        return [None] * 5 if p4 is None else list(p4[:3]) + [p4[3], None if p4[3] is None else 1 - p4[3]]
    return list(p4[:3]) + [p4[3], None if p4[3] is None else 1 - p4[3]]


def estimar_rho(linhas):
    """Máxima verosimilhança do ρ (só os resultados 0-0, 1-0, 0-1 e 1-1 dependem dele)."""
    melhor, best = -0.08, -1e18
    for k in range(-25, 11):
        rho = k / 100
        ll = 0.0
        ok = True
        for lh, la, gc, gf in linhas:
            if gc < 2 and gf < 2:
                t = tau(gc, gf, lh, la, rho)
                if t <= 0:
                    ok = False
                    break
                ll += math.log(t)
        if ok and ll > best:
            best, melhor = ll, rho
    return melhor


def estimar_alfa(pares):
    """α que minimiza a log-loss de α·modelo + (1−α)·mercado. pares = [(p_modelo, p_mercado)] do resultado ocorrido."""
    if len(pares) < 200:
        return 0.3
    melhor, best = 0.3, 1e18
    for k in range(0, 21):
        a = k / 20
        ll = -sum(math.log(max(1e-9, a * pm + (1 - a) * pk)) for pm, pk in pares)
        if ll < best:
            best, melhor = ll, a
    return melhor


# ================================================================== modelo de ratings
class Liga:
    def __init__(self, h, k, wg):
        self.h, self.k, self.wg = h, k, wg
        self.m = []          # (t, i, j, gc, gf, rc, rf)  rc/rf = remates à baliza (ou None)
        self.idx, self.nomes = {}, []
        self.att, self.dfn, self.pa, self.pd, self.n = [], [], [], [], []
        self.muH, self.muA, self.c = 1.4, 1.1, None
        self.ini = 0
        self.ajustado = False

    def equipa(self, nome, prior=(1.0, 1.0)):
        if nome not in self.idx:
            self.idx[nome] = len(self.nomes)
            self.nomes.append(nome)
            self.att.append(prior[0]); self.dfn.append(prior[1])
            self.pa.append(prior[0]); self.pd.append(prior[1]); self.n.append(0)
        return self.idx[nome]

    def novo_prior(self, nome, prior):
        i = self.equipa(nome, prior)
        self.pa[i], self.pd[i] = prior
        self.att[i], self.dfn[i] = prior

    def juntar(self, t, casa, fora, gc, gf, rc, rf):
        i, j = self.equipa(casa), self.equipa(fora)
        self.m.append((t, i, j, gc, gf, rc, rf))
        self.n[i] += 1; self.n[j] += 1

    def ajustar(self, t, iters):
        h = self.h
        while self.ini < len(self.m) and t - self.m[self.ini][0] > 5 * h:
            self.ini += 1
        M = [m for m in self.m[self.ini:] if m[0] < t]
        if len(M) < 20:
            return False
        w = [0.5 ** ((t - m[0]) / h) for m in M]
        sg = ss = 0.0
        for wi, m in zip(w, M):
            if m[5] is not None and m[6] is not None:
                sg += wi * (m[3] + m[4]); ss += wi * (m[5] + m[6])
        c = sg / ss if ss > 0 else None
        wg = self.wg
        Y = []
        for m in M:
            if c is not None and m[5] is not None and m[6] is not None:
                Y.append((wg * m[3] + (1 - wg) * c * m[5], wg * m[4] + (1 - wg) * c * m[6]))
            else:
                Y.append((m[3], m[4]))
        T = len(self.nomes)
        att, dfn = self.att[:], self.dfn[:]
        ativos = set()
        for m in M:
            ativos.add(m[1]); ativos.add(m[2])
        k = self.k
        muH, muA = self.muH, self.muA
        for _ in range(iters):
            nh = dh = na = da = 0.0
            for wi, m, (yh, ya) in zip(w, M, Y):
                i, j = m[1], m[2]
                nh += wi * yh; dh += wi * att[i] * dfn[j]
                na += wi * ya; da += wi * att[j] * dfn[i]
            muH, muA = nh / dh, na / da
            E = k * (muH + muA) / 2
            num_ = [E * self.pa[x] for x in range(T)]; den = [E] * T
            for wi, m, (yh, ya) in zip(w, M, Y):
                i, j = m[1], m[2]
                num_[i] += wi * yh; den[i] += wi * muH * dfn[j]
                num_[j] += wi * ya; den[j] += wi * muA * dfn[i]
            att = [num_[x] / den[x] for x in range(T)]
            num_ = [E * self.pd[x] for x in range(T)]; den = [E] * T
            for wi, m, (yh, ya) in zip(w, M, Y):
                i, j = m[1], m[2]
                num_[i] += wi * ya; den[i] += wi * muA * att[j]
                num_[j] += wi * yh; den[j] += wi * muH * att[i]
            dfn = [num_[x] / den[x] for x in range(T)]
            if ativos:
                sa = sum(att[x] for x in ativos) / len(ativos)
                sd = sum(dfn[x] for x in ativos) / len(ativos)
                att = [v / sa for v in att]; dfn = [v / sd for v in dfn]
        nh = dh = na = da = 0.0
        for wi, m, (yh, ya) in zip(w, M, Y):
            i, j = m[1], m[2]
            nh += wi * yh; dh += wi * att[i] * dfn[j]
            na += wi * ya; da += wi * att[j] * dfn[i]
        self.muH, self.muA = nh / dh, na / da
        self.att, self.dfn, self.c = att, dfn, c
        self.ajustado = True
        return True

    def lam(self, casa, fora):
        i, j = self.idx[casa], self.idx[fora]
        return self.muH * self.att[i] * self.dfn[j], self.muA * self.att[j] * self.dfn[i], min(self.n[i], self.n[j])


PRIOR_PROMOVIDA = (0.85, 1.15)
PRIOR_DESPROMOVIDA = (1.12, 0.90)


# ================================================================== principal
def epocas(hoje, anteriores):
    y = hoje.year % 100
    a = y if hoje.month >= 7 else (y - 1) % 100
    return [f"{(a - i) % 100:02d}{(a - i + 1) % 100:02d}" for i in range(anteriores, -1, -1)]  # ordem cronológica


def main():
    t0 = time.time()
    saida = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(saida, exist_ok=True)
    with open(os.path.join(AQUI, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    hoje = dt.date.fromisoformat(os.environ["FD_HOJE"]) if os.environ.get("FD_HOJE") else dt.date.today()
    eps = epocas(hoje, int(cfg.get("epocas_anteriores", 3)))
    ligas = {k: v for k, v in cfg["ligas"].items() if v.get("ativa")}
    H, K, WG = float(cfg["meia_vida_dias"]), float(cfg["regularizacao_k"]), float(cfg.get("peso_golos", 0.5))
    VAL = cfg.get("validacao", {})
    log(f"Hoje {hoje} | épocas {eps} | {len(ligas)} ligas | meia-vida {H} | k {K} | peso golos {WG}")

    # ---------------------------------------------------------- download
    estado = {"ficheiros_ok": 0, "ficheiros_falha": [], "avisos": []}
    jogos = {}
    equipas_ep = {}  # (div, ep) -> set
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
                ps = cinco(r, "PS")
                sharp = justas(ps[:3]) is not None
                mk = justas4(ps) if sharp else justas4(cinco(r, "Avg", "BbAv"))
                if mk[3] is None:
                    mk[3] = justas4(cinco(r, "Avg", "BbAv"))[3]
                fc = justas4(cinco(r, "PSC"))
                if fc[0] is None:
                    fc = justas4(cinco(r, "AvgC"))
                rc, rf = num(r.get("HST")), num(r.get("AST"))
                lst.append({"div": div, "ep": ep, "d": d, "t": d.toordinal(), "casa": casa, "fora": fora,
                            "gc": int(gc), "gf": int(gf), "rc": rc, "rf": rf,
                            "avg": cinco(r, "Avg", "BbAv"), "max": cinco(r, "Max", "BbMx"),
                            "mk": mk, "sharp": sharp, "fc": fc})
                equipas_ep.setdefault((div, ep), set()).update((casa, fora))
        lst.sort(key=lambda x: (x["t"], x["casa"]))
        jogos[div] = lst
        log(f"  {div}: {len(lst)} jogos")
    if estado["ficheiros_ok"] == 0:
        log("ERRO: nenhum ficheiro descarregado — a publicação anterior mantém-se.")
        sys.exit(1)

    def prior_para(div, ep, nome):
        i = eps.index(ep)
        if i == 0 or (div, eps[i - 1]) not in equipas_ep:
            return (1.0, 1.0)
        pais, nivel = ligas[div]["pais"], ligas[div].get("nivel", 1)
        for d2, v in ligas.items():
            if v["pais"] == pais and v.get("nivel", 1) < nivel and nome in equipas_ep.get((d2, eps[i - 1]), ()):
                return PRIOR_DESPROMOVIDA
        return PRIOR_PROMOVIDA

    # ---------------------------------------------------------- walk-forward
    modelos, previsoes = {}, []
    for div, lst in jogos.items():
        L = Liga(H, K, WG)
        ep_atual = None
        i = 0
        while i < len(lst):
            t = lst[i]["t"]
            j = i
            while j < len(lst) and lst[j]["t"] == t:
                j += 1
            dia = lst[i:j]
            if dia[0]["ep"] != ep_atual:
                ep_atual = dia[0]["ep"]
                ant = eps[eps.index(ep_atual) - 1] if eps.index(ep_atual) > 0 else None
                for nome in equipas_ep.get((div, ep_atual), ()):
                    if ant is None or nome not in equipas_ep.get((div, ant), ()):
                        L.novo_prior(nome, prior_para(div, ep_atual, nome))
            if L.ajustar(t, 25 if not L.ajustado else 4):
                for m in dia:
                    lh, la, nmin = L.lam(m["casa"], m["fora"])
                    m["lh"], m["la"], m["nmin"] = lh, la, nmin
                    previsoes.append(m)
            for m in dia:
                L.juntar(t, m["casa"], m["fora"], m["gc"], m["gf"], m["rc"], m["rf"])
            i = j
        modelos[div] = L
    log(f"Walk-forward: {len(previsoes)} previsões em {time.time() - t0:.0f}s")

    # ---------------------------------------------------------- ρ e α fora da amostra (por época)
    def resultado(m):
        gc, gf = m["gc"], m["gf"]
        return (0 if gc > gf else 1 if gc == gf else 2), (1 if gc + gf > 2 else 0)

    def treinar(lista):
        lin = [(m["lh"], m["la"], m["gc"], m["gf"]) for m in lista]
        rho = estimar_rho(lin) if len(lin) >= 300 else -0.08
        p1, po = [], []
        for m in lista:
            pm = dc_probs(m["lh"], m["la"], rho)
            k1, ko = resultado(m)
            if m["mk"][0] is not None:
                p1.append((pm[k1], m["mk"][k1]))
            if m["mk"][3] is not None:
                po.append((pm[3] if ko else 1 - pm[3], m["mk"][3] if ko else 1 - m["mk"][3]))
        return rho, estimar_alfa(p1), estimar_alfa(po)

    elegiveis = [m for m in previsoes if m["nmin"] >= MIN_JOGOS]
    params_ep = {}
    for k, ep in enumerate(eps):
        if k == 0:
            params_ep[ep] = None
            continue
        params_ep[ep] = treinar([m for m in elegiveis if eps.index(m["ep"]) < k])
        log(f"  época {ep}: ρ={params_ep[ep][0]:+.2f}  α1X2={params_ep[ep][1]:.2f}  αgolos={params_ep[ep][2]:.2f}")
    rho_f, a1_f, ao_f = treinar(elegiveis)
    log(f"Parâmetros atuais: ρ={rho_f:+.2f}  α1X2={a1_f:.2f}  αgolos={ao_f:.2f}")

    def final(pm4, mk4, a1, ao):
        out = []
        for i in range(4):
            a = a1 if i < 3 else ao
            out.append(a * pm4[i] + (1 - a) * mk4[i] if mk4[i] is not None else pm4[i])
        if mk4[0] is None or mk4[1] is None or mk4[2] is None:
            out[:3] = pm4[:3]
        return out

    # ---------------------------------------------------------- backtest + validação
    bt, val = [], {}
    vmin, omax = float(VAL.get("ev_min_modelo", 0.03)), float(VAL.get("odd_max", 4.0))
    pmin = float(VAL.get("ev_min_preco", 0.02))

    def acum(chave):
        return val.setdefault(chave, {"n": 0, "llM": 0.0, "llF": 0.0, "llK": 0.0, "llC": 0.0, "d": 0.0, "d2": 0.0,
                                      "A": [0, 0.0, 0.0, 0], "B": [0, 0.0, 0.0, 0]})

    for m in previsoes:
        p = params_ep.get(m["ep"])
        if p is None:
            continue
        rho, a1, ao = p
        pm = dc_probs(m["lh"], m["la"], rho)
        pf = final(pm, m["mk"], a1, ao)
        m["pm"], m["pf"] = pm, pf
        bt.append([m["div"], m["ep"], m["d"].isoformat(), m["nmin"]] + [r2(x) for x in pm] + [r2(x) for x in pf]
                  + [r2(x) for x in m["mk"]] + [1 if m["sharp"] else 0] + [r2(x) for x in m["fc"]]
                  + [r2(x, 2) for x in m["avg"]] + [r2(x, 2) for x in m["max"]] + [m["gc"], m["gf"]])
        if m["nmin"] < MIN_JOGOS or None in m["mk"][:3] or None in m["fc"][:3]:
            continue
        k1, ko = resultado(m)
        y = [k1 == 0, k1 == 1, k1 == 2, ko == 1, ko == 0]
        P5, F5, K5, C5 = cinco_p(pm), cinco_p(pf), cinco_p(m["mk"]), cinco_p(m["fc"])
        for chave in (m["div"], "TOTAL"):
            v = acum(chave)
            v["n"] += 1
            v["llM"] -= math.log(max(1e-9, pm[k1])); v["llF"] -= math.log(max(1e-9, pf[k1]))
            v["llK"] -= math.log(max(1e-9, m["mk"][k1])); v["llC"] -= math.log(max(1e-9, m["fc"][k1]))
            dif = math.log(max(1e-9, m["mk"][k1])) - math.log(max(1e-9, pf[k1]))  # >0 = modelo final pior que o mercado
            v["d"] += dif; v["d2"] += dif * dif
            # Estratégia A — modelo combinado vs odd média
            bi, be = -1, -9
            for i in range(5):
                o = m["avg"][i]
                if o and o <= omax and F5[i] is not None:
                    e = F5[i] * o - 1
                    if e > be:
                        be, bi = e, i
            if bi >= 0 and be >= vmin:
                A = v["A"]; A[0] += 1; A[1] += (m["avg"][bi] - 1) if y[bi] else -1
                if C5[bi] is not None:
                    A[2] += m["avg"][bi] * C5[bi] - 1; A[3] += 1
            # Estratégia B — melhor odd vs Pinnacle sem margem
            if m["sharp"]:
                bi, be = -1, -9
                for i in range(5):
                    o = m["max"][i]
                    if o and o <= omax and K5[i] is not None:
                        e = K5[i] * o - 1
                        if e > be:
                            be, bi = e, i
                if bi >= 0 and be >= pmin:
                    B = v["B"]; B[0] += 1; B[1] += (m["max"][bi] - 1) if y[bi] else -1
                    if C5[bi] is not None:
                        B[2] += m["max"][bi] * C5[bi] - 1; B[3] += 1

    validacao = {}
    nap = int(VAL.get("apostas_min", 80))
    for chave, v in val.items():
        n = v["n"]
        A, B = v["A"], v["B"]
        d = {"n": n, "llModelo": r2(v["llM"] / n, 4), "llFinal": r2(v["llF"] / n, 4), "llMercado": r2(v["llK"] / n, 4),
             "llFecho": r2(v["llC"] / n, 4),
             "A": {"n": A[0], "roi": r2(A[1] / A[0], 4) if A[0] else None, "clv": r2(A[2] / A[3], 4) if A[3] else None},
             "B": {"n": B[0], "roi": r2(B[1] / B[0], 4) if B[0] else None, "clv": r2(B[2] / B[3], 4) if B[3] else None}}
        md = v["d"] / n
        se = math.sqrt(max(0.0, v["d2"] / n - md * md) / n) if n > 1 else 1.0
        d["ganhoLL"] = r2(-md, 5)      # ganho médio de log-loss face ao mercado de abertura (positivo = melhor)
        d["ganhoLLse"] = r2(se, 5)
        # Validada só se o ganho for > 2 erros-padrão, com amostra suficiente e CLV positivo na estratégia do modelo
        d["validada"] = bool(A[0] >= nap and A[3] and A[2] / A[3] > 0 and -md > 2 * se)
        validacao[chave] = d
    log("Validação (log-loss final vs mercado | CLV A | CLV B):")
    for chave in sorted(validacao, key=lambda c: (c == "TOTAL", c)):
        d = validacao[chave]
        log(f"  {chave:6s} n={d['n']:5d}  LL final {d['llFinal']} vs mercado {d['llMercado']} (ganho {d['ganhoLL']:+.4f} ± {2*d['ganhoLLse']:.4f})  A:{d['A']}  B:{d['B']}  {'VALIDADA' if d['validada'] else ''}")

    # ---------------------------------------------------------- próximos jogos
    proximos = []
    try:
        fx = ler_csv(obter("fixtures.csv"))
    except Exception as e:  # noqa: BLE001
        fx = []
        estado["avisos"].append(f"fixtures.csv indisponível ({type(e).__name__}).")
    por_liga_data = {}
    for r in fx:
        div = r.get("Div", "")
        d = data_de(r.get("Date"))
        if div in ligas and d and d >= hoje and r.get("HomeTeam") and r.get("AwayTeam"):
            por_liga_data.setdefault((div, d), []).append(r)
    for (div, d), lst in sorted(por_liga_data.items(), key=lambda x: (x[0][1], x[0][0])):
        L = modelos.get(div)
        ok = False
        if L:
            for r in lst:
                for nome in (r["HomeTeam"], r["AwayTeam"]):
                    if nome not in L.idx:
                        L.novo_prior(nome, PRIOR_PROMOVIDA if len(L.m) else (1.0, 1.0))
            ok = L.ajustar(d.toordinal(), 6)
        for r in lst:
            casa, fora = r["HomeTeam"], r["AwayTeam"]
            ps = cinco(r, "PS")
            if justas(ps[:3]):
                pk, fonte = justas4(ps), "PS"
            elif justas(cinco(r, "BFE")[:3]):
                pk, fonte = justas4(cinco(r, "BFE")), "BFE"
            else:
                pk, fonte = justas4(cinco(r, "Avg")), "Avg"
            if ok:
                lh, la, nmin = L.lam(casa, fora)
                pm = dc_probs(lh, la, rho_f)
                pf = final(pm, pk, a1_f, ao_f)
            else:
                lh = la = None; nmin = 0; pm = pf = None
            proximos.append({
                "id": f"{div}|{d.isoformat()}|{casa}|{fora}", "div": div, "data": d.isoformat(), "hora": r.get("Time", ""),
                "casa": casa, "fora": fora, "lh": r2(lh), "la": r2(la), "nmin": nmin,
                "pm": [r2(x, 4) for x in cinco_p(pm)] if pm else None,
                "pf": [r2(x, 4) for x in cinco_p(pf)] if pf else None,
                "pk": [r2(x, 4) for x in cinco_p(pk)], "fonte": fonte,
                "odds": {"avg": [r2(x, 2) for x in cinco(r, "Avg")], "b365": [r2(x, 2) for x in cinco(r, "B365")],
                         "max": [r2(x, 2) for x in cinco(r, "Max")]},
            })
    proximos.sort(key=lambda x: (x["data"], x["hora"], x["div"], x["casa"]))
    if fx and not proximos:
        estado["avisos"].append("O ficheiro de próximos jogos ainda não tem jogos futuros das ligas ativas.")

    # ---------------------------------------------------------- ratings atuais (jogos manuais)
    ratings = {}
    for div, L in modelos.items():
        if not L.ajustar(hoje.toordinal() + 1, 6):
            continue
        recentes = set()
        for m in jogos[div]:
            if (hoje - m["d"]).days <= 400:
                recentes.update((m["casa"], m["fora"]))
        ratings[div] = {"muH": r2(L.muH, 4), "muA": r2(L.muA, 4),
                        "equipas": {n: [r2(L.att[L.idx[n]], 4), r2(L.dfn[L.idx[n]], 4), L.n[L.idx[n]]] for n in sorted(recentes) if n in L.idx}}

    # ---------------------------------------------------------- resultados recentes (liquidação)
    lim = hoje - dt.timedelta(days=int(cfg.get("dias_resultados", 120)))
    resultados = [[m["div"], m["d"].isoformat(), m["casa"], m["fora"], m["gc"], m["gf"]] + [r2(x, 4) for x in m["fc"]]
                  for div in jogos for m in jogos[div] if m["d"] >= lim]

    gerado = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    dados = {
        "versao": 2, "gerado": gerado, "hoje": hoje.isoformat(), "epocas": eps,
        "parametros": {"rho": rho_f, "alfa1x2": a1_f, "alfaGolos": ao_f, "meiaVida": H, "k": K, "pesoGolos": WG,
                       "porEpoca": {ep: (None if p is None else {"rho": p[0], "alfa1x2": p[1], "alfaGolos": p[2]}) for ep, p in params_ep.items()},
                       "validacao": VAL},
        "ligas": {k: {"nome": v["nome"], "pais": v["pais"]} for k, v in ligas.items()},
        "estado": estado, "validacao": validacao, "jogos": proximos, "ratings": ratings, "resultados": resultados,
    }
    campos = ["div", "ep", "data", "nmin", "pm1", "pmX", "pm2", "pmO", "pf1", "pfX", "pf2", "pfO",
              "mk1", "mkX", "mk2", "mkO", "sharp", "fc1", "fcX", "fc2", "fcO",
              "a1", "aX", "a2", "aO", "aU", "x1", "xX", "x2", "xO", "xU", "gc", "gf"]
    with open(os.path.join(saida, "dados.json"), "w", encoding="utf-8") as fh:
        json.dump(dados, fh, ensure_ascii=False, separators=(",", ":"))
    with open(os.path.join(saida, "backtest.json"), "w", encoding="utf-8") as fh:
        json.dump({"gerado": gerado, "campos": campos, "linhas": bt}, fh, ensure_ascii=False, separators=(",", ":"))
    log(f"OK: {len(proximos)} próximos jogos, {len(resultados)} resultados recentes, {len(bt)} linhas de backtest, {time.time() - t0:.0f}s.")
    if estado["ficheiros_falha"]:
        log("Falhas:", ", ".join(estado["ficheiros_falha"]))


if __name__ == "__main__":
    main()
