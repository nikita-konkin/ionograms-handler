"""Draw the simulator's figures, in English and in Russian.

    python docs/figures/build.py

writes ``simulator.svg``, ``related.svg`` and their ``.ru.svg`` twins next
to this file: the hybrid simulator's flow, and how it compares with the HF
channel simulators before it (``docs/simulator.md``, "Related work"). The
SVGs carry their own light and dark palettes, so they read on either theme
of a Markdown viewer.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent

STYLE = """
  <style>
    svg { --bg: #FFFFFF; --ink: #2C2C2A; --muted: #5F5E5A; --line: #888780;
          --teal-fill: #E1F5EE; --teal-stroke: #0F6E56; --teal-text: #085041;
          --purple-fill: #EEEDFE; --purple-stroke: #534AB7; --purple-text: #3C3489;
          --gray-fill: #F1EFE8; --gray-stroke: #5F5E5A; --gray-text: #444441; }
    @media (prefers-color-scheme: dark) {
      svg { --bg: #1F1F1E; --ink: #E8E6E0; --muted: #B4B2A9; --line: #888780;
            --teal-fill: #085041; --teal-stroke: #5DCAA5; --teal-text: #9FE1CB;
            --purple-fill: #3C3489; --purple-stroke: #AFA9EC; --purple-text: #CECBF6;
            --gray-fill: #444441; --gray-stroke: #B4B2A9; --gray-text: #D3D1C7; }
    }
    .bg { fill: var(--bg); }
    text { font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
           fill: var(--ink); }
    .title { font-size: 14px; font-weight: 600; }
    .sub { font-size: 12px; fill: var(--muted); }
    .head { font-size: 13px; font-weight: 600; fill: var(--muted); }
    .lane { font-size: 13px; font-weight: 600; }
    .teal rect { fill: var(--teal-fill); stroke: var(--teal-stroke); }
    .teal .title { fill: var(--teal-text); }
    .purple rect { fill: var(--purple-fill); stroke: var(--purple-stroke); }
    .purple .title { fill: var(--purple-text); }
    .gray rect { fill: var(--gray-fill); stroke: var(--gray-stroke); }
    .gray .title { fill: var(--gray-text); }
    .plan rect { fill: none; stroke: var(--gray-stroke); stroke-dasharray: 5 4; }
    .plan .title { fill: var(--muted); }
    .arrow { stroke: var(--line); stroke-width: 1.5; fill: none;
             marker-end: url(#head); }
    .loop { stroke: var(--teal-stroke); stroke-width: 1.5; fill: none;
            stroke-dasharray: 6 4; marker-end: url(#head-loop); }
    .dashed { stroke-dasharray: 5 4; }
    #head path { fill: var(--line); }
    #head-loop path { fill: var(--teal-stroke); }
  </style>"""

DEFS = """
  <defs>
    <marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"
            markerHeight="7" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z"/></marker>
    <marker id="head-loop" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7"
            markerHeight="7" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z"/></marker>
  </defs>"""


def box(kind: str, x: float, y: float, w: float, h: float, title: str,
        sub: str = "") -> str:
    cx = x + w / 2
    ty = y + (h / 2 - 3 if sub else h / 2 + 5)
    out = [f'  <g class="{kind}">',
           f'    <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" '
           f'stroke-width="1.5"/>',
           f'    <text class="title" x="{cx}" y="{ty}" text-anchor="middle">'
           f'{escape(title)}</text>']
    if sub:
        out.append(f'    <text class="sub" x="{cx}" y="{ty + 18}" text-anchor="middle">'
                   f'{escape(sub)}</text>')
    out.append("  </g>")
    return "\n".join(out)


def arrow(d: str, cls: str = "arrow") -> str:
    return f'  <path class="{cls}" d="{d}"/>'


def svg(width: int, height: int, title: str, body: list[str]) -> str:
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{escape(title)}">',
        f"  <title>{escape(title)}</title>", STYLE, DEFS,
        f'  <rect class="bg" x="0" y="0" width="{width}" height="{height}"/>',
        *body, "</svg>", ""])


def legend(y: float, items: list[tuple[str, str]]) -> list[str]:
    out, x = [], 40.0
    for kind, label in items:
        if kind == "loop":
            out.append(arrow(f"M{x} {y + 7} L{x + 26} {y + 7}", "loop"))
        else:
            out.append(f'  <g class="{kind}"><rect x="{x}" y="{y}" width="26" '
                       f'height="14" rx="3" stroke-width="1.5"/></g>')
        out.append(f'  <text class="sub" x="{x + 34}" y="{y + 11}">'
                   f'{escape(label)}</text>')
        x += 34 + 7.2 * len(label) + 28
    return out


# --- the simulator ---------------------------------------------------------------

SIMULATOR = {
    "en": {
        "title": "muf.sim: the hybrid HF channel simulator",
        "real": ("Real sounding", ".lfs or .h5"),
        "forecast": ("Forecast", "stage 3, planned"),
        "prior": ("Prior models", "F.1487, b = 7.52 µs/MHz"),
        "muf": ("muf processing", "trace, modes, MUF"),
        "fast": ("Fast part from phase", "spread, Rice K, o/x"),
        "state": ("Channel state", "modes: τ(f), C/N₀(f), Doppler, fading, o/x"),
        "sounding": ("Synthetic sounding", "chirp path, .lfs file"),
        "channel": ("Channel for a signal", "dispersion, fading, noise"),
        "again": ("Reprocessing", "the same muf"),
        "signal": ("Output signal", "modems, equalisers"),
        "compare": ("State comparison", "levels 1–2: loop, realism"),
        "theory": ("Check against theory", "level 0: F.1487, Rice, DBPSK"),
        "legend": [("teal", "measured"), ("purple", "prior or model"),
                   ("plan", "planned"), ("loop", "closed loop")],
    },
    "ru": {
        "title": "muf.sim: гибридный имитатор КВ-канала",
        "real": ("Реальный сеанс", ".lfs или .h5"),
        "forecast": ("Прогноз", "этап 3, план"),
        "prior": ("Априорные модели", "F.1487, b = 7,52 мкс/МГц"),
        "muf": ("Обработка muf", "след, моды, МНЧ"),
        "fast": ("Быстрая часть по фазе", "разброс, K Райса, o/x"),
        "state": ("Состояние канала", "моды: τ(f), C/N₀(f), доплер, замирания, o/x"),
        "sounding": ("Синтетический сеанс", "ЛЧМ-тракт, файл .lfs"),
        "channel": ("Канал для сигнала", "дисперсия, замирания, шум"),
        "again": ("Повторная обработка", "тот же muf"),
        "signal": ("Сигнал на выходе", "модемы, эквалайзеры"),
        "compare": ("Сравнение состояний", "уровни 1–2: цикл, реализм"),
        "theory": ("Проверка по теории", "уровень 0: F.1487, Райс, DBPSK"),
        "legend": [("teal", "измерено"), ("purple", "априори или модель"),
                   ("plan", "запланировано"), ("loop", "замкнутый цикл")],
    },
}


def simulator(lang: str) -> str:
    t = SIMULATOR[lang]
    h = 56
    body = [
        box("teal", 40, 40, 180, h, *t["real"]),
        box("plan", 250, 40, 180, h, *t["forecast"]),
        box("purple", 460, 40, 180, h, *t["prior"]),
        box("teal", 40, 130, 180, h, *t["muf"]),
        box("teal", 250, 130, 180, h, *t["fast"]),
        box("gray", 40, 230, 600, h, *t["state"]),
        box("gray", 40, 330, 280, h, *t["sounding"]),
        box("gray", 360, 330, 280, h, *t["channel"]),
        box("gray", 40, 430, 280, h, *t["again"]),
        box("gray", 360, 430, 280, h, *t["signal"]),
        box("gray", 40, 530, 280, h, *t["compare"]),
        box("gray", 360, 530, 280, h, *t["theory"]),
        # Into the state.
        arrow("M130 96 L130 128"),
        arrow("M190 96 L190 112 L340 112 L340 128"),
        arrow("M220 158 L248 158"),
        arrow("M130 186 L130 228"),
        arrow("M340 186 L340 228"),
        arrow("M430 68 L445 68 L445 228", "arrow dashed"),
        arrow("M550 96 L550 228"),
        # Out of it.
        arrow("M180 286 L180 328"),
        arrow("M500 286 L500 328"),
        arrow("M180 386 L180 428"),
        arrow("M500 386 L500 428"),
        arrow("M180 486 L180 528"),
        arrow("M500 486 L500 528"),
        # The closed loop: the state recovered against the state sounded.
        arrow("M40 558 L22 558 L22 258 L38 258", "loop"),
        *legend(622, t["legend"]),
    ]
    return svg(680, 660, t["title"], body)


# --- related work ------------------------------------------------------------------

RELATED = {
    "en": {
        "title": "HF channel simulators compared",
        "heads": ["Parameters from", "Channel described as", "Synthesis",
                  "Verification"],
        "lanes": [
            ("teal", "muf.sim (this work)", [
                ("teal", "Chirp sounding", "the same circuit"),
                ("teal", "Mode curves", "τ(f), C/N₀(f)"),
                ("teal", "Sounding, signal", "full dispersion"),
                ("teal", "Through muf", "closed loop")]),
            ("gray", "Watterson, ITU-R F.1487", [
                ("gray", "A table", "path conditions"),
                ("gray", "2 paths", "no dispersion"),
                ("gray", "Two taps", "Gaussian Doppler"),
                ("gray", "None", "band ≤ 12 kHz")]),
            ("gray", "Vogler–Hoffmeyer, ITS (1988–1997)", [
                ("purple", "NRL sounder", "and ionograms"),
                ("gray", "≤ 3 modes", "sech², slant"),
                ("gray", "H(t, f)", "1 MHz, hardware"),
                ("gray", "Scattering", "vs measurements")]),
            ("gray", "Physical: Gherm–Zernov, HiCIRF, Lv 2022, ISTP", [
                ("gray", "Medium model", "IRI, fluctuations"),
                ("gray", "Rays and Rytov", "o and x modes"),
                ("gray", "Realisations", "E(t), S(τ, ν)"),
                ("gray", "Comparison", "with sounding")]),
            ("gray", "DAMSON, Warrington", [
                ("purple", "DAMSON sounder", "S(τ, ν), 3–12 kHz"),
                ("gray", "Spreads", "τ and Doppler"),
                ("gray", "Watterson+", "Lorentzian Doppler"),
                ("gray", "Modems", "availability")]),
            ("gray", "Volgatech school: Ivanov, Ryabova", [
                ("purple", "Chirp sounder", "GDD and IRI"),
                ("gray", "Dispersion", "trend and noise"),
                ("gray", "Channel IR", "chirp, losses"),
                ("gray", "Calculation", "and correction")]),
        ],
        "legend": [("teal", "muf.sim"), ("purple", "parameters from measurements"),
                   ("gray", "a table or a medium model")],
    },
    "ru": {
        "title": "Имитаторы КВ-канала: сравнение",
        "heads": ["Откуда параметры", "Описание канала", "Синтез", "Проверка"],
        "lanes": [
            ("teal", "muf.sim (эта работа)", [
                ("teal", "ЛЧМ-сеанс", "та же трасса"),
                ("teal", "Кривые мод", "τ(f), C/N₀(f)"),
                ("teal", "Сеанс, сигнал", "дисперсия full"),
                ("teal", "Через muf", "замкнутый цикл")]),
            ("gray", "Уоттерсон, ITU-R F.1487", [
                ("gray", "Таблица", "условия трассы"),
                ("gray", "2 луча", "без дисперсии"),
                ("gray", "Два отвода", "гауссов доплер"),
                ("gray", "Нет", "полоса ≤ 12 кГц")]),
            ("gray", "Фоглер–Хоффмейер, ITS (1988–1997)", [
                ("purple", "Зонд NRL", "и ионограммы"),
                ("gray", "≤ 3 моды", "sech², slant"),
                ("gray", "H(t, f)", "1 МГц, аппарат"),
                ("gray", "Рассеяние", "с измерениями")]),
            ("gray", "Физические: Герм–Зернов, HiCIRF, Lv 2022, ИСЗФ", [
                ("gray", "Модель среды", "IRI, флуктуации"),
                ("gray", "Лучи и Рытов", "моды o и x"),
                ("gray", "Реализации", "E(t), S(τ, ν)"),
                ("gray", "Сравнение", "с зондированием")]),
            ("gray", "DAMSON, Уоррингтон", [
                ("purple", "Зонд DAMSON", "S(τ, ν), 3–12 кГц"),
                ("gray", "Разбросы", "τ и доплер"),
                ("gray", "Уоттерсон+", "лоренцев спектр"),
                ("gray", "Модемы", "доступность")]),
            ("gray", "Школа ПГТУ: Иванов, Рябова", [
                ("purple", "ЛЧМ-зонд", "ДЧХ и IRI"),
                ("gray", "Дисперсия", "тренд и шум"),
                ("gray", "ИХ канала", "чирп, потери"),
                ("gray", "Расчёт", "и коррекция")]),
        ],
        "legend": [("teal", "muf.sim"), ("purple", "параметры из измерений"),
                   ("gray", "таблица или модель среды")],
    },
}


def related(lang: str) -> str:
    t = RELATED[lang]
    xs = [40, 198, 356, 514]
    w, h = 126, 56
    body = [f'  <text class="head" x="{x + w / 2}" y="40" text-anchor="middle">'
            f'{escape(head)}</text>' for x, head in zip(xs, t["heads"])]
    for i, (_kind, name, cells) in enumerate(t["lanes"]):
        y0 = 64 + i * 104
        body.append(f'  <text class="lane" x="40" y="{y0 + 16}">{escape(name)}</text>')
        for j, (kind, title, sub) in enumerate(cells):
            body.append(box(kind, xs[j], y0 + 26, w, h, title, sub))
            if j < 3:
                body.append(arrow(f"M{xs[j] + w} {y0 + 54} L{xs[j + 1] - 2} {y0 + 54}"))
    body += legend(64 + 6 * 104 + 8, t["legend"])
    return svg(680, 64 + 6 * 104 + 40, t["title"], body)


def main() -> None:
    for lang in ("en", "ru"):
        suffix = "" if lang == "en" else ".ru"
        (HERE / f"simulator{suffix}.svg").write_text(simulator(lang), encoding="utf-8")
        (HERE / f"related{suffix}.svg").write_text(related(lang), encoding="utf-8")


if __name__ == "__main__":
    main()
