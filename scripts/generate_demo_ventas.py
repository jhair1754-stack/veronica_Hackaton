"""Generate the original, synthetic promotional-action demo dataset."""

from __future__ import annotations

import csv
import random
from pathlib import Path


SEED = 20261012
ROWS = 320
OUTPUT = Path(__file__).resolve().parents[1] / "demo_ventas_sinteticas.csv"
CHANNEL_EFFECT = {"email": 2.0, "redes": 5.0, "tienda": 8.0}


def generate() -> None:
    rng = random.Random(SEED)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("horas_promocion", "experiencia_equipo", "presupuesto_estimado", "canal", "ventas"))
        for _ in range(ROWS):
            hours = rng.randint(3, 18)
            experience = rng.randint(0, 12)
            budget = rng.randrange(80, 401, 10)
            if rng.random() < 0.025:
                budget += rng.randrange(650, 1001, 10)
            channel = rng.choice(tuple(CHANNEL_EFFECT))
            noise = rng.gauss(0.0, 4.0)
            sales = max(0, round(15 + 1.9 * hours + 1.4 * experience
                                 + 0.065 * budget + CHANNEL_EFFECT[channel] + noise))
            writer.writerow((hours, "" if rng.random() < 0.06 else experience,
                             "" if rng.random() < 0.04 else budget, channel, sales))


if __name__ == "__main__":
    generate()
    print(f"Generado: {OUTPUT} ({ROWS} filas; semilla {SEED})")
