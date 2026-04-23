"""
Supply Chain Risk Engine · Module 1 — Data Generation & Risk Aggregation
Author  : Ali-datasmith
Contact : rjptmhmmd@gmail.com
Version : 1.0.0
License : MIT
"""

from __future__ import annotations

import time
import random
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Final

import duckdb
import numpy as np
import polars as pl

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("SCRE")

# ── Constants ─────────────────────────────────────────────────────────────────
TOTAL_ROWS: Final[int] = 1_500_000
CHUNK_SIZE: Final[int] = 300_000
PARQUET_PATH: Final[Path] = Path("supply_chain_raw.parquet")
RISK_REPORT_PATH: Final[Path] = Path("risk_report.parquet")
RANDOM_SEED: Final[int] = 42

# ── Reference Data ────────────────────────────────────────────────────────────
ORIGIN_COUNTRIES: Final[list[str]] = [
    "China", "India", "USA", "Germany", "Vietnam",
    "Mexico", "South Korea", "Taiwan", "Bangladesh", "Brazil",
]
DEST_COUNTRIES: Final[list[str]] = [
    "USA", "Germany", "UK", "France", "Japan",
    "Canada", "Australia", "Netherlands", "UAE", "Singapore",
]
PRODUCT_CATEGORIES: Final[list[str]] = [
    "Electronics", "Pharmaceuticals", "Automotive", "Textiles",
    "Chemicals", "Food & Beverage", "Machinery", "Medical Devices",
    "Raw Materials", "Consumer Goods",
]
CARRIERS: Final[list[str]] = [
    "Maersk", "MSC", "CMA CGM", "Evergreen", "COSCO",
    "Hapag-Lloyd", "ONE", "Yang Ming", "ZIM", "PIL",
]
TRANSPORT_MODES: Final[list[str]] = ["Sea", "Air", "Rail", "Road"]
TRANSPORT_WEIGHTS: Final[list[float]] = [0.55, 0.20, 0.15, 0.10]

PORT_CONGESTION: Final[dict[str, float]] = {
    "China": 0.72, "USA": 0.61, "Germany": 0.38, "India": 0.55,
    "Vietnam": 0.49, "Taiwan": 0.44, "South Korea": 0.35,
    "Mexico": 0.58, "Bangladesh": 0.63, "Brazil": 0.51,
}

GEOPOLITICAL_RISK: Final[dict[str, float]] = {
    "China": 0.68, "Russia": 0.91, "Iran": 0.95, "India": 0.42,
    "Vietnam": 0.31, "Germany": 0.12, "USA": 0.15, "Taiwan": 0.76,
    "Mexico": 0.48, "Bangladesh": 0.55, "Brazil": 0.38,
    "UK": 0.18, "France": 0.14, "Japan": 0.21, "Canada": 0.10,
    "Australia": 0.12, "Netherlands": 0.13, "UAE": 0.37, "Singapore": 0.16,
    "South Korea": 0.28,
}


@dataclass(slots=True, frozen=True)
class EngineConfig:
    total_rows: int = TOTAL_ROWS
    chunk_size: int = CHUNK_SIZE
    parquet_path: Path = PARQUET_PATH
    risk_report_path: Path = RISK_REPORT_PATH
    seed: int = RANDOM_SEED
    risk_threshold: float = 0.65


def _shipment_id(chunk_idx: int, local_idx: int) -> str:
    raw = f"{chunk_idx:04d}-{local_idx:07d}-{random.random():.6f}"
    return "SHP-" + hashlib.sha1(raw.encode()).hexdigest()[:10].upper()


def _date_range(n: int, rng: np.random.Generator) -> list[str]:
    start = date(2020, 1, 1)
    end = date(2024, 12, 31)
    delta_days = (end - start).days
    offsets = rng.integers(0, delta_days, size=n)
    return [(start + timedelta(days=int(d))).isoformat() for d in offsets]


class ShipmentGenerator:
    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)

    def _make_chunk(self, chunk_idx: int, size: int) -> pl.DataFrame:
        rng = self._rng
        origins = rng.choice(ORIGIN_COUNTRIES, size=size)
        destinations = rng.choice(DEST_COUNTRIES, size=size)
        categories = rng.choice(PRODUCT_CATEGORIES, size=size)
        carriers = rng.choice(CARRIERS, size=size)
        modes = rng.choice(TRANSPORT_MODES, size=size, p=TRANSPORT_WEIGHTS)
        weight_kg = rng.lognormal(mean=6.5, sigma=1.2, size=size).clip(1, 50_000)
        value_usd = rng.lognormal(mean=10.0, sigma=1.5, size=size).clip(100, 5_000_000)
        lead_days = rng.integers(1, 90, size=size)
        delay_days = np.where(
            rng.random(size) < 0.25,
            rng.integers(1, 30, size=size), 0,
        ).astype(np.int32)
        congestion = np.array([PORT_CONGESTION.get(o, 0.40) for o in origins], dtype=np.float32)
        geo_risk_origin = np.array([GEOPOLITICAL_RISK.get(o, 0.30) for o in origins], dtype=np.float32)
        geo_risk_dest = np.array([GEOPOLITICAL_RISK.get(d, 0.30) for d in destinations], dtype=np.float32)
        disruption_noise = rng.uniform(0, 0.15, size=size).astype(np.float32)
        risk_score = (
            0.35 * congestion + 0.30 * geo_risk_origin + 0.20 * geo_risk_dest
            + 0.10 * (delay_days / 30).clip(0, 1).astype(np.float32)
            + 0.05 * disruption_noise
        ).clip(0.0, 1.0)
        dates = _date_range(size, rng)
        ids = [_shipment_id(chunk_idx, i) for i in range(size)]
        return pl.DataFrame({
            "shipment_id": ids, "shipment_date": dates,
            "origin_country": origins.tolist(), "dest_country": destinations.tolist(),
            "product_category": categories.tolist(), "carrier": carriers.tolist(),
            "transport_mode": modes.tolist(),
            "weight_kg": weight_kg.astype(np.float32),
            "value_usd": value_usd.astype(np.float64),
            "lead_days": lead_days.astype(np.int32), "delay_days": delay_days,
            "congestion_score": congestion, "geo_risk_origin": geo_risk_origin,
            "geo_risk_dest": geo_risk_dest, "composite_risk": risk_score.astype(np.float32),
        })

    def generate(self) -> Path:
        cfg = self.cfg
        chunks = cfg.total_rows // cfg.chunk_size
        remainder = cfg.total_rows % cfg.chunk_size
        log.info(f"Generating {cfg.total_rows:,} rows across {chunks} chunks …")
        t0 = time.perf_counter()
        frames = []
        for i in range(chunks):
            frames.append(self._make_chunk(i, cfg.chunk_size))
            log.info(f"  chunk {i+1}/{chunks} ✓  ({(i+1)*cfg.chunk_size:>10,} rows)")
        if remainder:
            frames.append(self._make_chunk(chunks, remainder))
        df = pl.concat(frames, rechunk=True)
        df.write_parquet(cfg.parquet_path, compression="snappy")
        elapsed = time.perf_counter() - t0
        size_mb = cfg.parquet_path.stat().st_size / 1_048_576
        log.info(f"Dataset written → {cfg.parquet_path}  ({size_mb:.1f} MB, {elapsed:.1f}s)")
        return cfg.parquet_path


class RiskAggregator:
    _SQL_RISK = """
        WITH base AS (
            SELECT origin_country, product_category, transport_mode,
                COUNT(*) AS shipment_count,
                ROUND(AVG(composite_risk), 4) AS avg_risk_score,
                ROUND(MAX(composite_risk), 4) AS max_risk_score,
                ROUND(AVG(delay_days), 2) AS avg_delay_days,
                ROUND(SUM(value_usd)/1e6, 2) AS total_value_musd,
                ROUND(AVG(congestion_score), 4) AS avg_congestion,
                ROUND(AVG(geo_risk_origin), 4) AS avg_geo_risk,
                ROUND(SUM(CASE WHEN composite_risk >= {threshold} THEN 1 ELSE 0 END)*100.0/COUNT(*),2) AS pct_high_risk
            FROM parquet_scan('{path}')
            GROUP BY origin_country, product_category, transport_mode
        )
        SELECT *, CASE
            WHEN avg_risk_score >= 0.75 THEN 'CRITICAL'
            WHEN avg_risk_score >= {threshold} THEN 'HIGH'
            WHEN avg_risk_score >= 0.45 THEN 'MEDIUM'
            ELSE 'LOW' END AS risk_tier,
            RANK() OVER (ORDER BY avg_risk_score DESC) AS global_risk_rank
        FROM base ORDER BY avg_risk_score DESC
    """
    _SQL_BOTTLENECK = """
        SELECT origin_country, ROUND(AVG(congestion_score),4) AS avg_congestion,
            ROUND(AVG(composite_risk),4) AS avg_risk, COUNT(*) AS shipment_count,
            ROUND(SUM(value_usd)/1e6,2) AS exposed_value_musd
        FROM parquet_scan('{path}')
        GROUP BY origin_country HAVING avg_congestion >= 0.50
        ORDER BY avg_congestion DESC
    """
    _SQL_CATEGORY = """
        SELECT product_category, ROUND(AVG(composite_risk),4) AS avg_risk_score,
            ROUND(SUM(value_usd)/1e9,3) AS total_value_busd, COUNT(*) AS shipments,
            ROUND(AVG(delay_days),2) AS avg_delay
        FROM parquet_scan('{path}')
        GROUP BY product_category ORDER BY avg_risk_score DESC
    """

    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        self._conn = duckdb.connect(":memory:")
        self._conn.execute("SET memory_limit='1.5GB'")
        self._conn.execute("SET threads=2")

    def run(self) -> dict[str, pl.DataFrame]:
        path = str(self.cfg.parquet_path)
        t = self.cfg.risk_threshold
        log.info("Running DuckDB risk aggregation …")
        t0 = time.perf_counter()
        results = {
            "risk_matrix":       self._conn.execute(self._SQL_RISK.format(path=path, threshold=t)).pl(),
            "port_bottlenecks":  self._conn.execute(self._SQL_BOTTLENECK.format(path=path)).pl(),
            "category_exposure": self._conn.execute(self._SQL_CATEGORY.format(path=path)).pl(),
        }
        log.info(f"Aggregation complete in {time.perf_counter()-t0:.1f}s")
        for name, df in results.items():
            log.info(f"  {name:<22} → {len(df):,} rows")
        return results

    def save(self, results: dict[str, pl.DataFrame]) -> None:
        for name, df in results.items():
            out = Path(f"{name}.parquet")
            df.write_parquet(out, compression="snappy")
            log.info(f"Saved {name} → {out}")


class SupplyChainRiskEngine:
    def __init__(self, cfg: EngineConfig | None = None) -> None:
        self.cfg = cfg or EngineConfig()

    def run(self) -> None:
        log.info("═" * 60)
        log.info("  Supply Chain Risk Engine  ·  Module 1")
        log.info("═" * 60)
        gen = ShipmentGenerator(self.cfg)
        gen.generate()
        agg = RiskAggregator(self.cfg)
        results = agg.run()
        agg.save(results)
        for name, df in results.items():
            print(f"\n── {name} (top 5) ──")
            print(df.head(5))
        log.info("Module 1 complete.")


if __name__ == "__main__":
    SupplyChainRiskEngine().run()
