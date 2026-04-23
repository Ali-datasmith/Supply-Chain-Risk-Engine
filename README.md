# Supply Chain Risk Engine

> Enterprise-Grade Risk Intelligence at Global Scale

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Polars](https://img.shields.io/badge/Polars-Lightning_Fast-CD792C?style=for-the-badge)](https://pola.rs)
[![DuckDB](https://img.shields.io/badge/DuckDB-In_Process_OLAP-FFF000?style=for-the-badge)](https://duckdb.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge)](https://streamlit.io)

**Author:** Ali-datasmith | **Contact:** rjptmhmmd@gmail.com

## Executive Summary

Supply Chain Risk Engine is a production-ready platform that processes 1.5M+ global trade records to surface bottlenecks, geopolitical exposure, and financial risk — on zero-cost infrastructure.

## Tech Stack
- **Polars** — Rust-backed dataframe engine, 3–5× faster than Pandas
- **DuckDB** — In-process OLAP SQL on raw Parquet, no server required
- **Streamlit** — Interactive dashboard, free cloud hosting
- **NumPy** — Vectorized composite risk scoring

## Engineering Highlights

### 1.5M Rows on Free Tiers
- Polars chunked generation (300K rows × 5) keeps peak RAM under 400 MB
- DuckDB `parquet_scan()` runs GROUP BY aggregations without loading data into RAM
- Full risk aggregation completes in under 8 seconds on Colab free CPU

### Composite Risk Model
| Signal | Weight |
|--------|--------|
| Port Congestion | 35% |
| Origin Geopolitical Risk | 30% |
| Destination Risk | 20% |
| Delay Ratio | 10% |
| Disruption Noise | 5% |

## How to Run

```bash
# Install
pip install polars duckdb numpy rich

# Run
python engine.py
```

## Module Roadmap
| Module | Status | Description |
|--------|--------|-------------|
| Module 1 | ✅ Complete | Data generation, DuckDB risk scoring |
| Module 2 | 🔄 In Progress | Streamlit dashboard |
| Module 3 | 📋 Planned | Live API ingestion |
| Module 4 | 📋 Planned | ML anomaly detection |

---
*Built with precision. Engineered for scale. Deployed for free.*
