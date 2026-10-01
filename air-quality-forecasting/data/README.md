# Static source data and provenance

- Dataset: UCI Beijing PM2.5, dataset 381.
- Source page: https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data
- Download URL: https://archive.ics.uci.edu/static/public/381/beijing+pm2+5+data.zip
- Citation: Chen, S. (2015). Beijing PM2.5 [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5JS49.
- License: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/
- Download date: October 1, 2026 (UTC).
- Archive member / checked-in file: `raw/PRSA_data_2010.1.1-2014.12.31.csv`.
- Original CSV byte size: **2,010,494**.
- SHA-256: `4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c`.
- The CSV was extracted unchanged from the source archive. Pollution comes from the US Embassy in Beijing; weather comes from Beijing Capital International Airport.

## Validation actually performed

There are 43,824 raw rows and 43,824 distinct hourly timestamps from 2010-01-01 00:00 to 2014-12-31 23:00, interpreted as Asia/Shanghai. No sorting was necessary, no timestamp rows were absent, and no duplicate or invalid timestamps were found. There are 2,067 missing PM2.5 readings and a longest missing PM2.5 run of 155 hours. No source weather values or other required columns are missing. No unusual wind categories were found. Numeric/date/domain checks passed.

The per-column counts and partition exclusion reports are saved in `artifacts/data_quality.json`. `config.json` pins the source hash, byte size, row count, and range. Pipeline execution is entirely local; it never redownloads the data. Verify independently on macOS with:

```bash
shasum -a 256 data/raw/PRSA_data_2010.1.1-2014.12.31.csv
python -m air_quality validate --config config.json
```

`Iws`, `Is`, and `Ir` are accumulated variables. This project uses the recorded current values and makes no assumption that they are standalone next-hour wind or precipitation. Their reset conventions require further investigation.

Synthetic tests create small temporary sources with independent byte checksums and date ranges. They are deliberately labeled synthetic, are never substituted for this real CSV, and never establish real model performance.
