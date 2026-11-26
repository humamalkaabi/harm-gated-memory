# Data

`python main.py setup` downloads every dataset into `data/raw/`:

| File | Dataset | Source |
|---|---|---|
| `ETTh1.csv`, `ETTh2.csv`, `ETTm1.csv`, `ETTm2.csv` | ETT | https://github.com/zhouhaoyi/ETDataset |
| `electricity.txt.gz` | ECL (UCI ElectricityLoadDiagrams20112014) | https://github.com/laiguokun/multivariate-time-series-data |
| `traffic.txt.gz` | Traffic (Caltrans PeMS) | https://github.com/laiguokun/multivariate-time-series-data |
| `weather.csv` | Weather (Max Planck Institute for Biogeochemistry, Jena, 2020), as distributed with Autoformer | included |

TCPD is cloned into `third_party/TCPD/` from https://github.com/alan-turing-institute/TCPD.

Every series is split in time order into 20% training, 5% validation and 75% online test, with the ETT series
restricted to their first 20 months, and standardised with training statistics (`hgm/data.py`). `WeatherClean`
is Weather with its 81 recording errors (-9999) treated as missing; it is used only in a post hoc check.
