# Combined Model Comparison

Models are ranked by emission MAE when emission metrics are available.

| model | target | mae | rmse | r2 | train_rows | test_rows |
| --- | --- | --- | --- | --- | --- | --- |
| rf | emission_nm | 23.8493 | 37.8891 | 0.8375 | 33105 | 8166 |
| extratrees | emission_nm | 28.2001 | 48.2149 | 0.7369 | 33105 | 8166 |
| histgb | emission_nm | 29.3118 | 40.4835 | 0.8145 | 33105 | 8166 |
| gbdt | emission_nm | 40.1038 | 51.8589 | 0.6956 | 33105 | 8166 |
| extratrees | quantum_yield | 0.1464 | 0.2203 | 0.5052 | 24824 | 6482 |
| rf | quantum_yield | 0.1505 | 0.2113 | 0.5448 | 24824 | 6482 |
| histgb | quantum_yield | 0.1749 | 0.2254 | 0.4818 | 24824 | 6482 |
| gbdt | quantum_yield | 0.2087 | 0.2540 | 0.3420 | 24824 | 6482 |
