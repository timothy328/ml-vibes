---
name: data-quality-check
description: Check California housing data before model training.
---

# Data quality check

Verify the eight numeric features, expected order, finite values, missingness,
target availability, both binary classes, duplicate records, and reasonable
row-count/statistic changes. Check for target leakage. Keep the example data
in memory and do not write CSV files. Stop and report any failed check.

