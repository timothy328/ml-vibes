---
name: data-quality-check
description: Validate refreshed California housing data before preprocessing or retraining.
---

# Data quality check

Verify the eight documented numeric feature columns and their order, finite
values, missingness, target availability, both derived classes, duplicate
records, and changes in row count and summary statistics. Check for target
leakage and malformed records.

Keep raw and derived data in approved storage. This project’s example loader
keeps data in memory and must not create a CSV. Stop retraining and report the
failing rule when a check fails.

