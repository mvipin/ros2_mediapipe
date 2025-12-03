# Confidence Threshold Optimization - Experiment Results

Generated: 2025-12-04 19:31:18

## Summary Table

| Threshold | Samples | CPU Mean (%) | Det Hz | Obj Count | Avg Conf | Det Rate (%) |
|-----------|---------|--------------|--------|-----------|----------|--------------|
| 0.3 | 12.0 | 64.8 | 5.00 | 4.00 | 0.478 | 100.0 |
| 0.4 | 12.0 | 62.8 | 5.00 | 3.20 | 0.499 | 100.0 |
| 0.5 | 12.0 | 63.5 | 5.00 | 1.65 | 0.550 | 100.0 |
| 0.6 | 12.0 | 62.5 | 3.06 | 1.00 | 0.612 | 100.0 |
| 0.7 | 12.0 | 63.2 | N/A | N/A | N/A | N/A |

## Key Observations

- **Object Count**: Expected inverse relationship with threshold (lower threshold = more detections)
- **Average Confidence**: Expected direct relationship with threshold (higher threshold = higher avg confidence)
- **CPU Usage**: Expected minimal impact (threshold applied post-inference)
- **Detection Rate**: Percentage of frames containing at least one detection
