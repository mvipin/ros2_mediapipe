#### Frame Skip Optimization

The `frame_skip` parameter controls how many camera frames are skipped between processing cycles. Higher values reduce CPU load but decrease detection responsiveness.

| frame_skip | Processing | CPU Mean | CPU P95 | Det Hz | Obj Count | Avg Conf | Notes |
|------------|------------|----------|---------|--------|-----------|----------|-------|
| 0 | All frames | 70.5% | 72.4% | 4.98 Hz | 1.0 | 0.55 | Max detection rate |
| 1 | Every 2 frame | 66.7% | 69.9% | 4.73 Hz | 1.0 | 0.56 | **Default** |
| 2 | Every 3 frame | 45.9% | 48.0% | 3.29 Hz | 1.0 | 0.56 |  |
| 3 | Every 4 frame | 35.9% | 36.8% | 2.47 Hz | 1.0 | 0.57 | Max CPU savings |

**Key Finding:** Frame skip trades detection rate for CPU savings while maintaining detection quality.
