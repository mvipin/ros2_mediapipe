#### Confidence Threshold Optimization

The `confidence_threshold` parameter filters detections post-inference. Tested at 10 FPS, 20ms exposure, frame_skip=0:

| Threshold | CPU Mean | Det Hz | Obj Count | Avg Conf | Det Rate | Notes |
|-----------|----------|--------|-----------|----------|----------|-------|
| 0.3 | 64.8% | 5.00 Hz | 4.0 | 0.48 | 100% | More false positives |
| 0.4 | 62.8% | 5.00 Hz | 3.2 | 0.50 | 100% |  |
| 0.5 | 63.5% | 5.00 Hz | 1.6 | 0.55 | 100% | **Default** |
| 0.6 | 62.5% | 3.06 Hz | 1.0 | 0.61 | 100% |  |
| 0.7 | 63.2% | N/A | N/A | N/A | N/A | May miss valid detections |

**Key Finding:** Confidence threshold filters post-inference, so CPU impact is minimal. Trade-off is between detection sensitivity (lower threshold) and precision (higher threshold).
