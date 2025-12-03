#### Camera Exposure Time Optimization

The `ExposureTime` parameter controls sensor exposure in microseconds. Tested at 10 FPS (100000μs frame duration):

| Exposure Time | ExposureTime | CPU Mean | Det Hz | Obj Count | Avg Conf | Notes |
|---------------|--------------|----------|--------|-----------|----------|-------|
| 10 ms | `10000` | 67.7% | 4.85 Hz | 1.0 | 0.69 | Darker image |
| 15 ms | `15000` | 66.2% | 4.94 Hz | 1.0 | 0.59 |  |
| 20 ms | `20000` | 68.4% | 4.75 Hz | 1.7 | 0.67 | **Default** |
| 25 ms | `25000` | 69.1% | 4.98 Hz | 1.8 | 0.63 | Brighter image |
| 30 ms | `30000` | 69.2% | 4.96 Hz | 1.0 | 0.68 | Brighter image |

**Key Finding:** Exposure time has minimal impact on CPU usage or detection rate, but affects image brightness and object detection count.
