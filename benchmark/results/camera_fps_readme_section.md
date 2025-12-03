#### Camera Frame Rate (FrameDurationLimits)

The following table shows measured performance for different camera frame rates on Raspberry Pi 5:

| Frame Rate | FrameDurationLimits | CPU Mean | CPU P95 | Temp Max | Drop Rate | Recommendation |
|------------|---------------------|----------|---------|----------|-----------|----------------|
| 5fps | `[200000, 200000]` | 32.5% | 34.5% | 49.4°C | 49.6% | ❌ Too High |
| 10fps | `[100000, 100000]` | 68.1% | 70.7% | 54.9°C | 48.8% | ❌ Too High |
| 15fps | `[66667, 66667]` | 95.5% | 100.5% | 58.2°C | 46.9% | ❌ Too High |
| 20fps | `[50000, 50000]` | 78.9% | 79.8% | 55.4°C | 73.6% | ❌ Too High |

**Note:** Camera frame rate is controlled via `FrameDurationLimits` in the launch file (microseconds).
Lower values = higher frame rate. Values are hardcoded for optimal Pi 5 performance.
