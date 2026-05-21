import time
def poll_sensor(target_time, N):
    start = time.time()
    # ANOMALY: No sleep, pinning the CPU core to 100%
    while time.time() - start < (N / 100000):
        pass