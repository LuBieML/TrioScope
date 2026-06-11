"""Tuning thresholds shared by the signal-metrics analysis modules."""

EPS_VEL_FRAC = 0.02   # 2% of peak demand velocity → "moving"
EPS_ACC_FRAC = 0.10   # 10% of peak demand accel → "constant velocity"
SETTLE_MS = 200       # settle window after each move end
NOISE_FLOOR_SIGMA = 5.0  # FFT peak threshold above median noise floor (raised for noise rejection)
SATURATION_FRAC = 0.95   # |current| > 95% of observed peak = near-saturation
MIN_OSCILLATION_HZ = 5.0 # position loop bandwidth floor
MIN_CRUISE_DURATION_S = 0.3   # need at least 300 ms of cruise for FFT
MIN_CYCLES_FOR_PEAK = 3       # peak must fit ≥3 cycles in analyzed window
MIN_COHERENCE = 0.7           # cross-phase coherence threshold (proxy used instead)
