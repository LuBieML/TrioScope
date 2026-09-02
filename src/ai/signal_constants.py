"""Shared tuning constants for the scope signal-analysis engine."""

EPS_VEL_FRAC = 0.02   # 2% of peak demand velocity → "moving"
EPS_ACC_FRAC = 0.10   # 10% of peak demand accel → accel/decel vs cruise
SETTLE_MS = 200       # settle window length after each move end
NOISE_FLOOR_SIGMA = 5.0  # FFT peak threshold above median noise floor
SATURATION_FRAC = 0.95   # |current| > 95% of observed peak = near-saturation
MIN_OSCILLATION_HZ = 5.0  # position loop bandwidth floor
MIN_CRUISE_DURATION_S = 0.3  # need >=300 ms of contiguous cruise for FFT
SPECTRAL_GUARD_S = 0.020  # trim demand-transition edges from cruise FFT runs
MIN_CYCLES_FOR_PEAK = 3      # peak must fit >=3 cycles in analyzed window
REVERSAL_HALF_WIDTH_S = 0.080  # ± window around demand-velocity zero crossings
SETTLE_BAND_SIGMA = 4.0   # auto tolerance band = 4x robust noise sigma
AUTO_SETTLE_SIGNAL_FRAC = 0.01  # auto band floor = 1% of typical move FE
FE_OSCILLATION_SIGNAL_FRAC = 0.10  # spectral line must be >=10% of move FE
VELOCITY_OSCILLATION_DEMAND_FRAC = 0.005  # or >=0.5% of peak demand speed
RINGING_CROSSINGS_MAX = 3  # more hysteresis crossings than this = ringing
MAX_PER_MOVE_REPORTED = 8  # cap per-move detail lists in reports
MIN_FFT_SAMPLES = 64       # shortest cruise run / Welch window worth analyzing
WELCH_MAX_NPERSEG = 1024   # Welch window cap (1.024 s at 1 kHz servo rate)
MIN_COHERENCE = 0.7        # magnitude-squared coherence gate for phase claims
