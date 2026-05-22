import Trio_UnifiedApi as TUA
import threading
import time
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s")
logger = logging.getLogger("test")

def event_handler(et, ival, sval):
    logger.info(f"Trio event: {et} {ival} {sval}")

ip = "192.168.0.245"
logger.info(f"Connecting to {ip}...")
try:
    conn = TUA.TrioConnectionTCP(event_handler, ip)
    conn.OpenConnection()
except Exception as e:
    logger.error(f"Connection failed: {e}")
    sys.exit(1)

conn_lock = threading.Lock()
watchdog_stop = threading.Event()
watchdog_thread = None
trio_connected = True

def watchdog_loop():
    global trio_connected
    logger.info("Watchdog loop started")
    while not watchdog_stop.wait(0.5):
        if not trio_connected:
            continue
        try:
            heartbeat_done = threading.Event()
            heartbeat_error = []

            def _heartbeat():
                try:
                    with conn_lock:
                        conn.GetVrValue(0)
                except Exception as e:
                    heartbeat_error.append(e)
                finally:
                    heartbeat_done.set()

            t = threading.Thread(target=_heartbeat, name="WatchdogHeartbeat", daemon=True)
            t.start()
            if not heartbeat_done.wait(timeout=5.0):
                logger.warning("Watchdog heartbeat timed out")
                trio_connected = False
                break
            if heartbeat_error:
                raise heartbeat_error[0]
        except Exception as exc:
            logger.warning(f"Watchdog detected connection loss: {exc}")
            trio_connected = False
            break
    logger.info("Watchdog loop exited")

def _call(fn, *args, default=None):
    try:
        with conn_lock:
            return fn(*args)
    except Exception as exc:
        logger.warning(f"Call failed inside _call({fn.__name__}, {args}): {type(exc).__name__}: {exc}")
        return default
    finally:
        time.sleep(0.01)

def scan_network_sim():
    logger.info("Starting scan_network simulation...")
    # Simulate a full scan with lots of calls
    for slot_idx in range(1):
        vr_scratch = 901
        _call(conn.SetVrValue, vr_scratch, -999.0)
        state_ok = _call(conn.Ethercat_GetState_VR, slot_idx, vr_scratch)
        state_val = _call(conn.GetVrValue, vr_scratch, default=-999.0)
        logger.info(f"Slot {slot_idx} state value: {state_val}")

        n = _call(conn.Ethercat_CheckNumberOfSlaves, slot_idx, default=0)
        logger.info(f"Slot {slot_idx} slaves: {n}")
        
        # Enumerate each slave (simulate 9 slaves like the user)
        for pos in range(9):
            _call(conn.Ethercat_CheckSlaveOnline, slot_idx, pos, default=False)
            _call(conn.Ethercat_GetSlaveAddress, slot_idx, pos, default=0)
            _call(conn.Ethercat_GetSlaveAxis, slot_idx, pos, default=-1)
            # simulate axis parameters
            _call(conn.GetAxisParameter_DRIVE_TYPE, pos, default=0)
            _call(conn.GetAxisParameter_DRIVE_STATUS, pos, default=0)
            _call(conn.GetAxisParameter_SLOT_NUMBER, pos, default=0)

    # Fallback axis mapping (0..31)
    logger.info("Starting fallback mapping simulation...")
    for ax in range(32):
        _call(conn.GetAxisParameter_DRIVE_TYPE, ax)

# Start watchdog
watchdog_thread = threading.Thread(target=watchdog_loop, name="WatchdogThread", daemon=True)
watchdog_thread.start()

# Let it run for 2 seconds
time.sleep(2.0)

# Simulate what ethercat_map_window does: stop watchdog, scan, start watchdog
logger.info("Stopping watchdog...")
watchdog_stop.set()
watchdog_thread.join(timeout=1.0)
# Reassign stop event like scope_app does
watchdog_stop = threading.Event()

# Run the scan
scan_network_sim()

# Restart watchdog
logger.info("Restarting watchdog...")
watchdog_thread = threading.Thread(target=watchdog_loop, name="WatchdogThread", daemon=True)
watchdog_thread.start()

# Let it run for 3 seconds
time.sleep(3.0)

# Cleanup
watchdog_stop.set()
watchdog_thread.join(timeout=1.0)
conn.CloseConnection()
logger.info("Done.")
