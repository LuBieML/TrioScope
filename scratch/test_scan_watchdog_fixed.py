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
    stop_event = watchdog_stop
    while not stop_event.wait(0.5):
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
                if stop_event.is_set():
                    break
                logger.warning("Watchdog heartbeat timed out")
                trio_connected = False
                break
            if stop_event.is_set():
                break
            if heartbeat_error:
                raise heartbeat_error[0]
        except Exception as exc:
            if stop_event.is_set():
                break
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
    # Simulate a full scan with slot 0
    vr_scratch = 901
    _call(conn.SetVrValue, vr_scratch, -999.0)
    state_ok = _call(conn.Ethercat_GetState_VR, 0, vr_scratch)
    state_val = _call(conn.GetVrValue, vr_scratch, default=-999.0)
    logger.info(f"Slot 0 state value: {state_val}")

    n = _call(conn.Ethercat_CheckNumberOfSlaves, 0, default=0)
    logger.info(f"Slot 0 slaves: {n}")
    
    # Enumerate each slave. (On our controller there are 0 slaves, so we shouldn't query out of range!)
    for pos in range(int(n)):
        _call(conn.Ethercat_CheckSlaveOnline, 0, pos, default=False)
        _call(conn.Ethercat_GetSlaveAddress, 0, pos, default=0)
        _call(conn.Ethercat_GetSlaveAxis, 0, pos, default=-1)

    # Fallback axis mapping (only runs if there are online slaves and all are axis < 0)
    # Since there are 0 online slaves, we do not run fallback mapping!
    logger.info("Skipping fallback mapping simulation since there are 0 online slaves.")

# Start watchdog
watchdog_thread = threading.Thread(target=watchdog_loop, name="WatchdogThread", daemon=True)
watchdog_thread.start()

# Let it run for 2 seconds
time.sleep(2.0)

# Stop watchdog
logger.info("Stopping watchdog...")
watchdog_stop.set()
watchdog_thread.join(timeout=1.0)
# DO NOT reassign watchdog_stop

# Run the scan
scan_network_sim()

# Restart watchdog
logger.info("Restarting watchdog...")
watchdog_stop.clear() # clear the existing event
watchdog_thread = threading.Thread(target=watchdog_loop, name="WatchdogThread", daemon=True)
watchdog_thread.start()

# Let it run for 3 seconds
time.sleep(3.0)

# Cleanup
watchdog_stop.set()
watchdog_thread.join(timeout=1.0)
conn.CloseConnection()
logger.info("Done.")
