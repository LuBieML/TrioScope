import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Object Dictionary Indices
SETUP_INDEX = 0x368C
START_STOP_INDEX = 0x368B
STATUS_INDEX = 0x3680
DATA_INDEX = 0x3687

# Common Variable Addresses
SPD_FB_RPM_ADDR = 0x0F10
SPD_CMD_RPM_ADDR = 0x0F11
CURRENT_POS_L1 = 0x0F16
CURRENT_POS_H1 = 0x0F17
CURRENT_POS_L2 = 0x0F18
CURRENT_POS_H2 = 0x0F19

class MockConnection:
    """Mock Trio controller connection to demonstrate the CoE sequence."""
    def __init__(self):
        self.memory = {}
    
    def Ethercat_CoWriteAxis_Value(self, axis, index, subindex, obj_type, value):
        logger.info(f"CoWrite [Axis {axis}] 0x{index:04X}:{subindex} = {value}")
        self.memory[(index, subindex)] = value
        
    def Ethercat_CoReadAxis(self, axis, index, subindex, obj_type, vr_num):
        # Mock status logic: automatically set status to 'done' (2) after a few reads
        if index == STATUS_INDEX:
            if not hasattr(self, '_read_count'):
                self._read_count = 0
            self._read_count += 1
            
            # bits 14-15 define status
            if self._read_count < 3:
                status_bits = 1 # sampling in progress
            else:
                status_bits = 2 # done
                
            val = (status_bits << 14)
            logger.info(f"CoRead  [Axis {axis}] 0x{index:04X}:{subindex} -> Status: {status_bits} (Raw: 0x{val:04X})")
            self.SetVrValue(vr_num, val)
        else:
            val = self.memory.get((index, subindex), 0)
            logger.info(f"CoRead  [Axis {axis}] 0x{index:04X}:{subindex} -> {val}")
            self.SetVrValue(vr_num, val)
            
    def SetVrValue(self, vr, val):
        self.memory[f"VR_{vr}"] = val
        
    def GetVrValue(self, vr):
        return self.memory.get(f"VR_{vr}", -9999)


def execute_drive_scope_sequence(conn, axis=0):
    logger.info("--- Step 1: Configure Capture Setup ---")
    
    # 0: Number of entity (not always used, but good practice if available)
    
    # 1: Trigger Mode -> 0: No trigger (starts immediately)
    conn.Ethercat_CoWriteAxis_Value(axis, SETUP_INDEX, 1, 6, 0)
    
    # 6: Channel1 Variable Data Type -> 1: Int16
    conn.Ethercat_CoWriteAxis_Value(axis, SETUP_INDEX, 6, 6, 1)
    
    # 7: Sample Time -> 8 (8 * 125us = 1ms)
    conn.Ethercat_CoWriteAxis_Value(axis, SETUP_INDEX, 7, 6, 8)
    
    # 8-15: Sample Channel Addresses
    # Let's map Speed FB, Speed CMD, and a 64-bit Position to channels 1-6
    channels = [
        SPD_FB_RPM_ADDR,
        SPD_CMD_RPM_ADDR,
        CURRENT_POS_L1,
        CURRENT_POS_H1,
        CURRENT_POS_L2,
        CURRENT_POS_H2,
        0x0000,
        0x0000
    ]
    for i, addr in enumerate(channels):
        conn.Ethercat_CoWriteAxis_Value(axis, SETUP_INDEX, 8 + i, 6, addr)
        
    logger.info("--- Step 2: Start Data Capture ---")
    # Write 1 to start
    conn.Ethercat_CoWriteAxis_Value(axis, START_STOP_INDEX, 0, 6, 1)
    
    logger.info("--- Step 3: Poll Capture Status ---")
    vr_scratch = 901
    status = 0
    while status != 2: # 2 = Sampling is done
        time.sleep(0.5)
        # Clear VR
        conn.SetVrValue(vr_scratch, -9999)
        # Read status
        conn.Ethercat_CoReadAxis(axis, STATUS_INDEX, 0, 6, vr_scratch)
        raw_val = conn.GetVrValue(vr_scratch)
        if raw_val != -9999:
            # Extract bits 14-15
            status = (int(raw_val) >> 14) & 0x3
            
    logger.info("--- Step 4: Read and Parse Capture Data ---")
    logger.info("Status indicates completion (2). Data is ready to be read from 0x3687.")
    logger.info("Total Size: 16000 bytes. This is typically read via block transfer (EC_COE_FIFO).")
    
if __name__ == "__main__":
    mock_conn = MockConnection()
    execute_drive_scope_sequence(mock_conn)
