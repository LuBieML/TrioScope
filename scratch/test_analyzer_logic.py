import struct
from typing import List, Tuple, Dict, Union
import numpy as np

# Total size defined by the COMBO document
TOTAL_BYTES = 16000
TOTAL_WORDS = 8000
NUM_CHANNELS = 8
SAMPLES_PER_CHANNEL = 1000

def parse_interleaved_data(raw_bytes: bytes) -> np.ndarray:
    """
    Parses the 1D array of 8000 words (16000 bytes) by grouping them into 
    chunks of 8. 
    
    Memory Layout:
    Word 0: Ch1[0], Word 1: Ch2[0] ... Word 7: Ch8[0]
    Word 8: Ch1[1], Word 9: Ch2[1] ... Word 15: Ch8[1]
    
    Returns a numpy array of shape (SAMPLES_PER_CHANNEL, NUM_CHANNELS)
    containing 16-bit unsigned integers.
    """
    if len(raw_bytes) < TOTAL_BYTES:
        raise ValueError(f"Expected {TOTAL_BYTES} bytes, got {len(raw_bytes)}")
        
    # Read as 16-bit little-endian words
    raw_words = np.frombuffer(raw_bytes[:TOTAL_BYTES], dtype=np.dtype('<u2'))
    
    # Reshape into (1000 samples, 8 channels)
    # This automatically groups them by 8 (interleaved format)
    interleaved_data = raw_words.reshape((SAMPLES_PER_CHANNEL, NUM_CHANNELS))
    return interleaved_data


def reconstruct_32bit_variable(low_channel_data: np.ndarray, high_channel_data: np.ndarray, signed: bool = True) -> np.ndarray:
    """
    Reconstructs a 32-bit variable from two 16-bit channels.
    Example: 0x0F2A (EST_SPD_ADDR low) and 0x0F2B (high).
    """
    # Combine low and high 16-bit words into a 32-bit integer
    combined = (high_channel_data.astype(np.uint32) << 16) | low_channel_data.astype(np.uint32)
    if signed:
        return combined.astype(np.int32)
    return combined


def reconstruct_64bit_variable(ch_l1: np.ndarray, ch_h1: np.ndarray, ch_l2: np.ndarray, ch_h2: np.ndarray, signed: bool = True) -> np.ndarray:
    """
    Reconstructs a 64-bit variable from four 16-bit channels.
    Example: CURRENT_POS_ADDR (0x0F16 - 0x0F19)
    ch_l1: Low 16 bits
    ch_h1: Mid-Low 16 bits
    ch_l2: Mid-High 16 bits
    ch_h2: High 16 bits
    """
    combined = (
        (ch_h2.astype(np.uint64) << 48) |
        (ch_l2.astype(np.uint64) << 32) |
        (ch_h1.astype(np.uint64) << 16) |
        ch_l1.astype(np.uint64)
    )
    if signed:
        return combined.astype(np.int64)
    return combined


def test_parsing_logic():
    print("Testing data parsing logic...")
    
    # Generate 16000 bytes of dummy data
    # Let's put specific values to test the interleaving
    dummy_data = np.zeros((SAMPLES_PER_CHANNEL, NUM_CHANNELS), dtype=np.uint16)
    
    for sample in range(SAMPLES_PER_CHANNEL):
        for ch in range(NUM_CHANNELS):
            dummy_data[sample, ch] = (ch + 1) * 1000 + sample
            
    raw_bytes = dummy_data.tobytes()
    
    # Test Step 4 Parsing
    parsed = parse_interleaved_data(raw_bytes)
    
    assert parsed.shape == (1000, 8), "Shape mismatch"
    assert parsed[0, 0] == 1000, "Ch1 Sample 0 mismatch"
    assert parsed[0, 1] == 2000, "Ch2 Sample 0 mismatch"
    assert parsed[1, 0] == 1001, "Ch1 Sample 1 mismatch"
    
    print("-> Basic interleaving parse test passed!")
    
    # Test 32-bit reconstruction
    # Simulate low word (0xFFFF) and high word (0x0001) -> 0x0001FFFF = 131071
    dummy_low = np.full(1000, 0xFFFF, dtype=np.uint16)
    dummy_high = np.full(1000, 0x0001, dtype=np.uint16)
    
    res_32 = reconstruct_32bit_variable(dummy_low, dummy_high, signed=True)
    assert res_32[0] == 131071, f"32-bit reconstruction failed: {res_32[0]}"
    print("-> 32-bit reconstruction test passed!")
    
    # Test 64-bit reconstruction
    dummy_l1 = np.full(1000, 0xDDDD, dtype=np.uint16)
    dummy_h1 = np.full(1000, 0xCCCC, dtype=np.uint16)
    dummy_l2 = np.full(1000, 0xBBBB, dtype=np.uint16)
    dummy_h2 = np.full(1000, 0xAAAA, dtype=np.uint16)
    
    res_64 = reconstruct_64bit_variable(dummy_l1, dummy_h1, dummy_l2, dummy_h2, signed=False)
    expected_64 = 0xAAAABBBBCCCCDDDD
    assert res_64[0] == expected_64, f"64-bit reconstruction failed: {res_64[0]:X}"
    print("-> 64-bit reconstruction test passed!")


if __name__ == "__main__":
    test_parsing_logic()
