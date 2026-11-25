"""Column names that differ between CICFlowMeter versions (newer/2018 name -> 2017 name).

Used by the cross-dataset report and the live-capture bridge. Extend as needed.
"""

RENAMES = {
    "Dst Port": "Destination Port",
    "Tot Fwd Pkts": "Total Fwd Packets",
    "Tot Bwd Pkts": "Total Backward Packets",
    "TotLen Fwd Pkts": "Total Length of Fwd Packets",
    "Fwd Pkt Len Mean": "Fwd Packet Length Mean",
    "Bwd Pkt Len Mean": "Bwd Packet Length Mean",
    "Flow Byts/s": "Flow Bytes/s",
    "Flow Pkts/s": "Flow Packets/s",
    "SYN Flag Cnt": "SYN Flag Count",
    "RST Flag Cnt": "RST Flag Count",
    "ACK Flag Cnt": "ACK Flag Count",
    "PSH Flag Cnt": "PSH Flag Count",
    "Pkt Size Avg": "Average Packet Size",
    "Init Fwd Win Byts": "Init_Win_bytes_forward",
}
