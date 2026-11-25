# Destination Port ablation

Model xgb, same 160,000-row training sample, scored on a 40,000-row validation split. Destination Port was selected by feature selection when available.

| | macro-F1 |
|---|---|
| with Destination Port | 0.9964 |
| without Destination Port | 0.9938 |

|            |   recall with port |   precision with port |   recall without port |   precision without port |   recall drop |
|:-----------|-------------------:|----------------------:|----------------------:|-------------------------:|--------------:|
| Benign     |             0.9993 |                0.9999 |                0.9989 |                   0.9996 |        0.0003 |
| BruteForce |             0.9931 |                0.9931 |                0.9793 |                   0.9861 |        0.0138 |
| DDoS       |             0.9985 |                0.9995 |                0.9975 |                   1      |        0.001  |
| DoS        |             0.9994 |                0.9968 |                0.9981 |                   0.9932 |        0.0013 |
| PortScan   |             0.9972 |                0.9876 |                0.9972 |                   0.9883 |        0      |

Largest recall drop: **BruteForce** (+0.0138). A small drop means the class is recognised from traffic behaviour, not just the port; a large drop means an attacker using a non-standard port could evade detection.