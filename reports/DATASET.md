# Dataset audit & provenance

Generated 2026-11-25 17:52 UTC by `scripts/audit_dataset.py`.

**Source:** CIC-IDS2017, Canadian Institute for Cybersecurity, University of New Brunswick (unb.ca/cic/datasets/ids-2017.html), MachineLearningCSV files. Full checksums: `data/raw/CHECKSUMS.sha256`.

## Files

| file                                                        |    MB |   rows |   inf cells |   NaN cells |   duplicate rows | labels                                                                                                                         | sha256            |
|:------------------------------------------------------------|------:|-------:|------------:|------------:|-----------------:|:-------------------------------------------------------------------------------------------------------------------------------|:------------------|
| Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv            |  77.1 | 225745 |          64 |           4 |             2633 | DDoS (128,027), BENIGN (97,718)                                                                                                | 6ff1580f5f81c0ae… |
| Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv        |  76.9 | 286467 |         727 |          15 |            72353 | PortScan (158,930), BENIGN (127,537)                                                                                           | ca1824c51bfbb7b3… |
| Friday-WorkingHours-Morning.pcap_ISCX.csv                   |  58.3 | 191033 |         216 |          28 |             6888 | BENIGN (189,067), Bot (1,966)                                                                                                  | 53a41c24d570ea83… |
| Monday-WorkingHours.pcap_ISCX.csv                           | 176.9 | 529918 |         810 |          64 |            26935 | BENIGN (529,918)                                                                                                               | 852c4beb34eda186… |
| Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv |  83.1 | 288602 |         396 |          18 |            35630 | BENIGN (288,566), Infiltration (36)                                                                                            | 6bcda3857c250467… |
| Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv      |  52   | 170366 |         250 |          20 |             6066 | BENIGN (168,186), Web Attack ï¿½ Brute Force (1,507), Web Attack ï¿½ XSS (652), Web Attack ï¿½ Sql Injection (21)              | d67066211fb1689c… |
| Tuesday-WorkingHours.pcap_ISCX.csv                          | 135.1 | 445909 |         327 |         201 |            24065 | BENIGN (432,074), FTP-Patator (7,938), SSH-Patator (5,897)                                                                     | 52b8692ae8c7d2ed… |
| Wednesday-workingHours.pcap_ISCX.csv                        | 225.2 | 692703 |        1586 |        1008 |            81909 | BENIGN (440,031), DoS Hulk (231,073), DoS GoldenEye (10,293), DoS slowloris (5,796), DoS Slowhttptest (5,499), Heartbleed (11) | 893c27dc968bf7a8… |

## Label mapping (raw → FlowSight classes)

| Label      |   flows |
|:-----------|--------:|
| Benign     | 2273097 |
| DoS        |  252661 |
| PortScan   |  158930 |
| DDoS       |  128027 |
| BruteForce |   13835 |
| Other      |    4193 |

## Cleaning

| step | effect |
|---|---|
| raw rows | 2,830,743 |
| identifier columns dropped (leakage) | none |
| classes dropped (`drop_other`) | 4,193 rows |
| rows after cleaning (inf/NaN dropped, duplicates removed) | 2,516,660 |
| removed in total | 314,083 (11.1%) |
| features kept | 70 |

## Final class balance

| Label      |   flows | share   |
|:-----------|--------:|:--------|
| Benign     | 2095057 | 83.248% |
| DoS        |  193745 | 7.698%  |
| DDoS       |  128014 | 5.087%  |
| PortScan   |   90694 | 3.604%  |
| BruteForce |    9150 | 0.364%  |
