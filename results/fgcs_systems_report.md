# FGCS-Oriented Systems Evidence Report

## Scope Alignment

- Distributed and collaborative computing: two-cloud, three-share, and SecureNN-style multi-party deployments.
- Cloud/edge/IoT infrastructures: projected end-to-end latency under edge LAN, cloudlet, regional cloud, cross-region cloud, and constrained IoT gateway links.
- High-performance and scalable computing: batch-throughput scaling from 256 to 2048 samples.
- Security and privacy in future systems: data/model split exposure, collusion threshold, and communication-round evidence.
- Data-intensive applications: official ISIC 2018 secure-CNN workload as a main image benchmark branch.

## Deployment Projection Summary

- Deployment rows: `192`.
- ASS / 2Cloud projected latency mean: `1.1212`.
- 3Share / ASS projected latency mean: `1.5089`.
- SecureNN / ASS projected latency mean: `13.8379`.
- Boundary cases where SecureNN projects below ASS: `12/48` rows, tasks=`Fashion, MNIST`.

| Scenario | Task | ASS ms | ASS/2Cloud | 3Share/ASS | SecureNN/ASS | ASS network share |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| edge_lan | Digits | 7.510 | 1.326 | 2.019 | 11.191 | 99.93 |
| edge_lan | Liver | 1.291 | 1.072 | 1.255 | 15.514 | 97.70 |
| edge_lan | MNIST | 486.841 | 1.209 | 2.091 | 0.807 | 100.00 |
| edge_lan | Medical | 1.950 | 1.148 | 1.554 | 51.299 | 99.10 |
| campus_cloudlet | Digits | 11.510 | 1.191 | 1.665 | 8.692 | 99.95 |
| campus_cloudlet | Liver | 5.291 | 1.017 | 1.062 | 6.809 | 99.44 |
| campus_cloudlet | MNIST | 490.841 | 1.207 | 2.082 | 0.833 | 100.00 |
| campus_cloudlet | Medical | 5.950 | 1.044 | 1.181 | 19.501 | 99.70 |
| metropolitan_edge | Digits | 52.525 | 1.213 | 1.728 | 9.139 | 99.99 |
| metropolitan_edge | Liver | 21.335 | 1.020 | 1.074 | 7.501 | 99.86 |
| metropolitan_edge | MNIST | 2449.201 | 1.208 | 2.084 | 0.825 | 100.00 |
| metropolitan_edge | Medical | 24.679 | 1.053 | 1.217 | 22.692 | 99.93 |
| regional_cloud | Digits | 115.045 | 1.191 | 1.665 | 8.693 | 100.00 |
| regional_cloud | Liver | 52.641 | 1.016 | 1.060 | 6.839 | 99.94 |
| regional_cloud | MNIST | 4908.400 | 1.207 | 2.082 | 0.831 | 100.00 |
| regional_cloud | Medical | 59.341 | 1.044 | 1.181 | 19.548 | 99.97 |
| cross_region_cloud | Digits | 230.084 | 1.191 | 1.665 | 8.693 | 100.00 |
| cross_region_cloud | Liver | 105.252 | 1.016 | 1.059 | 6.841 | 99.97 |
| cross_region_cloud | MNIST | 9816.798 | 1.207 | 2.082 | 0.831 | 100.00 |
| cross_region_cloud | Medical | 118.665 | 1.044 | 1.181 | 19.551 | 99.99 |
| iot_gateway | Digits | 355.201 | 1.351 | 2.076 | 11.599 | 100.00 |
| iot_gateway | Liver | 43.085 | 1.108 | 1.362 | 21.354 | 99.93 |
| iot_gateway | MNIST | 24321.994 | 1.209 | 2.092 | 0.803 | 100.00 |
| iot_gateway | Medical | 76.636 | 1.195 | 1.699 | 64.200 | 99.98 |

## Batch Throughput Scaling

| Method | Min batch | Max batch | Best batch | Best throughput | Best/min |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 2Cloud-D (Data-only) | 256 | 2048 | 2048 | 829536.0 | 6.949 |
| 3Share-DM (3-party) | 256 | 2048 | 2048 | 615292.2 | 7.595 |
| ASS (Ours) | 256 | 2048 | 2048 | 810078.5 | 7.119 |
| SecureNN (3PC) | 256 | 2048 | 256 | 14073.7 | 1.000 |

## Official ISIC 2018 Secure-CNN Mainline Workload

- Official subset size: `512` (`positive=256`, `negative=256`).
- ASS secure CNN: acc=`0.5312`, time=`4.0888 ms/sample`, comm=`3.8772 MB/sample` (`248.1421 MB total`).

## Recommended FGCS Framing

- Present ASS as a future cloud/edge privacy-preserving inference system, not only as a cryptographic primitive.
- Lead with model-split security and deployment-projected latency/communication under multi-cloud and edge scenarios.
- Treat the official ISIC secure-CNN workload as the main image branch, with explicit scope that it is a systems workload rather than a clinical diagnostic benchmark.
- State the MNIST/Fashion projection boundary explicitly: ASS buys model non-exposure at extra communication cost on high-dimensional inputs.
- Keep synthetic-data fallback and CKKS payload guards in the reproducibility checklist.