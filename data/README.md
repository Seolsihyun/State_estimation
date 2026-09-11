# 데이터 배치

원본 데이터는 Git에 올리지 않는다. 실험 전에 아래 구조로 배치한다.

```text
data/
├── euroc/
│   └── V1_01_easy/
│       └── mav0/
│           ├── imu0/
│           │   ├── data.csv
│           │   └── sensor.yaml
│           └── state_groundtruth_estimate0/
│               ├── data.csv
│               └── sensor.yaml
└── cf231_leave_one_out/
    └── csv/
        ├── 3/
        │   ├── cf231_imu_raw.csv
        │   └── poses.csv
        ├── 4/   (same files)
        ├── 5/   (same files; held-out test run)
        ├── 9/   (same files)
        └── 10/  (same files)
```

EuRoC `V1_01_easy` 원본은 EuRoC MAV Dataset에서 받는다. CF231은 실험실에서 수집한 데이터로, `cf231_imu_raw.csv`는 Crazyflie IMU, `poses.csv`는 mocap pose이다. loader가 가속도 `g`를 `m/s²`로, 각속도 `deg/s`를 `rad/s`로 변환한다.
