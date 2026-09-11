# 방학 중 상태추정 실험 코드와 결과

이 저장소는 방학 중 직접 수정·실험한 15차원 InEKF, IMU bias 보정, CF231 learned-velocity dead reckoning을 한 곳에 정리한 것이다.

## 코드

- `filters/Hoon_invariant_kalman_filter_15D.py`: 고정 bias, 연속 IMU propagation, position/velocity update 수정
- `filters/Hoon_invariant_kalman_analytic_15D.py`: left-invariant 오차 정의와 analytic Jacobian 수정
- `utils/imu_bias.py`: 구간 데이터에서 IMU bias를 계산하는 코드 추가
- `validation/`: EuRoC·CF231 loader, 실험 스크립트, ghaggin 비교 adapter, 결과 생성 코드 추가
- `validation/results/`: 방학 중 실행한 수치, 궤적, 그림과 학습 checkpoint

## 수정한 문제

1. 기존 sliding-window 실험은 창마다 GT `R, v, p`로 재초기화했다. 이 결과는 짧은 구간 drift이므로 최종 EuRoC 실험은 평가 시작점에서 한 번만 초기화하고 전체 구간을 연속 적분했다.
2. 구간마다 GT bias를 사용하던 `oracle_start`는 최종 방법에서 제외했다. EuRoC propagation 검증은 데이터셋이 제공하는 처음 5초 bias의 평균 하나를 고정했다. CF231은 운동 시작 전 정지 구간을 IMU로 감지하여 한 번 bias를 구한 후 고정했다.
3. InEKF의 오차 정의와 Jacobian이 혼재해 있었다. `ghaggin/invariant-ekf`의 left-invariant convention에 맞춰 보정식, 오차 순서, 위치·속도 Jacobian을 통일했다.
4. IMU-only prediction에서 covariance만 바꾸면 nominal trajectory는 바뀌지 않는다. 따라서 순수 IMU 적분과 learned velocity measurement update를 구분해 비교했다.
5. Run 5 GT가 학습이나 추론에 들어가지 않도록 CF231을 leave-one-run-out으로 재구성했다. Run 3/4/9/10으로 학습하고 Run 5는 초기 상태·초기 IMU 보정·이후 IMU만 사용했다.

## 상태와 식

평균 상태는 `SE_2(3)` 행렬과 IMU bias로 구성한다.

```text
        [ R  v  p ]
X   =   [ 0  1  0 ],       b = [b_g, b_a]
        [ 0  0  1 ]

delta = [dR, dv, dp, dbg, dba] in R^15
```

보정된 IMU 입력은 `omega = omega_m - b_g`, `a = a_m - b_a`이다. `phi = omega * dt`로 두면 nominal propagation은 다음과 같다.

```text
R(k+1) = R(k) Exp(phi)
v(k+1) = v(k) + R(k) Gamma1(phi) a dt + g dt
p(k+1) = p(k) + v(k)dt + R(k) Gamma2(phi) a dt^2 + 0.5 g dt^2
```

오차는 `X_plus = X Exp(delta)`로 보정한다. 이 정의에서 위치와 속도 measurement Jacobian의 활성 block은 각각 `H_p[:, dp] = R`, `H_v[:, dv] = R`이다. 공분산은 `P(k+1) = Phi P Phi^T + Q_d`로 전파하고, measurement가 있을 때만 Kalman correction을 적용한다.

관련 코드:

- 필터 본체: [`filters/Hoon_invariant_kalman_filter_15D.py`](filters/Hoon_invariant_kalman_filter_15D.py)
- analytic Jacobian: [`filters/Hoon_invariant_kalman_analytic_15D.py`](filters/Hoon_invariant_kalman_analytic_15D.py)
- Lie group 연산: [`models/Hoon_lie_group_utils.py`](models/Hoon_lie_group_utils.py)
- 예측·업데이트 보조: [`models/Hoon_invariant_inekf.py`](models/Hoon_invariant_inekf.py)

`filters/` 안의 다른 EKF, UKF, PF는 실험 기반 코드이다. 방학 중 수정과 최종 데이터셋 검증은 위의 두 InEKF 파일을 기준으로 했다.

## 실험 흐름

| 단계 | 실험 | 스크립트 | 결과 |
|---|---|---|---|
| 0 | CF231 Run 5, training-run transferable bias를 쓴 최초 순수 IMU 실험 | [`run_cf231_leave5_dead_reckoning.py`](validation/run_cf231_leave5_dead_reckoning.py) | 후속 bias 개선 실험으로 대체됨 |
| 1 | EuRoC V1_01, 고정 bias 연속 IMU-only | [`run_euroc_continuous_inekf.py`](validation/run_euroc_continuous_inekf.py) | [`euroc_v1_01_continuous`](validation/results/euroc_v1_01_continuous) |
| 2 | CF231 Run 5, 순수 IMU bias/내재오차 보정 | [`run_cf231_pure_imu_improved.py`](validation/run_cf231_pure_imu_improved.py) | [`cf231_run5_pure_imu_improved`](validation/results/cf231_run5_pure_imu_improved) |
| 3 | Python analytic InEKF와 ghaggin C++ 일치 검증 | [`run_cf231_ghaggin_comparison.py`](validation/run_cf231_ghaggin_comparison.py) | [`cf231_run5_ghaggin_comparison`](validation/results/cf231_run5_ghaggin_comparison) |
| 4 | ExtraTrees heading-velocity + InEKF | [`run_cf231_learned_velocity_leave5.py`](validation/run_cf231_learned_velocity_leave5.py) | [`cf231_leave5_learned_velocity`](validation/results/cf231_leave5_learned_velocity) |
| 5 | 손으로 압축한 IMU feature + ExtraTrees | [`run_cf231_shallow_learned_velocity_inekf.py`](validation/run_cf231_shallow_learned_velocity_inekf.py) | [`cf231_leave5_shallow_learned_velocity_inekf`](validation/results/cf231_leave5_shallow_learned_velocity_inekf) |
| 6 | 2초 raw IMU Small-TCN | [`run_cf231_small_tcn_velocity_inekf.py`](validation/run_cf231_small_tcn_velocity_inekf.py) | [`cf231_leave5_small_tcn_velocity_inekf`](validation/results/cf231_leave5_small_tcn_velocity_inekf) |
| 7 | 속도 결합 방식과 AirIO-style uncertainty 단계 비교 | [`run_cf231_imu_only_learned_inekf_stages.py`](validation/run_cf231_imu_only_learned_inekf_stages.py) | [`cf231_imu_only_learned_inekf_stages`](validation/results/cf231_imu_only_learned_inekf_stages) |

[`run_euroc_windowed_dr.py`](validation/run_euroc_windowed_dr.py)는 창 길이별 local drift를 보기 위한 진단용이다.

## 주요 결과

### EuRoC V1_01 easy

기본 설정 `bias_source=euroc_gt`는 데이터셋이 제공하는 처음 5초 gyro/accel bias의 평균을 쓴다. 평가 시작점의 `R, v, p`를 한 번만 사용하고, 이후 138.56초 동안 measurement update나 GT 재초기화는 없다. 따라서 이 결과는 bias가 고정된 조건에서 propagation을 검증한 것이며, online bias estimator의 성능을 주장하는 결과가 아니다.

- 고정 gyro bias [rad/s]: `[-0.002229, 0.020700, 0.076350]`
- 고정 accel bias [m/s²]: `[-0.012493, 0.547693, 0.069086]`
- fixed-bias SO(3) 자세 RMSE: `2.290°`
- zero-bias SO(3) 자세 RMSE: `111.049°`
- 최종 위치 오차: `1933.950 m`

자세에서 bias 보정 효과는 크지만, 외부 측정이 없는 위치는 가속도계 오차가 이중 적분되며 크게 발산했다. 그림: [`03_so3_attitude_error.png`](validation/results/euroc_v1_01_continuous/figures/03_so3_attitude_error.png), [`04_trajectory_diagnostic.png`](validation/results/euroc_v1_01_continuous/figures/04_trajectory_diagnostic.png).

### CF231 Run 5

전체 길이는 181.32초이다. 아래 위치 RMSE는 Run 5 전체 궤적을 mocap GT와 비교한 값이다.

| 방법 | 위치 RMSE [m] | 최종 위치 오차 [m] | SO(3) RMSE [deg] |
|---|---:|---:|---:|
| 기존 training-run bias 순수 IMU | 2316.583 | 5340.969 | 4.615 |
| Run 5 정지 구간 bias + gyro intrinsic 순수 IMU | 522.201 | 1005.430 | 3.554 |
| ExtraTrees learned velocity + InEKF | 4.392 | 7.747 | 4.026 |
| Shallow ExtraTrees, 고정 InEKF 자세 | 4.244 | 7.506 | 3.554 |
| Small-TCN loose velocity integration | **2.395** | **2.840** | 3.554 |
| Small-TCN velocity InEKF update | 2.615 | 3.247 | **1.800** |
| AirIO-style velocity + uncertainty InEKF | 32.889 | 44.052 | 4.362 |
| AirIO-style + displacement auxiliary loss | 16.955 | 17.552 | 3.393 |

순수 IMU 보정은 기존과 비교해 위치 RMSE를 77.46%, 최종 위치 오차를 81.18% 줄였지만 절대 오차는 여전히 크다. Small-TCN이 위치 수치는 가장 낮았다. 다만 2.395 m 결과는 예측 속도를 InEKF 밖에서 적분한 `loose` 결합이고, 실제 InEKF velocity update로 결합한 결과는 2.615 m이다. AirIO에서 착안한 body-velocity/uncertainty 모델은 이 데이터셋에서 성능이 좋아지지 않았다.

Python analytic InEKF와 ghaggin C++ 참조 구현의 최대 차이는 자세 `1.40e-13 deg`, 속도 `5.69e-12 m/s`, 위치 `5.88e-10 m`로 설정한 허용치를 통과했다.

최종 단계 그림: [`01_learned_trajectory_stages.png`](validation/results/cf231_imu_only_learned_inekf_stages/figures/01_learned_trajectory_stages.png), [`02_all_stage_errors.png`](validation/results/cf231_imu_only_learned_inekf_stages/figures/02_all_stage_errors.png). 순수 IMU와 learned velocity 비교 그림은 [`build_cf231_pure_vs_learned_figures.py`](validation/build_cf231_pure_vs_learned_figures.py)로 다시 생성할 수 있다.

## 재현 방법

Python 3.10 환경에서 실험했다.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

데이터 구조는 [`data/README.md`](data/README.md)를 따른다. 스크립트는 저장소 최상위에서 실행한다.

```bash
# EuRoC 연속 IMU-only
python3 -m validation.run_euroc_continuous_inekf

# CF231 순수 IMU 보정
python3 -m validation.run_cf231_pure_imu_improved

# 순수 IMU와 learned velocity 비교 그림 재생성
python3 -m validation.build_cf231_pure_vs_learned_figures

# CF231 ExtraTrees -> shallow -> Small-TCN -> 최종 단계 비교
python3 -m validation.run_cf231_learned_velocity_leave5
python3 -m validation.run_cf231_shallow_learned_velocity_inekf
python3 -m validation.run_cf231_small_tcn_velocity_inekf
python3 -m validation.run_cf231_imu_only_learned_inekf_stages
```

`최종 단계 비교` 스크립트에 `--reuse-trained`를 넘기면 포함된 `.pt` checkpoint를 사용해 필터·지표·그림만 다시 생성한다.


## 폴더 구조와 결과 파일

```text
filters/                 15차원 EKF/InEKF/UKF/PF 구현
models/                  Lie group과 InEKF 상태 연산
utils/                   수치 연산, sigma point, resampling, bias 보조
validation/              데이터 loader와 실험 스크립트
validation/results/      실제 실행 결과
data/README.md           외부 데이터 배치 방법
```

각 결과 폴더의 `summary.json`은 실험 설정, bias, 평가 수치를 갖는다. `trajectories.npz`는 시간별 GT와 추정 궤적이다. `figures/`는 해당 수치로 그린 비교 그림이다. `.pt`는 용량이 작은 최종 TCN checkpoint만 남겼다.

## 해석 시 주의할 점

- CF231 학습 방법은 Run 5를 보지 않았지만 평가 데이터셋이 하나다. 다른 장소·비행·센서에 대한 일반화를 주장할 수 없다.
- 학습 속도는 IMU만을 입력으로 쓰지만 학습 label은 training run의 mocap GT에서 만든다.
- CF231 Run 5의 초기 `R, v, p`는 GT를 사용했다. 정지 구간 gyro bias는 IMU 평균이고, accel bias는 정지 구간 가속도계 평균과 초기 자세에서 계산한 예상 중력의 차이다. 초기화 후 Run 5 GT는 평가에만 쓴다.
- EuRoC fixed bias는 데이터셋의 GT bias에서 구했으므로 실제 센서의 online calibration 결과로 해석하지 않는다.
- RPY는 pitch가 ±90°에 가까울 때 roll/yaw가 민감하다. 따라서 자세는 RPY와 SO(3) geodesic error를 함께 본다.
- 저장소에 없는 것: 원본 CSV/EuRoC 데이터, Python cache, 중간 빌드 산출물 등
