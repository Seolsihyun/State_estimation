# InEKF 수정 및 검증 기준

## 기존 실험에서 해석상 잘못된 점

1. Sliding-window 실험은 매 구간의 `R, v, p`를 GT로 다시
   초기화했다. 각 구간의 local drift를 보는 데에는 유효하지만 하나의
   연속 dead-reckoning trajectory 성능으로 제시하면 안 된다.
2. `oracle_start`는 구간마다 GT bias를 다시 사용하므로 최종 방법이
   아니다. 최종 실험에서는 calibration 구간에서 얻은 bias 하나만
   고정한다.
3. IMU-only에서는 measurement update가 없기 때문에 covariance와
   Kalman gain이 nominal trajectory를 수정하지 않는다. 이 조건에서
   필터 간 trajectory 차이는 InEKF 자체의 장점이 아니라 서로 다른
   propagation/초기화/입력 처리의 차이다.
4. 위치를 차분해 만든 pseudo-velocity를 다시 measurement로 넣지
   않는다. 현재 InEKF에는 이 동작이 없으며, 최종 validation에도
   velocity update가 없다.

## 코드에서 수정한 이론적 불일치

교수님이 지정한 `ghaggin/invariant-ekf`의 left-invariant error
convention을 사용하도록 다음을 통일했다.

- Group error/correction: `X_plus = X @ Exp(delta)`
- State covariance order: `[dR, dv, dp, dbg, dba]`
- Position Jacobian: `[0, 0, R, 0, 0]` (world position residual 형태)
- Velocity Jacobian: `[0, R, 0, 0, 0]`
- IMU 입력과 bias에 의존하는 body-frame process transition
- SO(3) gamma 함수를 이용한 rotation/velocity/position propagation
- Bias는 Euclidean state로 유지

Reference:

- <https://github.com/ghaggin/invariant-ekf>
- <https://github.com/ghaggin/invariant-ekf/blob/master/inekf/src/IEKF.cpp>

## 최종 IMU-only 실험

```bash
python3 -m validation.run_euroc_continuous_inekf
```

- 처음 5초: bias calibration
- 평가 시작점: GT `R, v, p`로 한 번만 초기화
- 이후 전체 구간: IMU propagation만 수행
- bias: 하나의 상수값으로 고정
- 사용하지 않는 것: 위치 update, velocity update, 자세 update, 구간별
  GT 재초기화, position-derived pseudo-velocity
- 주요 지표: roll/pitch/yaw 오차와 SO(3) geodesic attitude error
- trajectory: 누적 drift를 보여주는 보조 진단

Euler roll/pitch/yaw는 pitch가 ±90도에 가까울수록 roll/yaw가 민감하게
변한다. 따라서 RPY를 보고하되, 좌표 특이점에 영향을 받지 않는 SO(3)
geodesic error도 반드시 함께 보고한다.

## GPS를 추가할 때

실제 GNSS position만 `measurement_update()`에 넣는다. 두 position
measurement의 차분으로 만든 velocity를 별도의 measurement로 중복해서
넣지 않는다. GPS-fused trajectory를 평가할 때 ATE/RPE를 사용하는 것이
적절하다.
