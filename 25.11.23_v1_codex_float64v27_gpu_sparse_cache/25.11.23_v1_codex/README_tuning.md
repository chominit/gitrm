# 성능 튜닝 가이드 (CPU/GPU 동시 사용률 80% 목표)

작업 디렉터리: `C:\Users\jscool\uav_pipeline_codes\25.11.23_v1\25.11.23_v1_codex_float64v21_gpu`

## 핵심 환경변수
아래 변수만 조정하면 CPU/GPU 점유율과 VRAM 사용을 쉽게 맞출 수 있습니다.

| 변수 | 추천 시작값 | 역할 |
| --- | --- | --- |
| `CPU_THREAD_FRACTION` | `0.8` | OMP/MKL/numexpr 스레드 비율 (0.8 → CPU 80% 내외) |
| `IMAGE_WORKERS` | `12` (32코어 기준) | 이미지 단위 병렬 스레드 수. 너무 높이면 메모리/오버헤드↑ |
| `IMAGE_GPU_SLOTS` | `2` | GPU로 동시에 올릴 커널/업로드 수. VRAM 여유 없으면 1 |
| `ROW_STRIDE` | `8` | 행 샘플링 간격. 8이면 속도↑, 정밀도↓. 정밀 필요시 4~1 |
| `FP_MARGIN_M` | `0` | 카메라 XY 풋프린트 여유(m). 0이면 실제 범위만 사용 |
| `GROUND_BAND` | `0.10` | 1m 격자 평균 Z의 ±밴드. 지면 제거 기준(>밴드만 남김) |

## 실행 예시 (균형 세팅)
```
set CPU_THREAD_FRACTION=0.8
set IMAGE_WORKERS=12
set IMAGE_GPU_SLOTS=2
set ROW_STRIDE=8
set FP_MARGIN_M=0
set GROUND_BAND=0.10
python part3_complete_pipeline.py
```

## 느릴 때/VRAM 부족할 때
- `ROW_STRIDE` 값을 12→16으로 키워 행 수를 더 줄이기.
- `IMAGE_WORKERS` 8~10으로 낮추기 (CPU 오버헤드 감소).
- `IMAGE_GPU_SLOTS`를 1로 낮추기 (동시 커널/업로드 감소).
- `constants.py`의 `MAX_POINTS_PER_BATCH`(현재 2,000,000)를 더 줄여 VRAM 스필 방지.

## CPU만 빠르게 시험하고 싶을 때
```
set USE_GPU=0
set IMAGE_WORKERS=8
set ROW_STRIDE=8
python part3_complete_pipeline.py
```

## GPU만 최대로 돌리고 싶을 때 (VRAM 여유 필수)
```
set CPU_THREAD_FRACTION=1.0
set IMAGE_WORKERS=4
set IMAGE_GPU_SLOTS=2
set ROW_STRIDE=4
python part3_complete_pipeline.py
```

## 로그/벤치마크
- `benchmark_gpu.bat` / `benchmark_cpu.bat` : GPU/CPU 실행 시간 비교 로그 생성.
- `compare_benchmark.py` : 두 로그의 `Elapsed_sec` 비교 출력.

## 주의
- 지면 제거: 1m 격자 평균 Z ± `GROUND_BAND` 안은 제거하고 그 밖(전경)만 남김.
- 행 tqdm은 기본 OFF(`TQDM_ROWS=0`). 켜려면 `set TQDM_ROWS=1`.
- 워커를 과도하게 올리면 CPU 메모리와 I/O 오버헤드로 오히려 느려질 수 있습니다.
