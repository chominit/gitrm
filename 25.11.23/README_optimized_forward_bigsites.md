# UAV Footprint Forward Pipeline - Optimized Forward for Big Sites

이 폴더는 `uav_pipeline_footprint_forward_memray2_fixed` 기반으로,
다음과 같은 Forward 최적화를 추가한 버전입니다.

1. **Ray 방향 보정 (하늘 → 지면)**  
   - `scanline_engine.compute_row_ground_band()` 및 `process_image_chunked()`에서
     ray_world[2] > 0 인 경우 ray 방향을 뒤집어 항상 지면(음의 z)을 보도록 보정했습니다.
   - 기존에 발생하던 `Hits: 0/…` 문제(광선이 하늘을 향하는 경우)를 구조적으로 제거합니다.

2. **GPU 메모리 사용 상한 (VRAM 90% 이내)**  
   - `memory_utils.py`에서 CuPy의 `Device.mem_info`를 사용하여
     현재 사용 중인 VRAM과 총 VRAM을 읽고,
     `GPU_MEM_USAGE_RATIO = 0.90` (24GB 카드 기준 약 21.6GB) 안에서만
     Ray × Point 배치가 생성되도록 `get_dynamic_point_batch()`를 구현했습니다.
   - `scanline_engine.process_rays_chunk_gpu()`는 이 함수를 사용해
     point batch 크기를 동적으로 줄이므로, shared memory 스와핑에 빠질 위험을 줄입니다.

3. **Scanline + Row Chunk 기반 Forward (part3_complete_pipeline_optimized.py)**  
   - `part3_complete_pipeline_optimized.py`는 기존 `part3_complete_pipeline.py`를 보존한 채,
     Forward 전용 최적화 엔트리포인트를 제공합니다.
   - 주요 흐름:
     1. `load_point_cloud()`로 coords, colors 로드
     2. camera DB, K, tolerance, ground_Z 계산
     3. 카메라 XY 범위 ± margin_xy 로 footprint 포인트(`fp_coords`, `fp_indices`)만 필터링
     4. 각 이미지를 `scanline_engine.process_image_chunked()`로 처리
        - detection mask를 row chunk(기본 32줄) 단위로 나누고,
        - 각 chunk에 대해 ground band(XY, Z) 계산 후
        - footprint 내 band에 속하는 포인트 subset만 GPU로 전송하여 ray casting
     5. vote 결과를 `forward_votes_opt.npy`로 저장하고, threshold(7) 이상 포인트만 LAS로 저장

4. **CPU/RAM 상한 참고 상수**  
   - `memory_utils.get_safe_cpu_bytes()`를 통해
     RAM 250GB 상한을 기준으로 “현재 안전하게 쓸 수 있는 메모리”를 얻을 수 있습니다.
   - 아직 자동 분할(chopping)까지는 구현하지 않았지만,
     Site B(7억), Site C(20억)에서 필요할 경우 이 값을 참조해 chunk 크기를 조정할 수 있습니다.

## 사용 방법

1. 기존 코드와 동일한 폴더 구조 유지:
   - `constants.py`, `las_utils.py`, `camera_io.py` 등은 그대로 사용합니다.
   - 새로 추가된 파일:
     - `memory_utils.py`
     - `scanline_engine.py`
     - `part3_complete_pipeline_optimized.py`

2. Forward 실행 (예시):

```bash
(yolov11) C:\...\uav_pipeline_footprint_forward_memray2> python part3_complete_pipeline_optimized.py Zenmuse_AI_Site_A
```

3. 출력:
   - `PART3_DIR / <site_name> / forward_votes_opt.npy`
   - `PART3_DIR / <site_name> / vote_opt_7.las`

## 주의 사항 및 향후 확장 포인트

- 이 버전은 coords 전체를 CPU/RAM에 올리되,
  GPU에는 *footprint 내 band 포인트만* 부분적으로 전송하는 구조입니다.
- Site A(1.2억점) 수준에서는 이 구조만으로도 충분히 동작할 수 있고,
  Site B(7억점) 이상에서는 **추가로 XY 타일링 / 공간 인덱싱**을 도입하면
  band 필터링 비용을 더 줄일 수 있습니다.
- 타일링/인덱싱은 별도의 모듈로 확장하는 것이 좋으며,
  이때도 `scanline_engine.process_image_chunked()` 인터페이스는 그대로 재사용할 수 있습니다
  (band 내 포인트 subset을 제공하는 부분만 교체).
