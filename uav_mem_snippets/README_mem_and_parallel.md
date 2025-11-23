# UAV 파이프라인 메모리/병렬화 보완 스니펫

이 zip 에는 다음 세 가지를 포함했습니다.

1. `memory_utils.py`
   - CPU/GPU 메모리 상태를 조회하고
   - 안전한 메모리 사용 한계(예: RAM 250GB, GPU 24GB 의 90%)를 계산하는 유틸리티입니다.
   - CuPy 와 psutil 이 설치되어 있으면 실제 사용량 기준으로 동작합니다.

2. `forward_backward_snippets.py`
   - 기존 `process_rays_gpu` 를 대체할 수 있는 `process_rays_gpu_memsafe` 예시 구현
     (GPU 메모리 사용량을 고려해 point batch 크기를 동적으로 조절).
   - 포인트 클라우드 전체 메모리 사용량을 체크하는 `check_point_cloud_memory` 예시.
   - 사이트 단위 CPU 멀티프로세싱 패턴 예시(`process_all_sites_cpu_parallel`).

   → 이 파일은 **직접 실행하는 용도라기보다는**, 기존
     `part3_complete_pipeline.py` 안의 함수들을 개선할 때
     복사/참고용으로 사용하면 됩니다.

3. 적용 가이드 (요약)
   - `part3_complete_pipeline.py` 안의 `process_rays_gpu` 내용을
     `process_rays_gpu_memsafe` 로 교체하거나, 동일 로직을 반영합니다.
   - `forward_scanline` 시작 부분에서
     `check_point_cloud_memory(coords, colors)` 를 한 번 호출하여
     예상 RAM 사용량을 로그로 확인할 수 있습니다.
   - GPU 메모리가 빡빡한 경우 `min_point_batch`, `max_point_batch` 값을
     조금 더 보수적으로 줄이면 됩니다.
   - CPU 멀티프로세싱은 GPU 작업과 분리해서 (예: 사이트별 전/후처리) 사용하는 것을 권장합니다.

## 알고리즘 관점 코멘트 (정/역방향 플로우)

- 정방향(Forward): 영상 → 레이 → 포인트클라우드 매칭
  - 현재 구현은 중심투영 모델 + OPK + 내부표정(K) 을 이용한
    전통적인 사진측량 정방향 투영과 일관성이 있습니다.
  - Ray 방향 반전 버그를 이미 수정했기 때문에, 기하학적으로 큰 오류는 없습니다.
  - 다만, `ground_Z` 고정 평면을 기준으로 Row 밴드를 잡는 구조라서
    고층 구조물이 많은 지역에서는 Z 마진(±50~100m)을 충분히 주는 것이 안전합니다.

- 역방향(Backward): 포인트 → 영상 투영
  - Forward-vote 가 임계값 이상인 포인트만 선택하여
    다시 영상으로 투영하는 방식은, 정사영상 생성 시
    DSM/PointCloud 를 이용해 각 격자를 영상으로 역투영하는 방식과
    개념적으로 동일합니다.
  - 다만, 현재 구현에는 **가려짐(occlusion) 체크**가 없기 때문에
    일부 포인트가 실제로는 더 앞쪽 구조물에 가려져 있어도
    투표에 포함될 수 있습니다.
    - 이를 개선하려면 영상 단위로 Z-buffer(깊이맵)를 유지하거나,
      동일 픽셀로 투영되는 포인트 중 가장 가까운 것만 사용하는
      추가 필터가 필요합니다.

- 결론적으로, 현재 알고리즘의 큰 틀(정/역방향 투영 구조)은
  사진측량에서 사용하는 정사영상/DSM 생성 플로우와 잘 맞습니다.
  앞으로 개선이 필요하다면 기하학 자체보다는,
  - 메모리/속도 최적화
  - occlusion 처리
  - vote 가중치(시선각, 거리, 시차 등) 개선
  같은 쪽이 더 우선순위라고 보는 편이 적절합니다.
