# UAV 파이프라인 메모리/멀티프로세싱 보조 코드

이 폴더는 기존 `uav_pipeline_footprint_forward` 코드에
다음 기능을 추가하기 위한 **보조 모듈/예시 코드**를 포함합니다.

1. **CPU RAM 가드 (최대 250GB)**  
   - `mem_limits.py` 의 `MAX_CPU_RAM_GB = 250.0` 로 설정.  
   - 큰 배열을 만들기 전에 `can_allocate_cpu_bytes(예상바이트, label)` 로 체크.  
   - psutil 이 설치되어 있으면 실제 메모리 사용량을 기준으로 경고를 출력하고,  
     한도를 넘는 경우 `False`를 반환하여 분기 처리 가능.

2. **GPU VRAM 가드 (전용 메모리 90% 이내)**  
   - `GPU_MEM_USAGE_RATIO = 0.90`  
   - `get_dynamic_point_batch(n_rays, base_max_points)` 를 사용해서,  
     `process_rays_gpu()` 의 **point batch 크기**를 동적으로 조정.  
   - CuPy 의 `memGetInfo()` 로 현재 free/total VRAM 을 읽어서,
     90% 한도를 넘지 않도록 배치 크기를 줄이는 방식.

3. **CPU 멀티프로세싱 예시**  
   - `multiproc_example.py` 의 `process_all_sites_multiproc()` 참고.  
   - `constants.py` 에서 `CPU_WORKERS`, `EST_RAM_PER_SITE_GB` 등을 설정한 뒤,  
     `part3_complete_pipeline.process_all_sites()` 대신 이 함수를 호출하면 됨.  
   - GPU 1장 환경에서는 워커를 1~2 개 정도로 유지하는 것을 권장.

4. **실제 코드에 붙이는 위치**  
   - `part3_complete_pipeline.py` 상단 import 아래에 다음을 추가:

     ```python
     from mem_limits import (
         MAX_CPU_RAM_GB,
         GPU_MEM_USAGE_RATIO,
         can_allocate_cpu_bytes,
         get_dynamic_point_batch,
     )
     ```

   - `load_point_cloud()` 내부에서 coords/colors 를 로드한 뒤:

     ```python
     total_bytes = coords.nbytes + colors.nbytes
     if not can_allocate_cpu_bytes(total_bytes, label="load_point_cloud"):
         raise MemoryError(f"포인트 클라우드 메모리 사용량 {total_bytes/1024**3:.1f}GB 가 설정 한도 {MAX_CPU_RAM_GB}GB 를 초과합니다.")
     ```

   - `process_rays_gpu()` 내부의 point batch 루프는
     `forward_gpu_memory_guard_example.py` 를 참고해서
     `get_dynamic_point_batch()` 로 교체.

이 ZIP 을 해제한 뒤, 필요한 부분만 기존 파이프라인 코드에 복사/반영해서 사용하면 됩니다.
