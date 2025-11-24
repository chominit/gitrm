# 디버깅 포인트 & 코드 효율화 메모

(요약본)  
- Ray 방향: 모든 Forward/Scanline/Pixelwise 루프에서 월드 z>0 ray를 뒤집어 지면 방향을 보도록 통일.  
- 메모리: `constants.py`의 `MAX_GPU_MEMORY_USAGE_RATIO=0.95`와 확대된 배치 크기(N, M_batch)로 VRAM 95%까지 활용,
  GPU 전용 VRAM 90% 이내에서만 동작하도록 조정.  
- dtype: 주요 Cupy/Numpy 배열을 float64로 통일하여 수치 안정성(큰 절대좌표·미소 GSD) 우선. VRAM 사용량은 증가하므로 배치 크기는 동적 계산에 의존.  
- CPU RAM: `MAX_CPU_MEMORY_BYTES=250GB` 상수를 추가해 향후 대용량 데이터 처리 시 기준값으로 사용 가능.  

알고리즘 구조(정방향/역방향, Forward → Backward 투표)는 사진측량에서 정사영상/변화탐지 파이프라인의 전형적인 틀과
정합성이 있으며, 현재 단계에서 핵심은 **좌표계/방향성 검증과 메모리 안전성 확보**입니다.
