# 디버깅 포인트 & 코드 효율화 메모

(요약본)  
- Ray 방향: 모든 Forward/Scanline/Pixelwise 루프에서 월드 z>0 ray를 뒤집어 지면 방향을 보도록 통일.  
- 메모리: `constants.py`의 `MAX_GPU_MEMORY_USAGE_RATIO=0.9`에 맞춰 배치 크기(N, M_batch)를 동적으로 줄여,
  GPU 전용 VRAM 90% 이내에서만 동작하도록 조정.  
- dtype: 주요 Cupy 배열을 float32로 통일하여 VRAM 사용량을 절반 수준으로 감소.  
- CPU RAM: `MAX_CPU_MEMORY_BYTES=250GB` 상수를 추가해 향후 대용량 데이터 처리 시 기준값으로 사용 가능.  

알고리즘 구조(정방향/역방향, Forward → Backward 투표)는 사진측량에서 정사영상/변화탐지 파이프라인의 전형적인 틀과
정합성이 있으며, 현재 단계에서 핵심은 **좌표계/방향성 검증과 메모리 안전성 확보**입니다.
