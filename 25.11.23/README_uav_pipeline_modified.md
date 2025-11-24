# UAV Footprint Forward Pipeline (GPU) - 메모리 제한 + Ray 방향 보정 버전

이 버전에는 다음 변경 사항이 모두 반영되어 있습니다.

1. Ray 방향 보정 (정방향/역방향 모두)
2. GPU 메모리 사용 상한 (VRAM의 90%까지)
3. Cupy dtype 통일 (float64 → float32)
4. Forward에서 이미지 100장 제한 제거
5. CPU RAM 상한 상수 추가 (250GB)

자세한 디버깅 포인트와 효율화 아이디어는 `DEV_NOTES_debug_and_optimization.md`를 참고하세요.
