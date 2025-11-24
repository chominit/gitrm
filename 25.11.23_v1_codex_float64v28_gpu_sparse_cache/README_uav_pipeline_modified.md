# UAV Footprint Forward Pipeline (GPU) - 메모리 제한 + Ray 방향 보정 버전

이 버전에는 다음 변경 사항이 모두 반영되어 있습니다.

1. Ray 방향 보정 (정방향/역방향 모두)
2. GPU 메모리 사용 상한 (VRAM의 95%까지) + 배치 확장
3. Cupy/Numpy dtype 통일 (float32 → float64, 정밀도 우선)
4. Forward에서 이미지 100장 제한 제거
5. CPU RAM 상한 상수 256GB, CPU 스레드 자동 90% 사용
6. 기본 실행 시 Site B/C 계열만 처리 (Zenmuse/P4R/Joint B,C)

자세한 디버깅 포인트와 효율화 아이디어는 `DEV_NOTES_debug_and_optimization.md`를 참고하세요.
