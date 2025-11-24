import re
from pathlib import Path

def parse_elapsed(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.search(r"Elapsed_sec:\s*([0-9\.]+)", line)
        if m:
            return float(m.group(1))
    return None

def main():
    gpu = parse_elapsed(Path("benchmark_gpu.log"))
    cpu = parse_elapsed(Path("benchmark_cpu.log"))

    print("GPU elapsed (s):", gpu)
    print("CPU elapsed (s):", cpu)
    if gpu and cpu:
        print(f"CPU/GPU speed ratio (CPU slower x): {cpu / gpu:.2f}")

if __name__ == "__main__":
    main()
