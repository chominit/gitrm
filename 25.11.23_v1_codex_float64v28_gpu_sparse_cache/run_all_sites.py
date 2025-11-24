# -*- coding: utf-8 -*-
"""
run_all_sites.py
Footprint-based Forward 실행 스크립트
"""

import argparse
from part3_forward_pixelwise import run

ALL_SITES = [
    "Zenmuse_AI_Site_A",
    "Zenmuse_AI_Site_B",
    "Zenmuse_AI_Site_C",
    "P4R_Site_A_Solid",
    "P4R_Site_B_Solid_Merge_V2",
    "P4R_Site_C_Solid_Merge_V2",
]

def main():
    parser = argparse.ArgumentParser(description="Footprint-based Forward 처리")
    parser.add_argument("--sites", nargs="+", default=["Zenmuse_AI_Site_B"])
    parser.add_argument("--k-max", type=int, default=1)
    parser.add_argument("--sample", type=int, default=0)

    args = parser.parse_args()

    sites = ALL_SITES if "all" in args.sites else args.sites

    print(f"{'='*70}")
    print(f"Footprint-based Forward Pipeline")
    print(f"{'='*70}")
    print(f"사이트 수: {len(sites)}개")
    print(f"K_max: {args.k_max}")
    print(f"샘플링: {'전체' if args.sample == 0 else f'{args.sample:,} points'}")
    print(f"{'='*70}\n")

    print(f"처리할 사이트:")
    for i, site in enumerate(sites, 1):
        print(f"   {i}. {site}")
    print(f"\n{'='*70}\n")

    for idx, site_name in enumerate(sites, 1):
        print(f"\n[{idx}/{len(sites)}] {site_name}")
        print(f"-" * 50)

        try:
            run(
                site_name=site_name,
                k_max=args.k_max,
                use_sampling=(args.sample > 0),
                sample_size=args.sample
            )
        except Exception as e:
            print(f"[ERROR] {site_name} 처리 실패: {e}")
            continue

    print(f"\n{'='*70}")
    print(f"전체 처리 완료!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
