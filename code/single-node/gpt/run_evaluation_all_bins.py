import argparse
import os
import subprocess
import sys


DEFAULT_FILE_NAME = "test-set-gpt-singleagent.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run single-node evaluation for bin0 to bin9."
    )
    parser.add_argument(
        "-b",
        "--base-directory",
        default="./data/in-domain",
        help="Base directory containing bin0..bin9 (default: ./data/in-domain)",
    )
    parser.add_argument(
        "-f",
        "--file",
        default=DEFAULT_FILE_NAME,
        help=f'Input file name used in each bin (default: "{DEFAULT_FILE_NAME}")',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    evaluation_script = os.path.join(os.path.dirname(__file__), "..", "evaluation.py")

    completed_bins = []
    failed_bins = []
    skipped_bins = []

    for i in range(10):
        bin_name = f"bin{i}"
        dataset_dir = os.path.join(args.base_directory, bin_name)

        if not os.path.isdir(dataset_dir):
            print(f"[SKIP] {bin_name}: directory not found -> {dataset_dir}")
            skipped_bins.append(bin_name)
            continue

        command = [
            sys.executable,
            evaluation_script,
            "-d",
            dataset_dir,
            "-f",
            args.file,
        ]

        print(f"\n[RUN ] {bin_name}: {' '.join(command)}")
        result = subprocess.run(command, check=False)

        if result.returncode == 0:
            print(f"[ OK ] {bin_name}")
            completed_bins.append(bin_name)
        else:
            print(f"[FAIL] {bin_name} (exit code: {result.returncode})")
            failed_bins.append(bin_name)

    print("\nBatch evaluation summary")
    print("-" * 40)
    print(f"Completed: {len(completed_bins)} -> {completed_bins}")
    print(f"Failed:    {len(failed_bins)} -> {failed_bins}")
    print(f"Skipped:   {len(skipped_bins)} -> {skipped_bins}")

    if failed_bins:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
