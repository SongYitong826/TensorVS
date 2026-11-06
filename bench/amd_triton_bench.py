"""Run the shared TensorVS AMD benchmark using its Triton search route.

The CSV metrics and timing method are identical to amd_tensor_bench.py.
Results use TensorVS_Triton_ filenames in /root/TensorVS/result/.
"""

from amd_tensor_bench import main


if __name__ == "__main__":
    main(default_route="triton")
