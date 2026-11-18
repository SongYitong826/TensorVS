"""Backward-compatible Ascend entry point for the portable Tensor tuner."""

try:
    from .tune_tensor_tiles import main
except ImportError:
    from tune_tensor_tiles import main


if __name__ == "__main__":
    main()
