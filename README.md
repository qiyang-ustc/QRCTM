# QR-CTMRG: Minimal ⚡️ Working Example

This repository provides a minimal working example of the **QR-CTMRG** (Corner Transfer Matrix Renormalization Group with QR decomposition) algorithm, as described in our paper:

> **[Accelerating two-dimensional tensor network contractions using QR-decompositions](https://arxiv.org/abs/2505.00494)**
> *Yining Zhang, Qi Yang, Philippe Corboz, 2025*

This code demonstrates a simple and clean implementation of the QR-CTMRG algorithm for 2D tensor network contraction, suitable for variational optimization of PEPS (Projected Entangled Pair States) and related models.

## Features

- **Minimal**: The code is kept as simple as possible for clarity and educational purposes.
- **JAX-based**: Leverages JAX for automatic differentiation and fast linear algebra.
- **Heisenberg Model Example**: Includes a variational optimization of the 2D Heisenberg model energy. **⚡️Extremely Fast!**
- **Basis Symmetrization**: Uses a general basis symmetrization routine (could be further simplified, but kept for clarity and correctness).

## Requirements

- Python 3.8+
- [JAX](https://github.com/google/jax) (with GPU/TPU support optional)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [opt_einsum](https://optimized-einsum.readthedocs.io/)
- [matplotlib](https://matplotlib.org/)
- [jaxtyping](https://github.com/google/jaxtyping) (optional, for type hints)

Install dependencies via pip:

```bash
pip install jax numpy scipy opt_einsum matplotlib jaxtyping
```

## Usage

Simply run:

```bash
python heisenberg.py
```

- The script will perform variational optimization of a PEPS ansatz for the 2D Heisenberg model.
- Energy and timing curves will be saved as `energy.png` and `time.png`.
- All main algorithmic steps are contained in a single file for clarity.

You can adjust parameters such as `chi`, `D`, `maxiter`, etc. in the `main()` function.

## Reference

If you use this code or find it helpful, please cite:

```
@misc{zhang2025acceleratingtwodimensionaltensornetwork,
      title={Accelerating two-dimensional tensor network contractions using QR-decompositions}, 
      author={Yining Zhang and Qi Yang and Philippe Corboz},
      year={2025},
      eprint={2505.00494},
      archivePrefix={arXiv},
      primaryClass={cond-mat.str-el},
      url={https://arxiv.org/abs/2505.00494}, 
}
```

## Notes

- The current implementation is intentionally not the most minimal possible, especially in the basis construction part, to avoid logical errors and maintain clarity.
- For further simplification or adaptation to other models, feel free to modify the code.