# QR-CTMRG: Minimal ⚡️ Working Example

This repository provides a minimal working example of the **QR-CTMRG** (Corner Transfer Matrix Renormalization Group with QR decomposition) algorithm, as described in our paper:

> **[Accelerating two-dimensional tensor network contractions using QR-decompositions](https://arxiv.org/abs/2505.00494)**
> *Yining Zhang, Qi Yang, Philippe Corboz, 2025, (Co-first author)*

> **[Efficient iPEPS Simulation on the Honeycomb Lattice via QR-based CTMRG, Phys. Rev. B 113, 085109](https://journals.aps.org/prb/abstract/10.1103/9gmp-byx8)**
> *Qi Yang, Philippe Corboz, 2025*

This code demonstrates a simple and clean implementation of the QR-CTMRG algorithm for 2D tensor network contraction, suitable for variational optimization of PEPS (Projected Entangled Pair States) and related models.

## Features

- **Minimal**: The code is kept as simple as possible for clarity and educational purposes.
- **Heisenberg Model Example**: Includes a variational optimization of the 2D Heisenberg model energy. **⚡️Extremely Fast!**
- **Kitaev Model Example**: Includes a variational optimization of the Pure Isotropic Kitaev model energy. **⚡️Extremely Fast!**

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
@article{9gmp-byx8,
  title = {Efficient iPEPS simulation on the honeycomb lattice via QR-based corner transfer matrix renormalization group},
  author = {Yang, Qi and Corboz, Philippe},
  journal = {Phys. Rev. B},
  volume = {113},
  issue = {8},
  pages = {085109},
  numpages = {12},
  year = {2026},
  month = {Feb},
  publisher = {American Physical Society},
  doi = {10.1103/9gmp-byx8},
  url = {https://link.aps.org/doi/10.1103/9gmp-byx8}}
```
- The current implementation is intentionally not the most minimal possible, especially in the basis construction part, to avoid logical errors and maintain clarity.
- For further simplification or adaptation to other models, feel free to modify the code.
