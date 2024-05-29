# Five qubit code

This repository contains my implementation of the (non-fault tolerant) [Five-qubit error correcting code](https://en.wikipedia.org/wiki/Five-qubit_error_correcting_code) using [Qiskit](https://qiskit.org/) package for Python.

This python program is useful to check that the five-qubit code indeed produces non-degenerate outcomes from the syndrome measurements.

## The quantum circuit

The quantum circuit of the five-qubit code is shown below.

![5_qubit_code_qcircuit](assets/5_qubit_code_qcircuit.png)

## Prerequisite

```shell
python3 -m pip install qiskit
python3 -m pip install qiskit-aer
```

## How to run?

Go to the corresponding path then from your commandline, run:

```shell
    python3 main.py
```

## References

Some references that I found useful while implementing the five-qubit code.

- https://web.mit.edu/8.371/www/lectures/lect06.pdf
- https://www.lorentz.leidenuniv.nl/quantumcomputers/literature/preskill_7.pdf
- https://arxiv.org/pdf/1010.3242.pdf
- https://www.physics.unlv.edu/~bernard/MATH_book/Chap9/Notebook9_3.pdf
