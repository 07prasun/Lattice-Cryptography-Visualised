# Lattice Cryptography Visualisations

An interactive tool for looking at the real lattice geometry behind ML-KEM. It starts from the actual Module-LWE instance, not a generic 2D lattice. The 2D/3D views are projections; the full object is higher-dimensional and computed internally.

## What it does

You can move from ML-KEM algebra to Module-LWE equations to an attack-lattice representation to reduction/search geometry without changing models. The secret, public matrix, error, public-key relation, ciphertext, and reconciliation behavior are tied to the same underlying state. Projection changes recompute from that state.

You can rotate, zoom, pan, change the projection basis and dimensions, inspect vectors and samples, highlight secret/error/noise components, and see how the same high-dimensional object looks under different projections.

## Basis and reduction

There is a real basis view: lengths, angles, Gram-Schmidt information, determinant, short vectors. You can run LLL step-by-step or continuously. The lattice does not change; only the basis does. LLL is not SVP.

BKZ is included with variable block size β. β is a parameter of the reduction, not the dimension of the whole lattice. The UI keeps ambient/attack-lattice dimension separate from block dimension.

Sieving is represented with sampled vectors and pair differences. Displayed counts are samples. The scaling relationship is the point, not the visible dots.

## Attack view

The attacker gets only public data. It builds the attack representation, reduces the basis, searches for short vectors, proposes a candidate secret, and verifies it. You can increase attack resources and inspect intermediate quantities. The hidden secret is never handed to the attacker.

## Parameters and noise

ML-KEM parameter sets are selectable: n = 256, q = 3329, k = 2/3/4. Secret and error are centered binomial. Noise is part of the model, not decoration. The UI distinguishes the exact relation from the noisy observation. Increasing noise affects hardness and correctness differently.

If you edit parameters away from ML-KEM, the tool labels the result as a modified construction, not ML-KEM.

## Security plots

No single security score. Hardness, reduction cost, attack success probability, decryption failure probability, and performance are separate. Classical and quantum estimates are separate, with the cost model identified. Visualization runtime is not attack cost.

## Fidelity

Exact is exact. Projected is labeled. Heuristic is labeled. Visible points are not the full problem. ML-KEM dimensions and attack-lattice dimensions are never conflated.

## Stack
React. The Mathematics is implemented by myself and frontend done by Claude, including this readme.

## How to run
It is a react app so you can use the usual commands

You need Node 18+.
```bash
npm install
npm run dev
```

Open the URL it prints, usually `http://localhost:5173`.
