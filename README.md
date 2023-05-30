# NCIHackaton2023.jl

Repo for the NCI-NVIDIA hackaton on June 2023.

## Objective

The objective is to implement sum-factorization techniques for GPU architectures, within
the [Gridap.jl](https://github.com/gridap/Gridap.jl) ecosystem.

Sum-factorization takes advantage of the tensorial nature of Finite-Element(FE) shape-functions
and quadratures to reduce the number of operations required to evaluate FE integrals, by formulating the
integrals as a series of successive tensor contractions with dense small matrices.

This allows for fast and computationally-intensive matrix-free implementations of FE methods.
These implementations compete with the traditional sparse matrix-vector product methods, which
have traditionally been memory-bound. The advantages of sum-factorization techniques become
very noticeable for high-order FE schemes.

## Profiling

### CPU-only: Profile.jl + Flamegraphs.jl/PProf.jl

Compile the profiles by running from the project directory the following command

```bash
  julia --project=. profiling/profile.jl
```

Visualize the profiles by interactively running `profiling/viz_profile.jl`. You can choose between flamegraph vizualisation or PProf.

## Literature

- [S.Müthing 2017](https://arxiv.org/abs/1711.10885)
