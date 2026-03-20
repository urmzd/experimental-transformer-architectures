# Gauss's Discoveries × RegisterGPT

How the mathematics of Carl Friedrich Gauss (1777–1855) applies to register-based language modeling.

---

## 1. The Fast Fourier Transform (1805)

Gauss invented the FFT to interpolate asteroid orbits from trigonometric series — 160 years before Cooley-Tukey rediscovered it.

**Current state:** RegisterGPT v3 builds an explicit `(V, 2*n_basis)` matrix of cos/sin values, stores it as a buffer, and multiplies through it: `basis @ coeffs.T`. This is O(V * n_basis) to build the projection and O(V * C) to apply it. Only 16 of 512 possible frequencies are used.

**Gauss's improvement:** Since V = 1024 = 2^10 (a perfect power of 2 — the ideal FFT case), replace the explicit basis with `torch.fft.rfft`. This:
- Extracts frequency components in O(V log V) = O(10K) instead of O(V * n_basis) = O(33K)
- Accesses ALL 512 frequencies, not just 16 — then selects the most useful via learned weights
- Eliminates the `fourier_basis` buffer entirely (no storage, no serialization)

The mathematical equivalence: `x @ basis` computes the same thing as `rfft(x)` separated into real/imaginary parts. The current code is a manual DFT. Gauss's algorithm computes it faster.

**Implemented in:** `GaussProjection`, `GaussSynthesis`, `GaussRegisterOp`

## 2. Complex Numbers & Gaussian Integers

Gauss systematized complex arithmetic and proved unique factorization for Gaussian integers (a + bi, a,b ∈ Z).

**Current state:** The Fourier basis stores cos and sin as separate real columns. Learned coefficients are paired reals. This obscures the natural complex structure and doubles the indexing complexity.

**Gauss's improvement:** The cos/sin pair IS a complex exponential: e^{2πikx/V} = cos(2πkx/V) + i·sin(2πkx/V). A complex coefficient r·e^{iφ} encodes amplitude r (how much this frequency matters) and phase φ (where it peaks in vocabulary space). This is more interpretable than two opaque real weights.

In `GaussRegisterGPT`, FFT naturally produces complex values. We concatenate real/imag for linear transforms (since PyTorch optimizers handle real tensors more reliably), but the mathematical structure is complex throughout.

**Implemented in:** All FFT-based modules use `torch.fft.rfft` → complex → `cat([real, imag])` → learned transform

## 3. Cyclic Convolution (Disquisitiones Arithmeticae, 1801)

In his masterwork, Gauss systematized modular arithmetic and studied the characters of cyclic groups. The Fourier basis functions over Z/VZ are exactly these characters.

**Current state:** `FourierRegisterOp` reads from vocabulary space, mixes channels, writes back. The read/write patterns are parameterized by Fourier coefficients, making them implicitly smooth.

**Gauss's improvement:** The read→transform→write pipeline is a cyclic convolution over the vocabulary ring Z/VZ. By the convolution theorem (which follows from Gauss's character theory):

```
Convolution in vocab space  ←→  Pointwise multiply in frequency domain
```

`GaussRegisterOp` implements this directly: FFT → learned frequency-domain transform → IFFT. The output is band-limited (only n_freq harmonics), providing a built-in smoothness regularizer. The model can only produce smooth updates to the register state in each step — sharpness accumulates through residual connections and nonlinearities over multiple steps.

**Implemented in:** `GaussRegisterOp`

## 4. Gaussian Quadrature

Gauss proved that optimal numerical integration uses specifically chosen evaluation points (roots of Legendre polynomials), not uniform spacing. With n points you can exactly integrate polynomials of degree 2n-1.

**Possible application:** The current model uses frequencies 1, 2, ..., n_basis uniformly. But language statistics are not uniform over vocabulary — they follow Zipf's law. The optimal frequencies for capturing vocabulary structure may be non-uniform. Learning the frequencies themselves (not just their weights) would be Gaussian quadrature applied to the register state.

**Status:** Not yet implemented. Would require interpolation in the FFT output since FFT gives values at integer frequencies only.

## 5. Method of Least Squares & the Orbit of Ceres (1801)

Gauss predicted Ceres's orbit from 41 days of observations using least squares with physical priors (Kepler's laws).

**The analogy:**
- Sparse observations → one-hot token inputs
- Physical prior (elliptical orbits) → architectural bias (Fourier smoothness, decay memory)
- Orbit prediction → next-token prediction

Gauss succeeded because he had the *right prior*. The Fourier basis is one prior (vocabulary has smooth frequency structure). The exponential decay is another (recent tokens matter more). The question is whether these are the *right* priors for language.

**Possible application:** Parameterize the memory decay as a family of "orbital" functions with learnable physical parameters (energy, eccentricity, period) instead of a single exponential.

**Status:** Not yet implemented.

## 6. Theorema Egregium & Differential Geometry (1827)

Gauss proved that curvature is intrinsic — it depends only on distances measured within the surface, not on the ambient embedding.

**Possible application:** The register state is a distribution over words. The space of probability distributions has intrinsic Riemannian geometry (the Fisher information metric). Current operations are Euclidean (additive residuals). Information-geometric operations would:
- Use multiplicative updates: `x = x * exp(scale * f(x))` (exponential map on the simplex)
- Ensure rare-word updates are proportionally larger (matching information-geometric intuition)
- Respect the natural geometry of probability distributions

**Status:** Not yet implemented.

## 7. Roots of Unity & Multi-Resolution (1796)

At age 19, Gauss proved which regular polygons are constructible by analyzing roots of unity. For V = 1024 = 2^10, the roots of unity have a binary tree structure:
- Level 0: DC (1 component)
- Level 1: 2 components (even vs odd vocabulary indices)
- Level k: 2^k components
- Level 9: 512 components (full resolution)

**Possible application:** Early register steps operate on low frequencies (broad semantic categories: noun? verb?), later steps add higher frequencies (specific word choice). This mirrors language generation: first decide the category, then refine.

**Status:** Not yet implemented. Could be done by varying `n_freq` per step.

---

## GaussRegisterGPT: What Changed

| Component | RegisterGPT v3 | GaussRegisterGPT |
|-----------|---------------|-----------------|
| Frequency extraction | `x @ basis` where `basis` is stored (V, 2*n_basis) buffer | `rfft(x)` — computed on the fly, O(V log V) |
| Frequencies used | 16 (fixed) | 64 default (configurable) |
| Vocab→channel projection | `FourierProjection`: basis matmul + learned coeffs | `GaussProjection`: FFT + learned linear |
| Channel→vocab projection | `basis @ coeffs.T` transpose | `GaussSynthesis`: learned linear + IFFT |
| Register op | Read via softmax(basis @ coeffs), mix, write via basis @ coeffs | FFT → channel bottleneck → IFFT (cyclic convolution) |
| Stored buffers | `fourier_basis` (V × 2*n_basis) | None |
| Cross-position mixing | Decay-weighted associative memory (unchanged) | Same mechanism, FFT-based projections |

### Parameter counts (8 steps, n_channels=128)

| Config | Params |
|--------|--------|
| RegisterGPT v3 (n_basis=16) | ~329K |
| GaussRegisterGPT (n_freq=16) | ~329K (parameter-equivalent) |
| GaussRegisterGPT (n_freq=64) | ~919K (4x more frequency resolution) |

With n_freq=16, the models are parameter-equivalent — the only difference is FFT vs explicit basis matmul. With n_freq=64, GaussRegisterGPT uses more parameters but captures 4x more of the frequency spectrum.
