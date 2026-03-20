# Interesting Research

Related work for register-based genetic programming applied to neural sequence modeling.

---

## Closest Match: Linear Matrix Genetic Programming (LMGP)

**Almost exactly this idea, minus attention and minus LLM-directed search.**

Praczyk & Szymkowiak (2024) evolved programs of matrix operations (not scalar) over vector registers with persistent hidden state. Their "Matrix Operation Programs" (MOPs) process sequences recurrently — the register bank carries state across timesteps. They showed LMGP models were **up to 4x more accurate than LSTM/GRU** on underwater vehicle modeling, and critically, more interpretable because you can read the sequence of matrix operations.

- Registers are vector-valued (not scalar), with input register `r0` (read-only), output register `r1`, and hidden state registers `r2...rN`
- Operations: matrix multiply, register addition, element-wise product, logistic/tanh activations
- Hidden registers persist across time steps, giving recurrent behavior without explicit cell structures
- Programs are variable-length sequences of matrix operations, evolved via standard GP operators
- No attention mechanism. No LLM-directed mutation.

**Paper:** [Linear matrix genetic programming as a tool for data-driven black-box control-oriented modeling](https://www.nature.com/articles/s41598-024-63419-8)

---

## LLM-as-Evolutionary-Operator

### LLM_GP (Liventsev et al., 2024)

Replaces traditional GP mutation/crossover with LLM prompting. The LLM rephrases programs as text. Achieves comparable quality to standard GP but is **orders of magnitude slower** (~1600s vs 0.1s per run). Validates the concept but highlights the API latency bottleneck.

- Mutation works by prompting: "Rephrase the mathematical expression {X} into a new mathematical expression"
- Programs represented as text sequences, not parse trees
- External interpreters for code execution (not LLM-based evaluation)
- Few-shot prompting to guide LLM behavior

**Paper:** [Evolving Code with A Large Language Model](https://arxiv.org/html/2401.07102v1)

### FunSearch (DeepMind, Nature 2023)

Pairs a pretrained LLM with a systematic evaluator in an evolutionary loop to discover new mathematical constructions. Uses best-shot prompting (feed best programs back as context) and island-based population diversity. Made genuine mathematical discoveries (cap set problem).

- Operates in the space of programs, not constructions directly
- Iterative: sample high-scoring programs from database -> build prompt -> LLM generates new programs -> score and store
- Island-based evolutionary method for diversity
- First time an LLM made a new discovery on a challenging open math/science problem

**Paper:** [Mathematical discoveries from program search with large language models](https://www.nature.com/articles/s41586-023-06924-6)

### AlphaEvolve (Google DeepMind, May 2025)

Production-scale successor to FunSearch. Uses an ensemble of Gemini models as the evolutionary engine. Strongest existence proof that LLM-directed program evolution works at scale.

Key results:
- Found the first improvement to 4x4 complex matrix multiplication over Strassen's algorithm in 56 years (48 scalar multiplications)
- Discovered a data center scheduling heuristic now in production at Google, recovering 0.7% of worldwide compute
- Re-discovered SOTA on 75% of 50+ math problems, **beat SOTA on 20%**
- General-purpose: operates across scientific and engineering tasks by modifying code and optimizing for multiple objectives

**Paper:** [AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131)
**Blog:** [AlphaEvolve - Google DeepMind](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)

---

## Evolving Sequence Architectures with GP

### Dohan, So & Le (Google Brain, GECCO 2018)

Directly attempted to evolve modular neural sequence architectures via GP for language modeling and translation. Found that defining a search space flexible enough to express SOTA models while behaving well under evolution was the core challenge. Did not significantly beat LSTMs — the search space design was the bottleneck, not the evolutionary mechanism.

**Paper:** [Evolving modular neural sequence architectures with genetic programming](https://dl.acm.org/doi/abs/10.1145/3205651.3208782)

---

## Architecture Search Alternatives

### DARTS (Liu et al., 2019)

Makes architecture search differentiable by continuous relaxation of discrete choices. Found competitive recurrent cells for language modeling. This is the "use backprop for structure too" approach — the opposite bet from evolved programs, but sets the performance bar.

**Paper:** [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055)

### Differentiable Program Synthesis (Cui & Zhu, NeurIPS 2021)

Encodes program architecture as a probability distribution over grammar rules, searches via gradient descent. Produces interpretable programmatic classifiers competitive with neural nets.

**Paper:** [Differentiable Synthesis of Program Architectures](https://papers.nips.cc/paper_files/paper/2021/file/5c5a93a042235058b1ef7b0ac1e11b67-Paper.pdf)

---

## Recurrent Cartesian GP

### Turner & Miller (2014)

Adds recurrent connections to Cartesian GP, enabling it to handle partially observable / sequential tasks. Graph-based rather than linear, but same spirit of evolving recurrent computation.

**Paper:** [Recurrent Cartesian Genetic Programming](https://link.springer.com/chapter/10.1007/978-3-319-10762-2_47)

---

## Other Relevant Work

### LGP for Graph Neural Networks (GECCO 2025)

Applies linear genetic programming to design graph neural network architectures. Each individual is a list of register-based instructions evolved through genetic operations.

**Paper:** [Linear Genetic Programming for Design Graph Neural Networks](https://dl.acm.org/doi/pdf/10.1145/3712255.3734278)

### LLMatic (GECCO 2024)

Neural architecture search via large language models and quality-diversity optimization.

**Paper:** [LLMatic: Neural Architecture Search Via Large Language Models](https://dl.acm.org/doi/10.1145/3638529.3654017)

---

## Novelty Map

| Idea | Prior art | Status |
|------|-----------|--------|
| Register-based GP on sequences | LMGP (2024) validates this, beats LSTM/GRU | Established |
| ATTEND as an opcode | Not done. LMGP uses matrix ops but no cross-position attention | **Novel** |
| LLM-directed mutation for neural programs | FunSearch/AlphaEvolve prove LLM-directed evolution works at scale | Applying it to neural program evolution specifically is new |
| Two-timescale optimization (LLM for structure, backprop for weights) | DARTS does both with backprop; FunSearch does both without backprop | **Novel combination** |
| Interpretable language model via evolved programs | Dohan et al. tried, didn't beat LSTMs | Open problem |
