# Stanford-CS336-LM

Implementing concepts from the Stanford CS336 (Language Modeling from Scratch) lectures to build hands-on AI/ML engineering skills.

All code in this repository is written by me — [@Meshojs](https://github.com/Meshojs) 

---

## Lecture 1: Tokenization — Byte Pair Encoding

**Reference:** [Byte-pair encoding (Wikipedia)](https://en.wikipedia.org/wiki/Byte-pair_encoding)

Consider the sentence: `"Hello world"`

| Tokenizer | Tokens | Count |
|---|---|---|
| Tokenizer 1 | `["Hello", " ", "world"]` | 3 |
| Tokenizer 2 | `["Hello", " world"]` | 2 |

Compression ratio is defined as:

$$r = \frac{\text{number of bytes}}{\text{number of tokens}}$$

$$r_1 = \frac{11}{3}, \qquad r_2 = \frac{11}{2}$$

**Result:** Tokenizer 2 wins — a higher compression ratio means fewer tokens, which is more efficient.

---

## Lecture 2: Resource Accounting & Precision

### 2.1 FLOPs & Throughput

Suppose we train a 350M-parameter model (**Lad1-350M**, to be released soon) on 10B tokens using an H100 GPU.

$$\text{MFU}_{\text{H100}} \approx 60 \times 10^{12} \text{ FLOPs/s}$$

Using the $6ND$ approximation:

$$\text{Total FLOPs} = 6 \times 350 \times 10^{6} \times N_{\text{tokens}}$$

$$\text{Throughput} = N_{\text{GPUs}} \times \text{FLOPs}_{\text{GPU}} \times \text{MFU}$$

$$\text{Training Time} = \frac{\text{Total FLOPs}}{\text{Throughput}}$$

### 2.2 Numerical Precision Formats

| Format | Exponent bits | Mantissa bits | Range | Precision (~digits) |
|---|---|---|---|---|
| FP32 | 8 | 23 | huge | ~7 |
| TF32 | 8 | 10 | huge | ~3 |
| FP16 | 5 | 10 | small | ~3 |
| BF16 | 8 | 7 | huge | ~2 |
| FP8 (E4M3) | 4 | 3 | small | ~1 |
| FP8 (E5M2) | 5 | 2 | medium | <1 |

FP32 offers the best precision but is computationally expensive. BF16 shares FP32's exponent range (dynamic range) while being far cheaper to compute — making it a strong default for training.

### 2.3 Compute-Bound vs. Memory-Bound

Performance profiling helps identify whether a bottleneck is in compute or memory bandwidth.

> **Example:** While training Lad1-350M, I noticed slow throughput during training/inference. GPU profiling showed the compute cores idle, waiting on data delivery.
>
> **Diagnosis:** This is a **memory-bound** regime — bottlenecked by bandwidth, not compute.

### 2.4 Memory Accounting

Training a deep learning model requires memory for parameters, gradients, optimizer state, and activations:

$$\text{Parameters} = 2 \cdot (D \cdot D \cdot L)$$

$$\text{Activations} = 2 \cdot (B \cdot D \cdot L)$$

$$\text{Gradients} = 2 \cdot \text{Parameters}$$

$$\text{Optimizer State} = 4 \cdot \text{Parameters}$$

**Takeaway:** gradients and optimizer state dominate memory usage.

---

## Lecture 3: Architecture

### 3.1 Pre-Norm vs. Post-Norm

Pre-Norm is the modern standard — nobody uses Post-Norm in 2026. It's simply more stable.

$$\text{Pre-Norm}: \quad x + \text{MHA}(\text{LN}(x))$$

$$\text{Post-Norm}: \quad \text{LN}(x + \text{MHA}(x))$$

**Why Pre-Norm wins:** whot tf use POSTNORM ?? . with Pre-Norm, gradients backpropagate directly through the residual stream $x$. With Post-Norm, gradients must pass through LN before reaching the residual stream, which destabilizes training.

### 3.2 LayerNorm vs. RMSNorm

RMSNorm is cheaper and faster to compute. LayerNorm requires computing both mean and variance, plus a learnable bias and gain. RMSNorm drops all of that — normalizing only by root-mean-square.

### 3.3 Gated Activations

Rather than a plain activation function, gated activations introduce a learnable projection $V$ that lets the model decide which tokens to keep or suppress.

**Example:** `"The cat sat on the mat"` → The **(kill)**, cat **(keep)**, sat **(keep)**, on **(kill)**, the **(kill)**, mat **(keep)**

$$\text{Swish}(z) = z \cdot \sigma(z)$$

$$\text{SwiGLU}(x) = \text{Swish}(xW_1) \cdot (xV)$$

$$\text{GLU}(x) = \sigma(xW_1) \cdot (xV)$$
