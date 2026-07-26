# Stanford-CS336-LM

<b>Implementing things i learn from these lectures to level up AI skills.</b>
<br/>

$`where \ all \ codes \ here \ are \ written \ by \ me`$ @Meshojs

---

## Lecture 1 : Tokenization | $`using \ Byte \ Pair \ Encoding`$

**Algorithm:** https://en.wikipedia.org/wiki/Byte-pair_encoding

sentence = "Hello world"

1. **tokenizer 1** = 3 bytes `["Hello", " ", "world"]`
2. **tokenizer 2** = 2 bytes `["Hello", " world"]`

- $r1 = \frac{11}{3}$
- $r2 = \frac{11}{2}$
- $where \ rule = \frac{number \ of \ bytes}{number \ of \ tokens}$

> ***r2 wins*** , r2 has bigger $compression \ ratio$ , `which means lower tokens ,  {efficient}`

---

## Lecture 2 : Resource Accounting - Precisions

### 1. FLOPs & Throughput

let's say we have a 350M Parameter Model. Lad1-350M for example `(gonna share it soon)`, and i want to Train it on $`H-100 \ .10B\ Tokens`$.

$`where \ MFU \ of \ H-100 \  is \  approx. \ 60e^{12}`$

using 6ND.

- $total \ flops \ = \ 6 *\ 350^{10^{6}}$
- $Throughput \ = \ Ngpus \ × \ perGPUFLOPs \ × \ MFU$
- $time = \frac{total}{cluster}$

### 2. Precisions

| Format      | Exponent bits | Mantissa bits | Range   | Precision (~digits) |
|-------------|--------------|----------------|---------|----------------------|
| FP32        | 8            | 23             | huge    | ~7                   |
| TF32        | 8            | 10             | huge    | ~3                   |
| FP16        | 5            | 10             | small   | ~3                   |
| BF16        | 8            | 7              | huge    | ~2                   |
| FP8 (E4M3)  | 4            | 3              | small   | ~1                   |
| FP8 (E5M2)  | 5            | 2              | medium  | <1                   |

FP32 has the best Precision but expensive at computition, BF-16 has the same Exponent bits (range)

$which \ made \ it \ really \ good \ at \ training \ and \ low \ cost.$

### 3. Compute bound $`Or`$ Memory bound

performance characteristics where they help u find bottlenick in memory or compute

> *Example : so now i am training a model LETS say it is Lad1-350M haha again,*
>
> *so when i trained Lad i discoverd that the model - is slow in training or inference i went to the gpu profile and i found that gpu is idle waiting for data to get inside*
>
> *"gpue - cores" to compute*
>
> *THIS is a Memory bound, Low bandwidth speed.*

### 4. Memory

when training a Deeplearning model, ofc u need memory to load and save the Parameters , gradient (backprop), optimizer, Activation

so the Rules:

> - $`Parameters \ = 2 * (D * D * L)`$
> - $`Activations \ = 2 * (B * D * L)`$
> - $`gradients = \ 2 * Parameters`$
> - $`optimizer_state \ = \ 4 * Parameters`$
 
- Yes, gradients and optimizers are killers


## Lecture 3 : Architecture 

1 - Post Norm vs Pre Norm
   > who tf use PostNorm in 2026 haha, anyways PreNorm is actually better, (stable)

   $`PreNorm\ = \ x \ + \ MHA(LN(x))`$ <br>
   $`postNorm\ = \ LN(x \ + \ MHA(x))`$ <br>
 ### why tho ?
    let me tell u why. when using the preNorm, backprop goes through x (main stream)
    but PostNorm has to backprop through LN then x (main stream) which make it unstable
 
  
2 - LayerNorm vs RMS Norm
   > RMSNorm wins in computition cost , low , fast , <b>Because</b> LayerNorm has to measure Mean,variance and it has Bias and learnable gain  <br>
   > RMSNORM doesn't need all of that

