# Stanford-CS336-LM
<b>Implementing things i learn from these lectures to level up AI skills.</b> 
<br/>
$`where \ all \ codes \ here \ are \ written \ by \ me`$ @Meshojs 

# Lecture 1 : Tokenization | $`using \ Byte \ Pair \ Encoding`$
	sentence = "Hello world" 
 1. **tokenizer 1** =  3 bytes `["Hello", " ", "world"]` 
 2. **tokenizer 2** = 2 bytes `["Hello", " world"]`

* $r1 = \frac{11}{3}$
* $r2 = \frac{11}{2}$
* $where \ rule = \frac{number \ of \ bytes}{number \ of \ tokens}$

> ***r2 wins*** , r2 has bigger $compression \ ratio$ , `which means lower tokens ,  {efficient}`
#

# Lecture 2 : Resource Accounting - Precisions <br>
1 - let's say we have a 350M Parameter Model. Lad1-350M for example `(gonna share it soon)`, and i want to Train it on $`H-100 \ .10B\ Tokens`$. <br>
$`where \ MFU \ of \ H-100 \  is \  approx. \ 60e^{12}`$ <br> 
using 6ND.

* $total \ flops \ = \ 6 *\ 350^{10^{6}}$ <br>
* $cluster \ through \ put = Num \ of \ gpus \ * \ MFU \ * \ (gpu \ flops \ of \ fp32 \ for \ example )$ <br>
* $time = \frac{total}{cluster}$

<br>

2 - Precisions 

| Format      | Exponent bits | Mantissa bits | Range   | Precision (~digits) |
|-------------|--------------|----------------|---------|----------------------|
| FP32        | 8            | 23             | huge    | ~7                   |
| TF32        | 8            | 10             | huge    | ~3                   |
| FP16        | 5            | 10             | small   | ~3                   |
| BF16        | 8            | 7              | huge    | ~2                   |
| FP8 (E4M3)  | 4            | 3              | small   | ~1                   |
| FP8 (E5M2)  | 5            | 2              | medium  | <1                   |


FP32 has the best Precision but expensive at computition, BF-16 has the same Exponent bits (range) <br>
$which \ made \ it \ really \ good \ at \ training \ but \ low \ cost.$
