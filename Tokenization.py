# byte pair encoding :>. & yes i wrote that code not ai ;)
from collections import Counter
sentence = "The quick brown fox jumps over the lazy dog. The dog barks at the fox, but the fox runs away quickly into the forest. In the forest, the fox meets a rabbit, and they both run through the trees together. The rabbit is quick, but the fox is quicker. They both stop near a river to drink water before continuing their journey through the quiet forest."

sentence = sentence.split(" ")
tokens = [byte.encode("utf-8")[0] for word in sentence for byte in word]

pair_collector = []
for i in range(len(tokens) - 1):
    pair_collector.append((tokens[i],tokens[i+1]))

# so now we gotta check the (pairs) , highest frequent PAIRS
counter_pairs = Counter(pair_collector)


# ah.
def merge(tokens , pairs , newids) :
    merging = []
    i = 0
    while i < len(tokens) :
        if i < len(tokens) - 1 and  (tokens[i],tokens[i+1]) == pairs :
            merging.append(newids)
            i+=2 # escape the 2 words
        else :
            merging.append(tokens[i])
            i+=1
    return merging

merg = {}
newids = 256
for i in range(30):
    count = Counter(zip(tokens , tokens[1:]))
    best_pair = count.most_common(1)[0][0]
    if count[best_pair] < 2 :
        break
    tokens = merge(tokens , best_pair , newids)
    merg[best_pair] = newids
    newids +=1

vocab = {i:bytes([i]) for i in range(256)}

for (p0,p1) , newids in merg.items():
    vocab[newids] = vocab[p0] + vocab[p1]

def encode(text:str):
    tokens = text.encode("utf-8")
    for pair , newids in merg.items():
        tokens = merge(tokens , pair , newids)
    return tokens

def decode(ids):
    result = b"".join(vocab[id] for id in ids)
    return result.decode("utf-8")


s ="The quick brown fox jumps over the lazy dog. The dog barks at the fox, but the fox runs away quickly into the forest. In the forest, the fox meets a rabbit, and they both run through the trees together. The rabbit is quick, but the fox is quicker. They both stop near a river to drink water before continuing their journey through the quiet forest."

decode(encode(s))

encoded = encode(s)
print(encoded)
r1 = len(s) / len(encoded)


# r1 - with 30 vocab size , 1.408906882591093% ratio
# r2 - with 300 vocab size , 1.5605381165919283% ratio
# r2 > r1 , r2 wins !!!
# I WANT TO SLEEP
print(r1)
