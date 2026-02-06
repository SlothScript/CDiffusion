import gzip
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.decoders import BPEDecoder

CORPUS = "../data/corpus_clean.txt.gz"

tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(
    vocab_size=32768, # type: ignore
    special_tokens=["[PAD]", "[UNK]", "[MASK]"] # type: ignore
)

with gzip.open(CORPUS, "rt") as f:
    training_corpus = []
    for i,sentence in enumerate(f):
        training_corpus.append(sentence)
        if i == 250000: # Assuming 512 bytes per paragraph
            break

    tokenizer.train_from_iterator(training_corpus, trainer)

tokenizer.save("tokenizer.json")


# -- Test tokenizer --


# Test encoding
text = "The quick brown fox jumped over the lazy dog. A simple man with a great thought. An MLM is a model that starts with full [MASK] tokens, then replaces them with new tokens."
encoded = tokenizer.encode(text)
print(f"Text: {text}")
print(f"Token IDs: {encoded.ids}")
print(f"Tokens: {encoded.tokens}")

# Test decoding
tokenizer.decoder = BPEDecoder()
decoded = tokenizer.decode(encoded.ids)
print(f"Decoded: {decoded}")

# Check special tokens
print(f"\n[MASK] token ID: {tokenizer.token_to_id('[MASK]')}")
print(f"[PAD] token ID: {tokenizer.token_to_id('[PAD]')}")
print(f"[UNK] token ID: {tokenizer.token_to_id('[UNK]')}")
