import torch.nn

from utils import *
from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, OpenAIGPTLMHeadModel, OpenAIGPTTokenizerFast
import random
from torch.nn.functional import softmax

random.seed(42)

nltk.download('punkt')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "gpt"
NUM_OF_CLIENTS = 30

# Initialise the server model
if MODEL_NAME == "gpt2":
    server = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME, bos_token="<|startoftext|>", eos_token="<|endoftext|>",
                                                  pad_token="<|pad|>")
else:
    server = OpenAIGPTLMHeadModel.from_pretrained("openai-gpt").to(DEVICE)
    tokenizer = OpenAIGPTTokenizerFast.from_pretrained("openai-gpt", bos_token="<|startoftext|>",
                                                       eos_token="<|endoftext|>",
                                                       pad_token="<|pad|>")

# Resize the token embedding of the model to match the new vocabulary size
server.resize_token_embeddings(len(tokenizer))
server_parameters = get_parameters(server)
# Initialise client models
client = set_parameters(server, server_parameters)
# Load the data
train_data = tokenizer("<|startoftext|>Hello how are you<|endoftext|>", return_tensors="pt", padding=True,
                       truncation=True, max_length=3)
train_data["labels"] = train_data["input_ids"].clone()
batch = train_data
print(tokenizer.decode(train_data["input_ids"][0].tolist()))
# # Train the model with this first batch only
loss, attentions, hidden_states, logits = train_batch(client, batch, DEVICE)
# # Collect all the gradients
original_grads = {}
for name, param in client.named_parameters():
    original_grads[name] = param.grad

embedding_hidden_states = hidden_states[0].detach().clone()
first_attention_weights_grads = original_grads["transformer.h.0.attn.c_attn.weight"].detach().clone()
first_attention_bias_grads = original_grads["transformer.h.0.attn.c_attn.bias"].detach().clone()
results = first_attention_weights_grads / first_attention_bias_grads

for i in range(embedding_hidden_states.size()[1]):
    embedding = embedding_hidden_states[0][i, :]
    for j in range(results.size()[1]):
        recovered_embedding = results[:, j]
        if torch.eq(embedding, recovered_embedding).all():
            print(f"Number {i} embedding is recovered")