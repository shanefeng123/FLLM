from collections import Counter

from utils import *
from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, OpenAIGPTLMHeadModel, OpenAIGPTTokenizerFast
import random

random.seed(42)

nltk.download('punkt')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "gpt2"
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
# Create some toy train data to demonstrate the attack
data = ["<|startoftext|>I love love love<|endoftext|>", "<|startoftext|>I hate hate hate<|endoftext|>",
        "<|startoftext|>I love and hate you<|endoftext|>"]
train_data = tokenizer(data, return_tensors="pt", padding=True,
                       truncation=True, max_length=8)
train_data["labels"] = train_data["input_ids"].clone()
# Train the client model with the toy train data
loss, attentions, hidden_states, logits = train_batch(client, train_data, DEVICE)
# # Collect all layer gradients
original_grads_name = {}
for name, param in client.named_parameters():
    original_grads_name[name] = param.grad
# Get the token embedding layer gradients
token_grads = original_grads_name['transformer.wte.weight'].clone().detach()
# Start the attack
token_list = []
# Because the server would know how many tokens in a sentence, and how many sentence in a batch.
# We can just calculate how many tokens there should be in a batch
num_missing_tokens = train_data["input_ids"].shape[0] * train_data["input_ids"].shape[1]
# Calculate the magnitude of each token embedding with L2 norm. The result is a vector with the same length as the vocabulary size
token_grads_norm = torch.linalg.vector_norm(token_grads, dim=1)
# If the magnitude of a token embedding is greater than 1, we can add it to the token list
valid_classes = np.where(token_grads_norm.numpy() > 1)[0].tolist()
token_list += [*valid_classes]
# This is the naive part. We just assume that the impact for each occurrence of a token is the same,
# so that we can just calculate it by dividing the total magnitude of gradients by the number of missing tokens.
log_token_grads_norm = torch.log(token_grads_norm)
m_impact = log_token_grads_norm[valid_classes].sum() / num_missing_tokens
log_token_grads_norm[valid_classes] = log_token_grads_norm[valid_classes] - m_impact

# Stage 2
# Get the token that has the largest magnitude, add it to the list, then subtract it with the impact value
# Do this until we reach the number of missing tokens
while len(token_list) < num_missing_tokens:
    selected_idx = valid_classes[log_token_grads_norm[valid_classes].argmax()]
    token_list.append(selected_idx)
    log_token_grads_norm[selected_idx] -= m_impact

print(Counter(torch.flatten(train_data["input_ids"]).tolist()))
print(Counter(token_list))
