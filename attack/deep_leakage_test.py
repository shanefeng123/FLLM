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
                       truncation=True, max_length=8)
train_data["labels"] = train_data["input_ids"].clone()
batch = train_data
# # Train the model with this first batch only
loss, attentions, hidden_states, logits = train_batch(client, batch, DEVICE)
# # Collect all the gradients
original_grads = []
for param in client.parameters():
    original_grads.append(param.grad)

original_grads = list((_.detach().clone() for _ in original_grads))
client_parameters = get_parameters(client)
#
# Create dummy input text embedding vector, which is the same as the first batch of the first client
dummy_input_embedding = torch.randn(
    (batch["input_ids"].size()[0], batch["input_ids"].size()[1], client.config.n_embd)).to(DEVICE).requires_grad_(True)
dummy_labels = torch.randn((batch["input_ids"].size()[0], batch["input_ids"].size()[1], client.config.vocab_size)).to(
    DEVICE).requires_grad_(True)
#
token_embeddings = server.transformer.tokens_embed.weight
optimizer = torch.optim.AdamW([dummy_input_embedding, dummy_labels], lr=0.001)
server = set_parameters(server, server_parameters)
iters = 0
while True:
    def closure():
        optimizer.zero_grad()
        dummy_pred = server(inputs_embeds=dummy_input_embedding).logits
        loss_fc = torch.nn.CrossEntropyLoss()
        dummy_loss = loss_fc(dummy_pred[:, 1:, :].reshape(-1, dummy_pred.size()[-1]),
                             softmax(dummy_labels[:, :-1, :], dim=-1).reshape(-1, dummy_labels.size()[-1]))
        dummy_grads = torch.autograd.grad(dummy_loss, server.parameters(), create_graph=True)

        grad_diff = 0
        for gx, gy in zip(dummy_grads, original_grads):
            # sum the l1 norm and l2 norm of the gradients
            grad_diff += ((gx - gy) ** 2).sum()
        grad_diff.backward()

        return grad_diff


    optimizer.step(closure)
    # print(server.parameters().__iter__().__next__())
    current_loss = closure()
    if iters % 10 == 0:
        # Copy the dummy embeddings
        dummy_input_embedding_copy = dummy_input_embedding[0, :, :].detach().clone()
        # Calculate the similarity between each row of the dummy embeddings and each row of the token embeddings
        recovered_tokens = []
        similarity_matrix = torch.nn.functional.cosine_similarity(dummy_input_embedding_copy.unsqueeze(1),
                                                                  token_embeddings.unsqueeze(0), dim=-1)
        for i in range(similarity_matrix.size()[0]):
            # Find the index of the highest similarity
            _, index = torch.max(similarity_matrix[i, :], dim=0)
            recovered_tokens.append(index.item())
        # Decode the token ids
        print(tokenizer.decode(recovered_tokens))
        print(iters, "%.4f" % current_loss.item())
    iters += 1
