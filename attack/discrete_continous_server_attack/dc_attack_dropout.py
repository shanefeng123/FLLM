from utils import *
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, OpenAIGPTLMHeadModel, OpenAIGPTTokenizerFast
import random
from torch.nn.functional import softmax, one_hot

random.seed(42)

nltk.download('punkt')
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_NAME = "gpt2"
NUM_OF_CLIENTS = 30

# Initialise the server model
if MODEL_NAME == "gpt2":
    server = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
    client = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME, bos_token="<|startoftext|>", eos_token="<|endoftext|>",
                                                  pad_token="<|pad|>")
else:
    server = OpenAIGPTLMHeadModel.from_pretrained("openai-gpt").to(DEVICE)
    client = OpenAIGPTLMHeadModel.from_pretrained("openai-gpt").to(DEVICE)
    tokenizer = OpenAIGPTTokenizerFast.from_pretrained("openai-gpt", bos_token="<|startoftext|>",
                                                       eos_token="<|endoftext|>",
                                                       pad_token="<|pad|>")

# Resize the token embedding of the model to match the new vocabulary size
server.resize_token_embeddings(len(tokenizer))
client.resize_token_embeddings(len(tokenizer))
token_embeddings = server.transformer.wte.weight
server_parameters = get_parameters(server)
client = set_parameters(client, server_parameters)
# Load the data
train_data = tokenizer(
    "<|startoftext|>Hello how are you<|endoftext|>",
    return_tensors="pt", padding=True,
    truncation=True, max_length=8)
batch = train_data
batch["labels"] = batch["input_ids"].clone()
# Train the model with this first batch only
loss, attentions, hidden_states, logits = train_batch(client, batch, DEVICE)
# # Collect all the gradients
client_grads = []
for param in client.parameters():
    client_grads.append(param.grad)

dummy_input_embedding = torch.randn(
    (batch["input_ids"].size()[0], batch["input_ids"].size()[1], client.config.n_embd)).to(DEVICE).requires_grad_(True)
dummy_labels = torch.randn((batch["input_ids"].size()[0], batch["input_ids"].size()[1], client.config.vocab_size)).to(
    DEVICE).requires_grad_(True)

iters = 0
while True:
    dummy_outputs = server(inputs_embeds=dummy_input_embedding.to(DEVICE))
    dummy_preds = dummy_outputs.logits
    loss_function = torch.nn.CrossEntropyLoss()
    shifted_preds = dummy_preds[..., :-1, :].contiguous()
    shifted_labels = dummy_labels[..., 1:, :].contiguous()
    flatten_shifted_preds = shifted_preds.view(-1, shifted_preds.size(-1)).to(DEVICE)
    flatten_labels = shifted_labels.view(-1, shifted_labels.size(-1)).to(DEVICE)
    server_loss = loss_function(flatten_shifted_preds, flatten_labels)
    server_grads = torch.autograd.grad(server_loss, server.parameters(), create_graph=True)
    # server_grads = server_grads[1:]
    #
    optimizer = torch.optim.AdamW([dummy_input_embedding], lr=0.01)
    optimizer.zero_grad()
    grads_diff = 0
    alpha = 0.8
    for gx, gy in zip(server_grads, client_grads):
        grads_diff += torch.norm(gx - gy, p=2) + alpha * torch.norm(gx - gy, p=1)

    grads_diff.backward()
    optimizer.step()
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
    dummy_labels = torch.tensor(one_hot(torch.tensor(recovered_tokens), num_classes=len(tokenizer)),
                                dtype=torch.float32).to(DEVICE).requires_grad_(True)
    if iters % 10 == 0:
        # Decode the token ids
        print(tokenizer.decode(recovered_tokens))
        print(iters, "%.4f" % grads_diff)
    iters += 1
