from utils import *
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import random
from torch.nn.functional import one_hot
from datasets import load_dataset
from tqdm import tqdm
from collections import Counter

random.seed(4)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "gpt2"
DATASET = "linxinyuan/cola"
BATCH_SIZE = 2

# Initialise the server and client modelmodel
server = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
client = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME, bos_token="<|startoftext|>", eos_token="<|endoftext|>",
                                              pad_token="<|pad|>")
# Resize the token embedding of the model to match the new vocabulary size
server.resize_token_embeddings(len(tokenizer))
client.resize_token_embeddings(len(tokenizer))
token_embeddings = server.transformer.wte.weight
# Extract the server parameters and make sure that client has the same parameters
server_parameters = get_parameters(server)
client = set_parameters(client, server_parameters)
# Load the dataset as mentioned in the LAMP paper
data = load_dataset(DATASET, split="train")["text"]
# Randomly select 100 sentences from the dataset as client's training data
train_data = random.sample(data, 100)
# Add the start and end token
for i in range(len(train_data)):
    train_data[i] = "<|startoftext|>" + train_data[i] + "<|endoftext|>"
# Tokenize the data and add the labels
train_data = tokenizer(train_data, return_tensors="pt", padding=True, truncation=True, max_length=16)
train_data["labels"] = train_data["input_ids"].clone()
# Wrap the data in the dataloader
train_data = MyDataset(train_data)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
# Train the client model
train_loop = tqdm(train_loader, leave=True)
for training_batch in train_loop:
    loss, attentions, hidden_states, logits = train_batch(client, training_batch, DEVICE)
    # Collect the client's gradients
    client_name_grads = {}
    client_grads = []
    for param in client.parameters():
        client_grads.append(param.grad)
    for name, param in client.named_parameters():
        client_name_grads[name] = param.grad
    # Get the token embedding layer gradients
    token_grads = client_name_grads['transformer.wte.weight'].clone().detach()
    # Recover the tokens been used in the batch
    token_list = []
    token_grads_norm = torch.linalg.vector_norm(token_grads, dim=1)
    # If the magnitude of a token embedding is greater than 1, we can add it to the token list
    valid_classes = np.where(token_grads_norm.cpu().numpy() > 1)[0].tolist()
    token_list += [*valid_classes]
    # Then we use the method descried in Deception paper to estimate the token frequencies
    # Taking the log of the norm performs better
    log_token_grads_norm = torch.log(token_grads_norm)
    # Because the server would know how many tokens in a sentence, and how many sentence in a batch.
    # We can just calculate how many tokens there should be in a batch
    sequence_length = training_batch["input_ids"].shape[1]
    num_missing_tokens = BATCH_SIZE * sequence_length
    # This is the naive part. We just assume that the impact for each occurrence of a token is the same,
    # so that we can just calculate it by dividing the total magnitude of gradients by the number of missing tokens.
    # TODO: There is a smarter way to estimate the frequency of tokens because every sentence can only have one start
    #  token and one end token
    m_impact = log_token_grads_norm[valid_classes].sum() / num_missing_tokens
    # Subtract the impact from the gradient magnitude once as they are already in token list
    log_token_grads_norm[valid_classes] = log_token_grads_norm[valid_classes] - m_impact
    # Get the token that has the largest magnitude, add it to the list, then subtract it with the impact value
    # Do this until we reach the number of missing tokens
    while len(token_list) < num_missing_tokens:
        selected_idx = valid_classes[log_token_grads_norm[valid_classes].argmax()]
        token_list.append(selected_idx)
        log_token_grads_norm[selected_idx] -= m_impact
    # Transform to token set, token frequency and token probability
    token_set = set(token_list)
    token_freq = Counter(token_list)
    token_prob_dict = {token: freq / len(token_list) for token, freq in token_freq.items()}
    # Now that we have token set, token frequency and token probability, we can initialise the GA algorithm
    GA_optimizer = GA_LLM_optimizer(token_prob_dict, token_embeddings, sequence_length, BATCH_SIZE,
                                    client_grads)
    # Testing
    GA_optimizer.initialise_population()
    GA_optimizer.calculate_objective(server, tokenizer, device=DEVICE)
    GA_optimizer._normalise_objective_values()
    parents = GA_optimizer._select_parents()
    child_1, child_2 = GA_optimizer._crossover(parents[0], parents[1])
    GA_optimizer._mutate(child_1)
    break
#
# # dummy_input_embedding = construct_input_embeddings(batch["input_ids"].tolist()[0], token_embeddings).to(
# #     DEVICE).requires_grad_(True)
# dummy_input = tokenizer(["<|startoftext|>Hello how go go<|endoftext|>"],
#                         return_tensors="pt", padding=True,
#                         truncation=True, max_length=8)
# dummy_input = dummy_input["input_ids"].tolist()[0]
# dummy_input_embedding = construct_input_embeddings(dummy_input, token_embeddings).to(
#     DEVICE).requires_grad_(True)
# dummy_labels = torch.tensor(one_hot(torch.tensor(dummy_input), num_classes=len(tokenizer)),
#                             dtype=torch.float32).to(DEVICE).requires_grad_(True)
# # dummy_labels = torch.randn((batch["input_ids"].size()[0], batch["input_ids"].size()[1], client.config.vocab_size)).to(
# #     DEVICE).requires_grad_(True)
#
# iters = 0
# while True:
#     dummy_outputs = server(inputs_embeds=dummy_input_embedding.to(DEVICE))
#     dummy_preds = dummy_outputs.logits
#     loss_function = torch.nn.CrossEntropyLoss()
#     shifted_preds = dummy_preds[..., :-1, :].contiguous()
#     shifted_labels = dummy_labels[..., 1:, :].contiguous()
#     flatten_shifted_preds = shifted_preds.view(-1, shifted_preds.size(-1)).to(DEVICE)
#     flatten_labels = shifted_labels.view(-1, shifted_labels.size(-1)).to(DEVICE)
#     server_loss = loss_function(flatten_shifted_preds, flatten_labels)
#     server_grads = torch.autograd.grad(server_loss, server.parameters(), create_graph=True)
#     # server_grads = server_grads[1:]
#     #
#     optimizer = torch.optim.AdamW([dummy_input_embedding], lr=0.01)
#     optimizer.zero_grad()
#     grads_diff = 0
#     alpha = 0.8
#     for gx, gy in zip(server_grads, client_grads):
#         grads_diff += torch.norm(gx - gy, p=2) + alpha * torch.norm(gx - gy, p=1)
#
#     grads_diff.backward()
#     optimizer.step()
#     # Copy the dummy embeddings
#     dummy_input_embedding_copy = dummy_input_embedding[0, :, :].detach().clone()
#     # Calculate the similarity between each row of the dummy embeddings and each row of the token embeddings
#     recovered_tokens = []
#     similarity_matrix = torch.nn.functional.cosine_similarity(dummy_input_embedding_copy.unsqueeze(1),
#                                                               token_embeddings.unsqueeze(0), dim=-1)
#     for i in range(similarity_matrix.size()[0]):
#         # Find the index of the highest similarity
#         _, index = torch.max(similarity_matrix[i, :], dim=0)
#         recovered_tokens.append(index.item())
#     dummy_labels = torch.tensor(one_hot(torch.tensor(recovered_tokens), num_classes=len(tokenizer)),
#                                 dtype=torch.float32).to(DEVICE).requires_grad_(True)
#     if iters % 10 == 0:
#         # Decode the token ids
#         print(tokenizer.decode(recovered_tokens))
#         print(iters, "%.4f" % grads_diff)
#     iters += 1
