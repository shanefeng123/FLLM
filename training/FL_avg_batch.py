from utils import *
from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, OpenAIGPTLMHeadModel, OpenAIGPTTokenizerFast
import random

random.seed(42)

nltk.download('punkt')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE == torch.device("cpu"):
    DEVICE = torch.device("mps" if torch.has_mps else "cpu")

MODEL_NAME = "gpt2"
NUM_OF_CLIENTS = 30
SAMPLE_SIZE = 1
TEST_SIZE = 0.03
RESULTS_PATH = F"../results/{MODEL_NAME}/fed_avg_batch_{NUM_OF_CLIENTS}_clients_{int(SAMPLE_SIZE * 100)}_percent.txt"
MODEL_PATH = f"../model/{MODEL_NAME}/fed_avg_batch_{NUM_OF_CLIENTS}_clients_{int(SAMPLE_SIZE * 100)}_percent"
DATA_PATH = "../data/emails.csv"

# Initialise the server model
if MODEL_NAME == "gpt2":
    server = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME, bos_token="<|startoftext|>", eos_token="<|endoftext|>",
                                                  pad_token="<|pad|>")
else:
    server = OpenAIGPTLMHeadModel.from_pretrained("openai-gpt").to(DEVICE)
    tokenizer = OpenAIGPTTokenizerFast.from_pretrained("openai-gpt", bos_token="<|startoftext|>", eos_token="<|endoftext|>",
                                                   pad_token="<|pad|>")

# Resize the token embedding of the model to match the new vocabulary size
server.resize_token_embeddings(len(tokenizer))
server_parameters = get_parameters(server)
# Initialise client models
client_parameters = [server_parameters] * NUM_OF_CLIENTS
# Load the data
data = load_enron_email_data(DATA_PATH)
# Sample a subset to do the training to speed up the process
data = random.sample(data, int(len(data) * SAMPLE_SIZE))
# data = data[:1000]
# Append the start and end tokens
for i in range(len(data)):
    data[i] = "<|startoftext|>" + data[i] + "<|endoftext|>"
# Split the data into training and testing
train_data, test_data = train_test_split(data, test_size=TEST_SIZE, shuffle=True, random_state=42)
num_of_client_samples = int(len(train_data) / NUM_OF_CLIENTS)
clients_training_data = []
# Split the training data into individual client training data
for i in range(NUM_OF_CLIENTS):
    clients_training_data.append(train_data[i * num_of_client_samples:(i + 1) * num_of_client_samples])

train_loaders = []
for i in range(NUM_OF_CLIENTS):
    dataset = clients_training_data[i]
    # Max length 128 would leave about 3 percent of the sentences truncated
    data_inputs = tokenizer(dataset, return_tensors="pt", padding=True, truncation=True, max_length=128)
    labels = data_inputs["input_ids"].clone()
    data_inputs["labels"] = labels
    data_loader = DataLoader(MyDataset(data_inputs), batch_size=128, shuffle=True)
    train_loaders.append(data_loader)

test_data_inputs = tokenizer(test_data, return_tensors="pt", padding=True, truncation=True, max_length=128)
test_labels = test_data_inputs["input_ids"].clone()
test_data_inputs["labels"] = test_labels
test_loader = DataLoader(MyDataset(test_data_inputs), batch_size=128, shuffle=True)

i = 0
patience = 5
patience_count = 0
# This is for early stopping
min_test_loss = 100000000
min_loss_round = 0
num_of_batches = len(train_loaders[0])
while True:
    client_losses = [0] * NUM_OF_CLIENTS
    for j in range(num_of_batches):
        print(f"Round {i}:")
        client_loop = tqdm(range(NUM_OF_CLIENTS))
        for k in client_loop:
            client = set_parameters(server, client_parameters[k])
            batch = train_loaders[k].__iter__().__next__()
            batch_loss = train_batch(client, batch, device=DEVICE)
            client_losses[k] += batch_loss
            client_parameters[k] = get_parameters(client)
        # do aggregation after every batch
        aggregated_parameters = aggregate_parameters(server, client_parameters)
        server = set_parameters(server, aggregated_parameters)
        server_parameters = get_parameters(server)
        client_parameters = [server_parameters] * NUM_OF_CLIENTS
        print(f"Round {i}, server test loss:")
        average_test_loss = test(server, test_loader, device=DEVICE)
        with open(f"{RESULTS_PATH}", "a") as file:
            file.write(f"Round {i}, server test loss: {average_test_loss}\n")
        if average_test_loss > min_test_loss:
            patience_count += 1
        else:
            min_test_loss = average_test_loss
            min_loss_round = i
            patience_count = 0
        if patience_count >= patience:
            torch.save(server.state_dict(), f"{MODEL_PATH}/server_{i}.pt")
            print(f"Early stopping at round {i}")
            print(f"Minimum test loss: {min_test_loss} at round {min_loss_round}")
            with open(f"{RESULTS_PATH}", "a") as file:
                file.write(f"Early stopping at round {i}\n")
                file.write(f"Minimum test loss: {min_test_loss} at round {min_loss_round}\n")
            break
        i += 1

    for j in range(NUM_OF_CLIENTS):
        client_losses[j] = client_losses[j] / num_of_batches
        print(f"Round {i}, client {j} average loss: {client_losses[j]}")
        with open(f"{RESULTS_PATH}", "a") as file:
            file.write(f"Round {i}, client {j} average loss: {client_losses[j]}\n")
