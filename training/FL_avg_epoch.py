from utils import *
from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, OpenAIGPTLMHeadModel, OpenAIGPTTokenizerFast
import random

random.seed(42)

nltk.download('punkt')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "gpt"
NUM_OF_CLIENTS = 30
SAMPLE_SIZE = 1
TEST_SIZE = 0.1
RESULTS_PATH = F"../results/{MODEL_NAME}/fed_avg_{NUM_OF_CLIENTS}_clients_{int(SAMPLE_SIZE * 100)}_percent.txt"
MODEL_PATH = f"../model/{MODEL_NAME}/fed_avg_{NUM_OF_CLIENTS}_clients_{int(SAMPLE_SIZE * 100)}_percent"
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
client_parameters = initialise_client_parameters(server_parameters, NUM_OF_CLIENTS)
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

torch.save(server.state_dict(), f"{MODEL_PATH}/server_init.pt")

i = 0
patience = 5
patience_count = 0
# This is for early stopping
min_test_loss = 100000000
min_loss_round = 0
while True:
    print(f"Round {i}:")
    for j in range(NUM_OF_CLIENTS):
        client = set_parameters(server, client_parameters[j])
        print(f"Round {i}, client {j}:")
        average_train_loss = train_epoch(client, train_loader=train_loaders[j], device=DEVICE)
        with open(f"{RESULTS_PATH}", "a") as file:
            file.write(f"Round {i}, client {j} loss: {average_train_loss}\n")
        client_parameters[j] = get_parameters(client)

    # Aggregate the parameters
    aggregated_parameters = aggregate_parameters(server, client_parameters)

    server = set_parameters(server, aggregated_parameters)
    server_parameters = get_parameters(server)
    print(f"Round {i} server test:")
    average_test_loss = test(server, test_loader, device=DEVICE)
    with open(f"{RESULTS_PATH}", "a") as file:
        file.write(f"Round {i} server test loss: {average_test_loss}\n")
    client_parameters = initialise_client_parameters(server_parameters, NUM_OF_CLIENTS)
    torch.save(server.state_dict(), f"{MODEL_PATH}/server_{i}.pt")
    if average_test_loss > min_test_loss:
        patience_count += 1
    else:
        patience_count = 0
        min_test_loss = average_test_loss
        min_loss_round = i

    if patience_count == patience:
        print(f"Early stopping at round {i}")
        print(f"Minimum test loss: {min_test_loss} at round {min_loss_round}")
        with open(f"{RESULTS_PATH}", "a") as file:
            file.write(f"Early stopping at round {i}\n")
            file.write(f"Minimum test loss: {min_test_loss} at round {min_loss_round}\n")
        break
    i += 1
