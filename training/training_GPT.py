from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizerFast
import random

random.seed(42)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])


nltk.download('punkt')
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
if DEVICE == torch.device("cpu"):
    DEVICE = torch.device("mps" if torch.has_mps else "cpu")


def load_enron_email_data():
    """
    Load entron email dataset and preprocess it.
    Returns:
    List of sentences from the message body of the emails.
    """
    email_data = pd.read_csv("../data/emails.csv")
    num_of_rows_original = email_data.shape[0]
    message_body_data = []
    # Determine the place to split the message body out of the whole email.
    for i in range(num_of_rows_original):
        email_str = email_data["message"][i]
        email_lines = email_str.split("\n")
        split_index = 0
        for j in range(len(email_lines)):
            if email_lines[j].startswith("X-FileName"):
                split_index = j
            if email_lines[j].startswith("Subject:"):
                split_index = j
        processed_email_str = " ".join(" ".join(email_lines[split_index + 1:]).split())
        if processed_email_str != "":
            message_body_data.append(processed_email_str)

    # Split the message body into sentences.
    message_sentences = []
    for i in range(len(message_body_data)):
        sentences = nltk.sent_tokenize(message_body_data[i])
        message_sentences.extend(sentences)
    message_sentences = list(dict.fromkeys(message_sentences))

    return message_sentences


def get_parameters(model):
    """
    Get the parameters of a model.
    Args:
        model: A neural network models with parameters.

    Returns:
        A list of parameters of the model.
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    """
    Set the parameters of a model.
    Args:
        model: A neural network models with parameters.
        parameters: A list of parameters for the model.

    Returns:
        The model with the new parameters.
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model


def train(model, train_loader, epochs: int):
    """
    Train a client model on local data.
    Args:
        model: The model to be trained
        train_loader: The training data loader
        epochs: the number of training epochs

    Returns:
        The average training loss
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    average_epoch_loss = 0
    for i in range(epochs):
        model.train()
        train_loop = tqdm(train_loader, leave=False)
        epoch_train_loss = 0
        for batch in train_loop:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            train_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            train_loss = train_outputs.loss
            train_loss.backward()
            optimizer.step()
            epoch_train_loss += train_loss.item()
            train_loop.set_description(f"Training loss: {train_loss.item()}")
        average_epoch_loss = epoch_train_loss / len(train_loop)
        print(f"Epoch average training loss: {average_epoch_loss}")
    return average_epoch_loss


def test(model, test_loader):
    """
    Test the server model after aggregation.
    Args:
        model: The server model to be tested
        test_loader: The testing data loader

    Returns:
        The average testing loss
    """
    model.eval()
    test_loop = tqdm(test_loader, leave=False)
    epoch_test_loss = 0
    with torch.no_grad():
        for batch in test_loop:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            test_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            test_loss = test_outputs.loss
            epoch_test_loss += test_loss.item()
            test_loop.set_description(f"Test loss: {test_loss.item()}")
        average_epoch_loss = epoch_test_loss / len(test_loop)
        print(f"Epoch Average test loss: {average_epoch_loss}")
    return average_epoch_loss


# Initialise the server model
server = OpenAIGPTLMHeadModel.from_pretrained("openai-gpt").to(DEVICE)
tokenizer = OpenAIGPTTokenizerFast.from_pretrained("openai-gpt", bos_token="<|startoftext|>", eos_token="<|endoftext|>",
                                          pad_token="<|pad|>")
# Resize the token embedding of the model to match the new vocabulary size
server.resize_token_embeddings(len(tokenizer))
server_parameters = get_parameters(server)
# Initialise client models
num_of_clients = 30
client_parameters = [server_parameters] * num_of_clients

data = load_enron_email_data()
# Sample a subset to do the training to speed up the process
sample_size = 0.2
data = random.sample(data, int(len(data) * sample_size))
# data = data[:1000]
for i in range(len(data)):
    data[i] = "<|startoftext|>" + data[i] + "<|endoftext|>"



test_size = 0.1
train_data, test_data = train_test_split(data, test_size=test_size, shuffle=True, random_state=42)
num_of_client_samples = int(len(train_data) / num_of_clients)
clients_training_data = []
# Split the training data into individual client training data
for i in range(num_of_clients):
    clients_training_data.append(train_data[i * num_of_client_samples:(i + 1) * num_of_client_samples])

train_loaders = []
for i in range(num_of_clients):
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

torch.save(server.state_dict(), r"../model/gpt/fed_avg_30_clients_20_percent/server_init.pt")
#
i = 0
# This is for early stopping
previous_average_test_loss = 100000000
while True:
    print(f"Round {i}:")
    for j in range(num_of_clients):
        client = set_parameters(server, client_parameters[j])
        print(f"Round {i}, client {j}:")
        average_train_loss = train(client, train_loader=train_loaders[j], epochs=1)
        with open(r"../results/gpt/fed_avg_30_clients_20_percent.txt", "a") as file:
            file.write(f"Round {i}, client {j} loss: {average_train_loss}\n")
        client_parameters[j] = get_parameters(client)

    # Aggregate the parameters
    aggregated_parameters = []
    for param in get_parameters(server):
        aggregated_parameters.append(torch.zeros(param.shape))
    for j in range(num_of_clients):
        single_client_parameter = client_parameters[j]
        for k, param in enumerate(single_client_parameter):
            aggregated_parameters[k] += torch.Tensor(param)

    for j in range(len(aggregated_parameters)):
        aggregated_parameters[j] /= num_of_clients

    server = set_parameters(server, aggregated_parameters)
    server_parameters = get_parameters(server)
    print(f"Round {i} server test:")
    average_test_loss = test(server, test_loader)
    with open(r"../results/gpt/fed_avg_30_clients_20_percent.txt", "a") as file:
        file.write(f"Round {i} server test loss: {average_test_loss}\n")
    client_parameters = [server_parameters] * num_of_clients
    torch.save(server.state_dict(), f"../model/gpt/fed_avg_30_clients_20_percent/server_{i}.pt")
    if average_test_loss > previous_average_test_loss:
        print("Early stopping")
        break
    previous_average_test_loss = average_test_loss
    i += 1
