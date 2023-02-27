from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import nltk


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])


def load_enron_email_data(data_path):
    """
    Load entron email dataset and preprocess it.
    Returns:
    List of sentences from the message body of the emails.
    """
    email_data = pd.read_csv(data_path)
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
            email_lines[j] = email_lines[j].replace(">", "")
            email_lines[j] = email_lines[j].replace("<", "")
            email_lines[j] = email_lines[j].replace("=", "")
            email_lines[j] = email_lines[j].replace("-", "")
            email_lines[j] = email_lines[j].replace("_", "")
            email_lines[j] = email_lines[j].replace("*", "")
            email_lines[j] = email_lines[j].replace("~", "")
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


def train_epoch(model, train_loader, device):
    """
    Train a client model on local data for a complete epoch.
    Args:
        model: The model to be trained
        train_loader: The training data loader
        device: The device to run the model on

    Returns:
        The average training loss
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()
    train_loop = tqdm(train_loader, leave=False)
    epoch_train_loss = 0
    for batch in train_loop:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        train_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        train_loss = train_outputs.loss
        train_loss.backward()
        optimizer.step()
        epoch_train_loss += train_loss.item()
        train_loop.set_description(f"Training loss: {train_loss.item()}")
    average_epoch_loss = epoch_train_loss / len(train_loop)
    print(f"Epoch average training loss: {average_epoch_loss}")
    return average_epoch_loss


def train_batch(model, batch, device):
    """
    Train a client model on local data for a single batch.
    Args:
        model: The model to be trained
        batch: The training batch
        device: The device to run the model on

    Returns:
        The batch training loss
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()
    optimizer.zero_grad()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    train_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    batch_loss = train_outputs.loss
    batch_loss.backward()
    optimizer.step()
    return batch_loss.item()


def test(model, test_loader, device):
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
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            test_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            test_loss = test_outputs.loss
            epoch_test_loss += test_loss.item()
            test_loop.set_description(f"Test loss: {test_loss.item()}")
        average_epoch_loss = epoch_test_loss / len(test_loop)
        print(f"Epoch Average test loss: {average_epoch_loss}")
    return average_epoch_loss


def aggregate_parameters(server, client_parameters):
    aggregated_parameters = []
    for param in get_parameters(server):
        aggregated_parameters.append(torch.zeros(param.shape))

    for j in range(len(client_parameters)):
        single_client_parameter = client_parameters[j]
        for k, param in enumerate(single_client_parameter):
            aggregated_parameters[k] += torch.Tensor(param)

    for j in range(len(aggregated_parameters)):
        aggregated_parameters[j] /= len(client_parameters)
    return aggregated_parameters
