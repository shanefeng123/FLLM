import nltk
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import random
from collections import OrderedDict
import torch
from sklearn.model_selection import train_test_split

random.seed(42)
DEVICE = torch.device("cpu")


def load_enron_email_data():
    """
    Load entron email dataset and preprocess it.
    Returns:
    List of sentences from the message body of the emails.
    """
    email_data = pd.read_csv("../../../data/emails.csv")
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
            email_lines[j] = email_lines[j].replace("=", "")
            email_lines[j] = email_lines[j].replace("-", "")
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

def generate(sentence, model, tokenizer):
    data = f"<|startoftext|>{sentence}"
    input = tokenizer(data, return_tensors="pt")
    input_ids = input["input_ids"].to(DEVICE)
    attention_mask = input["attention_mask"].to(DEVICE)
    generated = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128,
                               pad_token_id=tokenizer.eos_token_id)
    return generated


model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", bos_token="<|startoftext|>", eos_token="<|endoftext|>",
                                              pad_token="<|pad|>")
# Resize the token embedding of the model to match the new vocabulary size
model.resize_token_embeddings(len(tokenizer))
# model.load_state_dict(torch.load("../../../model/gpt2/fed_avg_30_clients_10_percent/server_88.pt", map_location=DEVICE))
num_of_clients = 30
data = load_enron_email_data()
# Sample a subset to do the training to speed up the process
sample_size = 0.1
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

test_sentence = 'dave.mabell@lethbridge'
generated = generate(test_sentence, model, tokenizer)
print(tokenizer.decode(generated[0], skip_special_tokens=True))