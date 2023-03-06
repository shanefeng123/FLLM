from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import random
import torch
from sklearn.model_selection import train_test_split
from utils import *

random.seed(42)
DEVICE = torch.device("cpu")
DATA_PATH = "../data/emails.csv"
NUM_OF_CLIENTS = 30
SAMPLE_SIZE = 1
TEST_SIZE = 0.03

model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", bos_token="<|startoftext|>", eos_token="<|endoftext|>",
                                              pad_token="<|pad|>")
# Resize the token embedding of the model to match the new vocabulary size
model.resize_token_embeddings(len(tokenizer))
# Load the trained model
model.load_state_dict(
    torch.load("../model/gpt2/fed_avg_batch_30_clients_100_percent/server_2512.pt", map_location=DEVICE))
data = load_enron_email_data(DATA_PATH)
# Sample a subset to do the training to speed up the process
data = random.sample(data, int(len(data) * SAMPLE_SIZE))
# Add start and end tokens
for i in range(len(data)):
    data[i] = "<|startoftext|>" + data[i] + "<|endoftext|>"

train_data, test_data = train_test_split(data, test_size=TEST_SIZE, shuffle=True, random_state=42)
num_of_client_samples = int(len(train_data) / NUM_OF_CLIENTS)
clients_training_data = []
# Split the training data into individual client training data
for i in range(NUM_OF_CLIENTS):
    clients_training_data.append(train_data[i * num_of_client_samples:(i + 1) * num_of_client_samples])

# Take the second sentence in the first client's training data as an example
# You need to modify the generate function, include top_p, top_k, temperature, etc., to allow more randomness in the generation
# See the paper for more details
test_sentence = "Dear Dean Walker Ken Lay asked me to respond to you re: 1) Naming the new professorship the Pinkney C. Walker Distinguished Teaching Professorship this is"
generated = generate(test_sentence, model, tokenizer, DEVICE)
print(tokenizer.decode(generated[0], skip_special_tokens=True))
