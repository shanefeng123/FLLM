import numpy as np

from utils import *
from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, OpenAIGPTLMHeadModel, OpenAIGPTTokenizerFast
import random

random.seed(42)

nltk.download('punkt')
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "gpt2"
NUM_OF_CLIENTS = 30
SAMPLE_SIZE = 0.01
TEST_SIZE = 0.03
# RESULTS_PATH = F"../results/{MODEL_NAME}/fed_avg_batch_{NUM_OF_CLIENTS}_clients_{int(SAMPLE_SIZE * 100)}_percent.txt"
# MODEL_PATH = f"../model/{MODEL_NAME}/fed_avg_batch_{NUM_OF_CLIENTS}_clients_{int(SAMPLE_SIZE * 100)}_percent"
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
    data_loader = DataLoader(MyDataset(data_inputs), batch_size=1, shuffle=True)
    train_loaders.append(data_loader)
    break

batch = train_loaders[0].__iter__().__next__()
batch_input_ids = batch["input_ids"].tolist()
old_embeddings = client_parameters[0][0]
client = set_parameters(server, client_parameters[0])
loss, attentions = train_batch(client, batch, DEVICE)
print(loss)
grads = {}
for name, param in client.named_parameters():
    grads[name] = param.grad
# updated_embeddings = get_parameters(client)[0]
# gradients = updated_embeddings - old_embeddings
# gradients = np.absolute(gradients)

# token_grads = np.absolute(grads[0].clone().detach().numpy())
# token_gradient_sum = np.sum(token_grads, axis=1)
# token_gradient_indexes = np.argsort(token_gradient_sum)[::-1]