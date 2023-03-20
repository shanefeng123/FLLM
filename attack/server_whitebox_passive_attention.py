from utils import *
from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, OpenAIGPTLMHeadModel, OpenAIGPTTokenizerFast
import random

random.seed(42)

nltk.download('punkt')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "gpt2"
NUM_OF_CLIENTS = 30
SAMPLE_SIZE = 0.01
TEST_SIZE = 0.03
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
    # Set batch size as 1 for initial experiment. Turn off shuffle for reproducibility
    data_loader = DataLoader(MyDataset(data_inputs), batch_size=1, shuffle=False)
    train_loaders.append(data_loader)
    break
# Get the first batch of the first client, which only contains 1 sample
training_batch = train_loaders[0].__iter__().__next__()
training_batch_input_ids = [i for i in training_batch["input_ids"].tolist()[0] if i != 50258]
training_batch_input_ids.append(50258)
decoded_training_sample = tokenizer.decode(training_batch_input_ids)
testing_sample = "This email informs you"
testing_batch = tokenizer(testing_sample, return_tensors="pt", padding=True, truncation=True, max_length=128)
testing_batch["labels"] = testing_batch["input_ids"].clone()
testing_batch_ids = [i for i in testing_batch["input_ids"].tolist()[0] if i != 50258]
testing_batch_ids.append(50258)
decoded_testing_sample = tokenizer.decode(testing_batch_ids)
client = set_parameters(server, client_parameters[0])

_, attention = test_batch(client, testing_batch, DEVICE)
old_attn_scores = attention[0][0, :, :, :].detach().cpu().numpy()
_, _ = train_batch(client, training_batch, DEVICE)
_, attention = test_batch(client, testing_batch, DEVICE)
new_attn_scores = attention[0][0, :, :, :].detach().cpu().numpy()
# Collect all the gradients
# grads = {}
# for name, param in client.named_parameters():
#     grads[name] = param.grad
# # Get the first attention weights
# first_attn_grads = grads["transformer.h.0.attn.c_attn.weight"]
# # Reshape to Q, K, V
# first_attn_grads = first_attn_grads.reshape(3, 768, 768)
# Q_grads = first_attn_grads[0]
# K_grads = first_attn_grads[1]
# V_grads = first_attn_grads[2]
# # Reshape to (num_heads, seq_len, head_dim)
# Q_grads = Q_grads.reshape(12, 768, 64)
# K_grads = K_grads.reshape(12, 768, 64)
# V_grads = V_grads.reshape(12, 768, 64)