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
    # Set batch size as 1 for initial experiment
    data_loader = DataLoader(MyDataset(data_inputs), batch_size=1, shuffle=True)
    train_loaders.append(data_loader)
    break
# Get the first batch of the first client, which only contains 1 sample
batch = train_loaders[0].__iter__().__next__()
batch_input_ids = [i for i in batch["input_ids"].tolist()[0] if i != 50258]
batch_input_ids.append(50258)
client = set_parameters(server, client_parameters[0])
# Train the model with this first batch only
loss, attentions = train_batch(client, batch, DEVICE)
print(loss)
# Collect all the gradients
grads = {}
for name, param in client.named_parameters():
    grads[name] = param.grad

# Get the token embedding gradients, sum up the absolute values of the gradients for each token
token_grads = np.absolute(grads['transformer.wte.weight'].clone().detach().numpy())
token_gradient_sum = np.sum(token_grads, axis=1)
# Get the token ids that have a gradient sum greater than 1
token_ids_extracted = np.where(token_gradient_sum > 1)[0].tolist()
# Verify that the token ids extracted are in the batch
token_ids_used = []
for token_id in token_ids_extracted:
    if token_id in batch_input_ids:
        token_ids_used.append(1)
    else:
        token_ids_used.append(0)

print(len(token_ids_used) == len(set(batch_input_ids)))