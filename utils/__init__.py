from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import nltk
import numpy as np
import numpy
import random
from torch.nn.functional import one_hot
import copy


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
        model: A neural network model with parameters.

    Returns:
        A deep copy list of parameters of the model.
    """
    return [val.clone().detach().cpu().numpy() for _, val in model.state_dict().items()]


def initialise_client_parameters(server_parameters, num_of_clients):
    """
    Initialise the parameters of the clients.
    Args:
        server_parameters: The parameters of the server model.
        num_of_clients: The number of clients.

    Returns:
        A list of parameters for the clients.
    """
    client_parameters = []
    for i in range(num_of_clients):
        client_parameters.append([np.copy(p) for p in server_parameters])
    return client_parameters


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
        The batch training loss and the attention scores
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()
    optimizer.zero_grad()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    train_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_attentions=True,
                          output_hidden_states=True)
    attentions = train_outputs.attentions
    batch_loss = train_outputs.loss
    batch_loss.backward()
    optimizer.step()
    return batch_loss.item(), attentions, train_outputs.hidden_states, train_outputs.logits


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


def test_batch(model, batch, device):
    """
    Test the model with batch data.
    Args:
        model: The model to be tested
        batch: The batch data
        device: The device to run the model on

    Returns:
        The batch testing loss and the attention scores
    """
    model.eval()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    test_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                         output_attentions=True)
    test_loss = test_outputs.loss
    return test_loss.item(), test_outputs.attentions


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


def cosine_similarity(vector_a, vector_b):
    """
    Calculate the cosine similarity between two vectors.
    Args:
        vector_a: A vector.
        vector_b: Another vector.

    Returns:
        The cosine similarity between the two vectors.
    """
    return np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))


def generate(sentence, model, tokenizer, device):
    data = f"<|startoftext|>{sentence}"
    input = tokenizer(data, return_tensors="pt")
    input_ids = input["input_ids"].to(device)
    attention_mask = input["attention_mask"].to(device)
    generated = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=128,
                               pad_token_id=tokenizer.eos_token_id)
    return generated


def construct_input_embeddings(token_list, token_embeddings):
    """
    Construct the input embeddings for the model given the token list.
    Args:
        token_list: A list of tokens.
        token_embeddings: The token embeddings.
        batch_size: The batch size.

    Returns:
        The input embeddings.
    """
    input_embeddings = []
    for token in token_list:
        if input_embeddings == []:
            input_embeddings = torch.reshape(token_embeddings[token].clone().detach(),
                                             (1, token_embeddings[token].shape[0]))
        else:
            input_embeddings = torch.cat(
                (input_embeddings,
                 torch.reshape(token_embeddings[token].clone().detach(), (1, token_embeddings[token].shape[0]))),
                dim=0)
    input_embeddings = torch.reshape(input_embeddings,
                                     (1, input_embeddings.size()[0], input_embeddings.size()[1]))
    return input_embeddings


class GA_LLM_optimizer():
    def __init__(self, token_prob_dict, token_embedding, seq_len,
                 batch_size, client_grads, num_of_iterations=500, num_of_chromosomes=400, prob_cross=0.85,
                 prob_mutate=0.5,
                 elitism_rate=0.2, crossover="double"):
        self.num_of_iterations = num_of_iterations
        self.num_of_chromosomes = num_of_chromosomes
        self.prob_cross = prob_cross
        self.prob_mutate = prob_mutate
        self.elitism_rate = elitism_rate
        self.crossover = crossover
        self.token_prob_dict = token_prob_dict
        self.token_set = list(token_prob_dict.keys())
        self.token_probs = list(token_prob_dict.values())
        self.cum_token_probs = np.cumsum(self.token_probs)
        self.token_embedding = token_embedding
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.population = None
        self.new_population = None
        self.objective_values = None
        self.norm_obj_values = None
        self.client_grads = client_grads

    def initialise_population(self):
        # TODO: There is a smarter way to initialise the population based on the observation that the first token is
        #  always the start token and the last token should either be end token or pad token
        self.population = []
        for i in range(self.num_of_chromosomes):
            chromosome = []
            for j in range(self.batch_size):
                sequence = [50257]
                sequence.extend(np.random.choice(a=np.array(self.token_set), size=self.seq_len - 1,
                                                 p=np.array(self.token_probs).tolist()))
                chromosome.append(sequence)
            self.population.append(chromosome)

    def calculate_objective(self, model, tokenizer, device):
        self.objective_values = []
        for chromosome in self.population:
            dummy_x = torch.tensor(chromosome).to(device)
            dummy_y = torch.tensor(one_hot(dummy_x, num_classes=len(tokenizer)),
                                   dtype=torch.float32).to(device)
            dummy_outputs = model(input_ids=dummy_x)
            dummy_preds = dummy_outputs.logits
            loss_function = torch.nn.CrossEntropyLoss()
            shifted_preds = dummy_preds[..., :-1, :].contiguous()
            shifted_labels = dummy_y[..., 1:, :].contiguous()
            flatten_shifted_preds = shifted_preds.view(-1, shifted_preds.size(-1)).to(device)
            flatten_labels = shifted_labels.view(-1, shifted_labels.size(-1)).to(device)
            dummy_loss = loss_function(flatten_shifted_preds, flatten_labels)
            server_grads = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
            grads_diff = 0
            alpha = 0.8
            for gx, gy in zip(server_grads, self.client_grads):
                grads_diff += torch.norm(gx - gy, p=2) + alpha * torch.norm(gx - gy, p=1)
            # Add the negative value such that bigger objective value means a better solution
            self.objective_values.append(-grads_diff.item())

    def _normalise_objective_values(self):
        min_value = min(self.objective_values)
        max_value = max(self.objective_values)
        norm_values = (np.array(self.objective_values) - min_value) / (max_value - min_value)
        norm_values = norm_values / np.sum(norm_values)
        self.norm_obj_values = norm_values

    def _roulette_wheel(self):
        """Calculate the cumulative sum of the objective values and output the selected index based on probability
        Args:
            normalised_objective_values (numpy.ndarray): Normalised objective values of the population, which are all
                between 0 and 1.
        Returns:
            Selected index.
        """
        cum_sum = np.cumsum(self.norm_obj_values)
        r = random.random()
        for index, condition in enumerate(r <= cum_sum):
            if condition:
                return index

    def _select_parents(self):
        """Use the roulette_wheel method to select 2 solutions from the population as parents. The selected parents are
        passed in to crossover method.
        This is a pure function of its inputs - it does not read or update any self fields.

        Returns:
            The selected solutions as parents.
        """
        selected_parents_indexes = []
        selected_parents = []
        for i in range(2):
            index = self._roulette_wheel()
            while index in selected_parents_indexes:
                index = self._roulette_wheel()
            selected_parents_indexes.append(index)
            selected_parents.append(copy.deepcopy(self.population[index]))
        return selected_parents

    def _crossover(self, parent1, parent2):
        """Exchange a subset of the two parents and produce new solutions. With single point crossover, randomly pick a
        point in the solution and exchange the other part. With double point crossover, randomly pick two points in the
        solution and exchange two different parts. After crossover, use probability to decide whether to keep the
        changed solution or not.
        This never mutates the input parent1 or parent2.  It returns fresh child arrays, not aliased
        with the parent arrays.
        Args:
            parent1 (numpy.ndarray): A solution used to crossover.
            parent2 (numpy.ndarray): A solution used to crossover.
        Returns:
            Either the changed solutions or a copy of the original solutions based on probability.
        """
        child1 = []
        child2 = []
        if self.crossover == "single":
            crossover_point = random.randint(1, self.seq_len - 2)
            for i in range(len(parent1)):
                child1.append(
                    np.concatenate([parent1[i][0:crossover_point], parent2[i][crossover_point: self.seq_len]]).tolist())
                child2.append(
                    np.concatenate([parent2[i][0:crossover_point], parent1[i][crossover_point: self.seq_len]]).tolist())

        elif self.crossover == "double":
            crossover_point1 = random.randint(1, self.seq_len - 2)
            crossover_point2 = random.randint(1, self.seq_len - 2)
            while crossover_point1 == crossover_point2:
                crossover_point2 = random.randint(1, self.seq_len - 2)
            if crossover_point1 > crossover_point2:
                temp = crossover_point1
                crossover_point1 = crossover_point2
                crossover_point2 = temp
            for i in range(len(parent1)):
                child1.append(
                    np.concatenate([parent1[i][0:crossover_point1], parent2[i][crossover_point1: crossover_point2],
                                    parent1[i][crossover_point2:self.seq_len]]).tolist())
                child2.append(
                    np.concatenate([parent2[i][0:crossover_point1], parent1[i][crossover_point1: crossover_point2],
                                    parent2[i][crossover_point2:self.seq_len]]).tolist())

        r1 = random.random()
        child1 = child1 if r1 <= self.prob_cross else copy.deepcopy(parent1)  # copy, because we may mutate child
        r2 = random.random()
        child2 = child2 if r2 <= self.prob_cross else copy.deepcopy(parent2)
        return child1, child2

    def _mutate(self, child):
        for i in range(len(child)):
            sequence = child[i]
            for j in range(self.seq_len):
                r = random.random()
                if r <= self.prob_mutate:
                    r_mutate = random.random()
                    for index, condition in enumerate(r_mutate <= self.cum_token_probs):
                        if condition:
                            sequence[j] = self.token_set[index]
                            break

    def _add_elite(self, new_population, new_population_obj_values):
        num_of_elites = int(self.num_of_chromosomes * self.elitism_rate)
        old_obj_values = self.objective_values.copy()
        for i in range(num_of_elites):
            old_max_index = np.argmax(old_obj_values)
            elite = copy.deepcopy(self.population[old_max_index])
            old_max_value = old_obj_values[old_max_index]
            new_min_index = np.argmin(new_population_obj_values)
            new_min_value = new_population_obj_values[new_min_index]
            if new_min_value < old_max_value:
                new_population[new_min_index] = elite
                new_population_obj_values[new_min_index] = old_max_value
                old_obj_values[old_max_index] = -1000
            else:
                break
        return new_population, new_population_obj_values


