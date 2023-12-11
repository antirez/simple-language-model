import torch
import torch.nn as nn
from torch.nn import functional as F
import random, math, sys, os, time

# Hyperparameters
batch_size = 64 # How many independent sequences will we process in parallel?
block_size = 64 # What is the maximum context length for predictions?
max_hidden_nodes = 2048 # Wider (first) hidden layer size of the funnel
max_iters = 100000000
eval_interval = 500
learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200 # To estimate the loss. Higher = more precise but slower.
char_encoding_len = 12 # Number of inputs for each character. Must be even.
use_batch_norm = False
dropout_rate=0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
filename = 'input.txt' if len(sys.argv) < 2 else sys.argv[1]
with open(filename, 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("Vocabulary size is", vocab_size)
print("Encoding len", char_encoding_len)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = []
    for i in ix:
        block = data[i:i+block_size]
        char_tensors = [encoded_patterns[idx] for idx in block]
        char_tensors = torch.stack(char_tensors).view(-1)
        x.append(char_tensors)
    x = torch.stack(x).to(device)
    y = torch.stack([data[i+block_size:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    
    # For 1/4 of our batches, set the first N random elements in 'x' to
    # zero, so that the network learn how to start a sequence from
    # an incomplete prompt.
    num_batches_to_modify = batch_size // 4
    for batch_index in range(num_batches_to_modify):
        N = random.randint(1, block_size)
        x[batch_index, :N] = 0

    return x, y

# We don't want to use one-hot encoding since it's too sparse, nor binary
# coding that may introduce biases. So we encode each input character
# into char_encoding_len inputs having different patterns of 1 and 0, so that
# each pattern is always composed of 50% ones and 50% zeros.
#
# For example if we have 14 inputs per symbol, we can rapresent:
#
#   14! / (6!*6!) = 3432 total symbols
#
# We call this permutation coding.
def gen_coding_patterns():
    # Calculate if there are enough permutations of char_encoding_len
    # length bits 10101010 pattern to actually represent vocab_size
    # symbols. Otherwise this function would run forever...
    permutations = math.factorial(char_encoding_len) / \
                    (math.factorial(char_encoding_len//2)* \
                     math.factorial(char_encoding_len//2))
    if permutations < vocab_size:
        print("Insufficient 'char_encoding_len' value for vocabulary size.")
        exit(1)

    # We want the result of this function to be stable, so let's
    # create a PRNG with a seed which is always the same.
    r = random.Random()
    r.seed(1234)

    pattern = "01"*(char_encoding_len//2)
    patterns = {}
    while len(patterns) != vocab_size:
        pattern = list(pattern)
        r.shuffle(pattern)
        pattern = "".join(pattern)
        patterns[pattern] = True
    string_lists = list(patterns)
    int_lists = [[int(char) for char in string] for string in string_lists]
    tensors = torch.tensor(int_lists, device=device, dtype=torch.float)
    return tensors

# Function to convert tensor of indexes to permutation patterns that
# we give as input to our neural network.
def encode_chars(tensor):
    encoded = torch.zeros(*tensor.shape, char_encoding_len).to(device)
    # Iterating over each element in the input tensor to set the
    # corresponding position in the one-hot tensor
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            encoded[i, j] = encoded_patterns[tensor[i,j]]
    encoded = encoded.to(device)
    return encoded

# Compute the current performance of the NN, against both
# training and evaluation datasets.
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Toy language model. N chars in input -> next char prediction.
class SimpleLanguageModel(nn.Module):
    def __init__(self):
        hidden_nodes = max_hidden_nodes
        super().__init__()

        input_size = char_encoding_len * block_size
        self.fc1 = nn.Linear(input_size, hidden_nodes)
        if use_batch_norm: self.bn1 = nn.BatchNorm1d(hidden_nodes)
        self.do1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_nodes, hidden_nodes // 2)
        if use_batch_norm: self.bn2 = nn.BatchNorm1d(hidden_nodes // 2)
        self.do2 = nn.Dropout(dropout_rate)

        hidden_nodes //= 2

        self.fc3 = nn.Linear(hidden_nodes, hidden_nodes)
        if use_batch_norm: self.bn3 = nn.BatchNorm1d(hidden_nodes)
        self.do3 = nn.Dropout(dropout_rate)

        self.fc4 = nn.Linear(hidden_nodes, hidden_nodes // 2)
        if use_batch_norm: self.bn4 = nn.BatchNorm1d(hidden_nodes // 2)
        self.do4 = nn.Dropout(dropout_rate)

        hidden_nodes //= 2

        self.fc5 = nn.Linear(hidden_nodes, vocab_size)

    def forward(self, inp, targets=None):
        x = self.fc1(inp)
        if use_batch_norm: x = self.bn1(x)
        x = F.relu(x)
        x = self.do1(x)

        x = self.fc2(x)
        if use_batch_norm: x = self.bn2(x)
        x = F.relu(x)
        x = self.do2(x)

        x = self.fc3(x)
        if use_batch_norm: x = self.bn3(x)
        x = F.relu(x)
        x = self.do3(x)

        x = self.fc4(x)
        if use_batch_norm: x = self.bn4(x)
        x = F.relu(x)
        x = self.do4(x)

        x = self.fc5(x)
        logits = x

        if targets is None:
            loss = None
        else:
            loss = F.cross_entropy(logits, targets.view(-1))

        return logits, loss

    def generate(self, ctx, max_new_tokens):
        output = []
        self.eval() # Otherwise batch normalization will raise an error.
        for _ in range(max_new_tokens):
            # crop context to the last block_size tokens
            idx_cond = ctx[:, -block_size*char_encoding_len:]
            # get the predictions
            logits, loss = self(idx_cond)
            # apply softmax to get probabilities
            probs = F.softmax(logits,dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            output.append(idx_next[0].tolist()[0])
            # append sampled index to the running sequence
            ctx = torch.cat((ctx, encoded_patterns[idx_next][0]), dim=1)
        self.train()
        return output

# Generate the patters for the inputs encoding
encoded_patterns = gen_coding_patterns()

model = SimpleLanguageModel()
m = model.to(device)
# print the number of parameters in the model
model_million_params = sum(p.numel() for p in m.parameters())/1e6
print(model_million_params, 'M parameters')
print(m)

# If the second argument is a model name, we just load the model
# and generate some text with it.
if len(sys.argv) == 3:
    torch.manual_seed(int(time.time()*1000))
    m.load_state_dict(torch.load(sys.argv[2]))
    context = torch.zeros((1,(block_size*char_encoding_len)), dtype=torch.float, device=device)
    print(decode(m.generate(context, max_new_tokens=1024)).strip())
    exit(0)

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# Training stage
iter_duration = 0
if __name__ == "__main__":
    # Log the loss in some target file
    model_id = f"loss_BA:{batch_size}_BL:{block_size}_PAR:{model_million_params:.2f}_E:{char_encoding_len}_V:{vocab_size}_BN:{use_batch_norm}_LR:{learning_rate}_{os.path.basename(filename)}"
    model_filename = model_id+".pth"

    # If a model with this parameters was already trained, don't overwrite
    # the weights and loss log.
    if os.path.exists(model_filename):
        sys.exit(f"Pretrained weights found for this model: {model_filename}. If you want to proceed remove the file.")

    loss_file = open(model_id,'w')
    print("Logging to", model_id)

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    minloss = 1000 # Track minimum validation loss found so far.
    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            best_so_far = losses['val'] < minloss
            minloss = min(minloss,losses['val'])
            print(f">>> step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, min loss {minloss:.4f}, {iter_duration*1000:.2f} ms per step")
            if iter > 0:
                loss_file.write(f"{iter} {losses['train']:.4f} {losses['val']:.4f}\n")
                loss_file.flush()
            context = torch.zeros((1,(block_size*char_encoding_len)), dtype=torch.float, device=device)
            if best_so_far:
                print(decode(m.generate(context, max_new_tokens=200)).strip())
                torch.save(m.state_dict(),model_filename)
                print("Saving model ",model_filename)
                print("")

        # sample a batch of data
        iter_start_time = time.time()
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        iter_duration = time.time() - iter_start_time
