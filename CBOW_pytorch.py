import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

torch.manual_seed(1)

# Implementing CBOW model for the exercise given by a tutorial in pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
context_size = 2 # {w_i-2 ... w_i ... w_i+2}
embedding_dim = 10

raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

def make_context_vector(context, word_to_idx):
    idxs = [word_to_idx[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

vocab = set(raw_text)
vocab_size = len(vocab)

word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}

data = []

for i in range(2, len(raw_text) - 2):
    context = [raw_text[i-2], raw_text[i-1],
               raw_text[i+1], raw_text[i+2]]
    target = raw_text[i]
    data.append((context, target))

class CBOW(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.proj = nn.Linear(embedding_dim, 128)
        self.output = nn.Linear(128, vocab_size)
        
    def forward(self, inputs):
        embeds = sum(self.embeddings(inputs)).view(1, -1)
        out = F.relu(self.proj(embeds))
        out = self.output(out)
        nll_prob = F.log_softmax(out, dim=-1)
        return nll_prob

model = CBOW(vocab_size, embedding_dim)
optimizer = optim.SGD(model.parameters(), lr=0.001)

losses = []
loss_function = nn.NLLLoss()

for epoch in range(100):
    total_loss = 0
    for context, target in data:
        context_vector = make_context_vector(context, word_to_idx)
        
        # Remember PyTorch accumulates gradients; zero them out
        model.zero_grad()
        
        nll_prob = model(context_vector)
        loss = loss_function(nll_prob, Variable(torch.tensor([word_to_idx[target]])))
        
        # backpropagation
        loss.backward()
        # update the parameters
        optimizer.step() 
        
        total_loss += loss.item()
        
    losses.append(total_loss)

print(losses)

# Let's see if our CBOW model works or not

print("*************************************************************************")

context = ['process.','Computational','are', 'abstract']
context_vector = make_context_vector(context, word_to_idx)
a = model(context_vector).data.numpy()
print('Raw text: {}\n'.format(' '.join(raw_text)))
print('Test Context: {}\n'.format(context))
max_idx = np.argmax(a)
print('Prediction: {}'.format(idx_to_word[max_idx]))

