import torch
import torch.nn as nn

# import torchvision.datasets as dsets
# import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

# from tree import Tree

class TreeNode(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, vocab_size, word_scale=0.05, glove_mat=None, cuda_enabled=True):
        super(TreeNode, self).__init__()
        
        self.cuda_enabled = cuda_enabled
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab_size = vocab_size

        if glove_mat is not None:
            self.init_embeddings(glove_mat)
        else:
            self.embeds = nn.Embedding(vocab_size, self.input_size)
            self.embeds.weight.data.uniform_(-word_scale, word_scale) # uniform vs guassion distribution
        
        if self.cuda_enabled: self.embeds = self.embeds.cuda()
        
        self.encode_inputx = nn.Linear(self.input_size, self.hidden_size)  # only used to encode word
        # hidden state of last child is input of parent, encode as input
        self.encode_inputh = nn.Linear(self.hidden_size, self.hidden_size) 
        self.encode_prevh = nn.Linear(self.hidden_size, self.hidden_size)  # encode hidden state 
        
        self.iofu = nn.Linear(self.hidden_size, self.hidden_size * 4)
    
    def toggle_embeddings_state(self, state=False):
        self.embeds.weight.requires_grad = state
    
    def init_embeddings(self, mat): # for futures's glove, word2vec and others 
        vocab_size, self.input_size = mat.size() # override the input_size
        assert vocab_size >= self.vocab_size
        
        self.embeds = nn.Embedding(vocab_size, self.input_size)
        self.embeds.weight.data.copy_(mat)
        
    def forward(self, tree, prev_h, prev_c):
        assert len(tree.children) != 0 or tree.word is not None
        assert not (len(tree.children) != 0 and tree.word is not None) # move assert code into tree
        
        if tree.getType(): # mid node
            tree.zero_state = self.init_state()
            self.forward(tree.children[0], * tree.zero_state)
            for idx in range(1, len(tree.children)):
                self.forward(tree.children[idx], *tree.children[idx-1].state)
            
            prev_ch, _ = tree.children[-1].state
            if self.hidden_size == self.input_size: # when input_x(word_embedding) == hidden_size, use same weights
                hx = self.encode_inputx(prev_ch) + self.encode_prevh(prev_h) # copy prev_ch or reference it?????  reference makes connected
            else:
                # last child's hidden is input; prev_h is sibling's hidden   # backward of mid-Node
                hx = self.encode_inputh(prev_ch) + self.encode_prevh(prev_h) # above.   should use reference
            
        else: # Leaf Node
            input_idx = Variable(torch.LongTensor([tree.word]))
            input_x = self.embeds(input_idx.cuda() if self.cuda_enabled else input_idx) # embedding is cuda/cpu, so input_x is correponding one
            hx = self.encode_inputx(input_x) + self.encode_prevh(prev_h) # prev_h is given as params (siblings or zeros)
        
        iofu = self.iofu(hx)
        i, o, f, u = torch.split(iofu, iofu.size(1) // 4, dim=1)
        i, o, f, u = F.sigmoid(i), F.sigmoid(o), F.sigmoid(f), F.tanh(u)
        cell = torch.mul(i, u) + torch.mul(f, prev_c)
        hidden = torch.mul(o, F.tanh(cell))
        tree.state = (hidden, cell)
        return tree.state
        
    def init_state(self, requires_grad=True):
        h, c = Variable(torch.zeros(1, self.hidden_size), requires_grad=requires_grad), Variable(torch.zeros(1, self.hidden_size), requires_grad=requires_grad) # 
        if self.cuda_enabled:
            return h.cuda(), c.cuda()
        else:
            return h, c


class PairCrossLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, pair_dim, output_dim, vocab_size, 
                 num_classes, word_scale=0.05, cuda_enbaled=False, glove_mat = None, freeze = False):
        super(PairCrossLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim #
        self.pair_dim = pair_dim
        self.vocab_size = vocab_size
        self.num_classes = num_classes

        self.freeze = freeze
        
        self.crosslstm = TreeNode(input_dim, hidden_dim, output_dim, vocab_size, word_scale, glove_mat, cuda_enbaled)
        if self.freeze:
            self.crosslstm.toggle_embeddings_state(freeze)

        self.wh = nn.Linear(2 * self.hidden_dim, self.pair_dim)
        self.wp = nn.Linear(self.pair_dim, self.num_classes)

    def forward(self, sent_a, sent_b):
        prev_ha, prev_ca = self.crosslstm.init_state()
        ha, _ = self.crosslstm(sent_a, prev_ha, prev_ca)
        prev_hb, prev_cb = self.crosslstm.init_state()
        hb, _ = self.crosslstm(sent_b, prev_hb, prev_cb)
        
        mult_dist = torch.mul(ha, hb)
        abs_dist = torch.abs(torch.add(ha, -hb))
        vec_dist = torch.cat((mult_dist, abs_dist), 1)

        out = F.sigmoid(self.wh(vec_dist))
        out = F.log_softmax(self.wp(out))
        return out
