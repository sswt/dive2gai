import sys
import math
import copy
import logging
from collections import namedtuple
from time import strftime, gmtime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

args = {
  # Program
  "cuda": torch.cuda.is_available(),
  "shuffle": 0,
  "gen_init": "normal",
  "dis_init": "uniform",
  # Basic train
  "samples_num": 8000,  # 10000,
#   "vocab_size": 0,
  "mle_epoch": 500,  # 120,
  "adv_epoch": 100,  # 200,
#   "batch_size": 64,
#   "max_seq_len": 20,
  "start_letter": 1,
  "padding_idx": 0,
  "start_token": 'BOS',
  "padding_token": 'EOS',
  "gen_lr": 0.01,
  "dis_lr": 0.0001,
  "clip_norm": 5.0,
  "pre_log_step": 10,
  # Generator
  "adv_g_step": 1,
  "rollout_num": 16,
  "gen_embed_dim": 32,
  "gen_hidden_dim": 32,
  # Discriminator
  "d_step": 3,  # 5,
  "d_epoch": 3,
  "adv_d_step": 4,
  "adv_d_epoch": 2,
  "dis_embed_dim": 64,
  # Metrics
  "use_nll_gen": 1,
  "use_nll_div": 1,
  # Log
  "log_file": "log/log_1017_0931_25.txt",
  "tips": "SeqGAN experiments"
}

cfg = namedtuple('Struct', args)(**args)

class GANDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class GenDataIter:
    def __init__(self, samples, batch_size, max_seq_len, start_letter, shuffle, device=torch.device('cpu')):
        # self.max_seq_len = max_seq_len
        # self.start_letter = start_letter
        # self.word2idx_dict = word2idx_dict
        # self.idx2word_dict = idx2word_dict
        self.device = device
        data = [{'input': i, 'target': t}
                for (i, t) in zip(*self.prepare(samples, start_letter, max_seq_len, self.device))]
        self.loader = DataLoader(
            dataset=GANDataset(data),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True)

        self.input = self._all_data_('input')
        self.target = self._all_data_('target')

    # def random_batch(self):
    #     """Randomly choose a batch from loader, please note that the data should not be shuffled."""
    #     idx = random.randint(0, len(self.loader) - 1)
    #     return list(self.loader)[idx]

    def _all_data_(self, col):
        return torch.cat([data[col].unsqueeze(0) for data in self.loader.dataset.data], 0)

    @staticmethod
    def prepare(samples, start_letter, max_seq_len, device):
        """Add start_letter to samples as inp, target same as samples"""
        inp = torch.zeros(samples.size()).long()
        target = samples
        inp[:, 0] = start_letter
        inp[:, 1:] = target[:, :max_seq_len - 1]
        return inp.to(device), target.to(device)


class DisDataIter:
    def __init__(self, pos_samples, neg_samples, batch_size, max_seq_len, start_letter, shuffle, device=torch.device('cpu')):
        # self.max_seq_len = max_seq_len
        # self.start_letter = start_letter
        self.device = device
        self.loader = DataLoader(
            dataset=GANDataset(self.__read_data__(pos_samples, neg_samples)),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True)

    def __read_data__(self, pos_samples, neg_samples):
        """
        input: same as target, but start with start_letter.
        """
        inp, target = self.prepare(pos_samples, neg_samples, self.device)
        all_data = [{'input': i, 'target': t} for (i, t) in zip(inp, target)]
        return all_data

    # def random_batch(self):
    #     idx = random.randint(0, len(self.loader) - 1)
    #     return list(self.loader)[idx]

    @staticmethod
    def prepare(pos_samples, neg_samples, device):
        """Build inp and target"""
        inp = torch.cat((pos_samples, neg_samples), dim=0).long().detach()  # !!!need .detach()
        target = torch.ones(inp.size(0)).long()
        target[pos_samples.size(0):] = 0

        # shuffle
        perm = torch.randperm(inp.size(0))
        inp = inp[perm]
        target = target[perm]
        
        return inp.to(device), target.to(device)


def get_dict(word_set, cfg):
    word2idx_dict = {cfg.padding_token: str(cfg.padding_idx), cfg.start_token: str(cfg.start_letter)}
    offset = len(word2idx_dict)
    for i, word in enumerate(word_set):
        word2idx_dict[word] = str(i + offset)
    idx2word_dict = {v: k for k, v in word2idx_dict.items()}
    return word2idx_dict, idx2word_dict


def to_tensor(line, vocab, max_len, padding_idx):
    t = torch.LongTensor([int(vocab[line[li]]) for li in range(len(line[:max_len]))])
    if len(line) < max_len:
        t = nn.ConstantPad1d((0, max_len - len(line)), padding_idx)(t)
    return t

def tokens_to_tensor(lines, vocab, max_len, padding_idx):
    return torch.stack([to_tensor(l, vocab, max_len, padding_idx) for l in lines], axis=0)

def tensor_to_tokens(tensor, dictionary):
    """transform Tensor to word tokens"""
    tokens = []
    for sent in tensor:
        sent_token = []
        for word in sent.tolist():
            if word == cfg.padding_idx:
                break
            sent_token.append(dictionary[str(word)])
        tokens.append(sent_token)
    return tokens


def to_text(tensor, inv_vocab):
    return ''.join([inv_vocab[str(i.item())] for i in tensor if i != cfg.padding_idx])


def create_logger(name, silent=False, to_disk=False, log_file=None):
    """Create a new logger"""
    # setup logger
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    formatter = logging.Formatter(fmt='%(asctime)s - %(message)s', datefmt='%Y-%m-%dT%I:%M:%S')
    if not silent:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        log.addHandler(ch)
    if to_disk:
        log_file = log_file if log_file is not None else strftime("log/log_%m%d_%H%M.txt", gmtime())
        if type(log_file) == list:
            for filename in log_file:
                fh = logging.FileHandler(filename, mode='w')
                fh.setLevel(logging.INFO)
                fh.setFormatter(formatter)
                log.addHandler(fh)
        if type(log_file) == str:
            fh = logging.FileHandler(log_file, mode='w')
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            log.addHandler(fh)
    return log

def show_config():
    log.info(100 * '=')
    log.info('> training arguments:')
    for arg, val in cfg._asdict().items():
        log.info('>>> {0}: {1}'.format(arg, val))
    log.info(100 * '=')

class NLL():
    def __init__(self, name, if_use=False, gpu=False):
        self.name = name
        self.if_use = if_use
        self.model = None
        self.data_loader = None
        self.leak_dis = None
        self.gpu = gpu
        self.criterion = nn.NLLLoss()

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    def get_score(self):
        """note that NLL score need the updated model and data loader each time, use reset() before get_score()"""
        if not self.if_use:
            return 0
        assert self.model and self.data_loader, 'Need to reset() before get_score()!'

        if self.leak_dis is not None:  # For LeakGAN
            return self.cal_nll_with_leak_dis(self.model, self.data_loader, self.leak_dis, self.gpu)
        else:
            return self.cal_nll(self.model, self.data_loader, self.criterion, self.gpu)

    def reset(self, model=None, data_loader=None, label_i=None, leak_dis=None):
        self.model = model
        self.data_loader = data_loader
        self.label_i = label_i
        self.leak_dis = leak_dis

    @staticmethod
    def cal_nll(model, data_loader, criterion, gpu=cfg.cuda):
        """NLL score for general text generation model."""
        total_loss = 0
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inp, target = data['input'], data['target']
                if gpu:
                    inp, target = inp.cuda(), target.cuda()

                hidden = model.init_hidden(data_loader.batch_size)
                pred = model.forward(inp, hidden)
                loss = criterion(pred, target.view(-1))
                total_loss += loss.item()
        return round(total_loss / len(data_loader), 4)

    @staticmethod
    def cal_nll_with_leak_dis(model, data_loader, leak_dis, gpu=cfg.cuda):
        """NLL score for LeakGAN."""
        total_loss = 0
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inp, target = data['input'], data['target']
                if gpu:
                    inp, target = inp.cuda(), target.cuda()

                loss = model.batchNLLLoss(target, leak_dis)
                total_loss += loss.item()
        return round(total_loss / len(data_loader), 4)

log = create_logger(__name__, silent=False, to_disk=True, log_file=cfg.log_file)
show_config()
clas = None
max_len = 32  # default - 128, 32 for debug
batch_size = 128
device = torch.device('cuda') if cfg.cuda else torch.device('cpu')

with open('title_summary_ascii.txt') as fp:
    lines = [line[:max_len] for line in fp.readlines()]
train_lines = lines[:cfg.samples_num]
test_lines = lines[cfg.samples_num:cfg.samples_num+1024]
tokens = sorted(set([c for l in lines for c in l]))
vocab, inv_vocab = get_dict(tokens, cfg)
vocab_size = len(vocab)
train_samples = tokens_to_tensor(train_lines, vocab, max_len, cfg.padding_idx)
test_samples = tokens_to_tensor(test_lines, vocab, max_len, cfg.padding_idx)
train_data = GenDataIter(train_samples, batch_size, max_len, cfg.start_letter, cfg.shuffle)
test_data = GenDataIter(test_samples, batch_size, max_len, cfg.start_letter, cfg.shuffle)

# Criterion
mle_criterion = nn.NLLLoss()
dis_criterion = nn.CrossEntropyLoss()
clas_criterion = nn.CrossEntropyLoss()

# Optimizer
clas_opt = None

# Metrics
nll_gen = NLL('NLL_gen', if_use=cfg.use_nll_gen, gpu=cfg.cuda)
nll_div = NLL('NLL_div', if_use=cfg.use_nll_div, gpu=cfg.cuda)
nll_val = NLL('NLL_val', if_use=True, gpu=cfg.cuda)
all_metrics = [nll_gen, nll_div, nll_val]


def train_gen_epoch(model, data_loader, criterion, optimizer):
    total_loss = 0
    for i, data in enumerate(data_loader):
        inp, target = data['input'], data['target']
        if cfg.cuda:
            inp, target = inp.cuda(), target.cuda()

        hidden = model.init_hidden(data_loader.batch_size)
        pred = model.forward(inp, hidden)
        loss = criterion(pred, target.view(-1))
        optimize(optimizer, loss, model)
        total_loss += loss.item()
    return total_loss / len(data_loader)

def train_dis_epoch(model, data_loader, criterion, optimizer):
    total_loss = 0
    total_acc = 0
    total_num = 0
    for i, data in enumerate(data_loader):
        inp, target = data['input'], data['target']
        if cfg.cuda:
            inp, target = inp.cuda(), target.cuda()

        pred = model.forward(inp)
        loss = criterion(pred, target)
        optimize(optimizer, loss, model)

        total_loss += loss.item()
        total_acc += torch.sum((pred.argmax(dim=-1) == target)).item()
        total_num += inp.size(0)

    total_loss /= len(data_loader)
    total_acc /= total_num
    return total_loss, total_acc

def eval_dis(model, data_loader, criterion):
    total_loss = 0
    total_acc = 0
    total_num = 0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inp, target = data['input'], data['target']
            if cfg.cuda:
                inp, target = inp.cuda(), target.cuda()

            pred = model.forward(inp)
            loss = criterion(pred, target)
            total_loss += loss.item()
            total_acc += torch.sum((pred.argmax(dim=-1) == target)).item()
            total_num += inp.size(0)
        total_loss /= len(data_loader)
        total_acc /= total_num
    return total_loss, total_acc

def optimize(opt, loss, model=None, retain_graph=False):
    opt.zero_grad()
    loss.backward(retain_graph=retain_graph)
    if model is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
    opt.step()

def cal_metrics(fmt_str=False):
    """
    Calculate metrics
    :param fmt_str: if return format string for logging
    """
    with torch.no_grad():
        # Prepare data for evaluation
        eval_samples = gen.sample(cfg.samples_num, 4 * batch_size)
        gen_data = GenDataIter(eval_samples, batch_size, max_len, cfg.start_letter, cfg.shuffle)
        # Reset metrics
        nll_gen.reset(gen, train_data.loader)
        nll_div.reset(gen, gen_data.loader)
        nll_val.reset(gen, test_data.loader)

    if fmt_str:
        return ', '.join(['%s = %s' % (metric.get_name(), metric.get_score()) for metric in all_metrics])
    else:
        return [metric.get_score() for metric in all_metrics]

def pretrain_generator(epochs):
    """
    Max Likelihood Pre-training for the generator
    """
    losses = []
    for epoch in range(epochs):
        pre_loss = train_gen_epoch(gen, train_data.loader, mle_criterion, gen_opt)
        losses.append(pre_loss)
        # ===Test===
        if epoch % cfg.pre_log_step == 0 or epoch == epochs - 1:
            log.info('[MLE-GEN] epoch %d : pre_loss = %.4f, %s' % (epoch, pre_loss, cal_metrics(fmt_str=True)))
    return losses

def adv_train_generator(g_step):
    """
    The gen is trained using policy gradients, using the reward from the discriminator.
    Training is done for num_batches batches.
    """
    rollout_func = ROLLOUT(gen, cfg.cuda)
    total_g_loss = 0
    for step in range(g_step):
        inp, target = GenDataIter.prepare(gen.sample(batch_size, batch_size), cfg.start_letter, max_len, device)

        # ===Train===
        rewards = rollout_func.get_reward(target, cfg.rollout_num, dis)
        adv_loss = gen.batchPGLoss(inp, target, rewards)
        optimize(gen_adv_opt, adv_loss)
        total_g_loss += adv_loss.item()

    # ===Test===
    log.info('[ADV-GEN]: g_loss = %.4f, %s' % (total_g_loss, cal_metrics(fmt_str=True)))

def train_discriminator(d_step, d_epoch, phase='MLE'):
    """
    Training the discriminator on real_data_samples (positive) and generated samples from gen (negative).
    Samples are drawn d_step times, and the discriminator is trained for d_epoch d_epoch.
    """
    # prepare loader for validate
    global d_loss, train_acc
    for step in range(d_step):
        # prepare loader for training
        pos_samples = train_data.target
        neg_samples = gen.sample(cfg.samples_num, 4 * batch_size)
        dis_data = DisDataIter(pos_samples, neg_samples, batch_size, max_len, cfg.start_letter, cfg.shuffle)

        for epoch in range(d_epoch):
            # ===Train===
            d_loss, train_acc = train_dis_epoch(dis, dis_data.loader, dis_criterion, dis_opt)
        # ===Test===
        log.info('[%s-DIS] d_step %d: d_loss = %.4f, train_acc = %.4f,' % (phase, step, d_loss, train_acc))

class LSTMGenerator(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu=False):
        super(LSTMGenerator, self).__init__()
        self.name = 'vanilla'

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.gpu = gpu

        self.temperature = 1.0

        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.lstm2out = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

        self.init_params()

    def forward(self, inp, hidden, need_hidden=False):
        """
        Embeds input and applies LSTM
        :param inp: batch_size * seq_len
        :param hidden: (h, c)
        :param need_hidden: if return hidden, use for sampling
        """
        emb = self.embeddings(inp)  # batch_size * len * embedding_dim
        if len(inp.size()) == 1:
            emb = emb.unsqueeze(1)  # batch_size * 1 * embedding_dim

        out, hidden = self.lstm(emb, hidden)  # out: batch_size * seq_len * hidden_dim
        out = out.contiguous().view(-1, self.hidden_dim)  # out: (batch_size * len) * hidden_dim
        out = self.lstm2out(out)  # (batch_size * seq_len) * vocab_size
        # out = self.temperature * out  # temperature
        pred = self.softmax(out)

        if need_hidden:
            return pred, hidden
        else:
            return pred

    def sample(self, num_samples, batch_size, start_letter=cfg.start_letter):
        """
        Samples the network and returns num_samples samples of length max_seq_len.
        :return samples: num_samples * max_seq_length (a sampled sequence in each row)
        """
        num_batch = num_samples // batch_size + 1 if num_samples != batch_size else 1
        samples = torch.zeros(num_batch * batch_size, self.max_seq_len).long()

        # Generate sentences with multinomial sampling strategy
        for b in range(num_batch):
            hidden = self.init_hidden(batch_size)
            inp = torch.LongTensor([start_letter] * batch_size)
            if self.gpu:
                inp = inp.cuda()

            for i in range(self.max_seq_len):
                out, hidden = self.forward(inp, hidden, need_hidden=True)  # out: batch_size * vocab_size
                next_token = torch.multinomial(torch.exp(out), 1)  # batch_size * 1 (sampling from each row)
                samples[b * batch_size:(b + 1) * batch_size, i] = next_token.view(-1)
                inp = next_token.view(-1)
        samples = samples[:num_samples]

        return samples

    def init_params(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / math.sqrt(param.shape[0])
                if cfg.gen_init == 'uniform':
                    torch.nn.init.uniform_(param, a=-0.05, b=0.05)
                elif cfg.gen_init == 'normal':
                    torch.nn.init.normal_(param, std=stddev)
                elif cfg.gen_init == 'truncated_normal':
                    truncated_normal_(param, std=stddev)

    def init_oracle(self):
        for param in self.parameters():
            if param.requires_grad:
                torch.nn.init.normal_(param, mean=0, std=1)

    def init_hidden(self, batch_size=batch_size):
        h = torch.zeros(1, batch_size, self.hidden_dim)
        c = torch.zeros(1, batch_size, self.hidden_dim)

        if self.gpu:
            return h.cuda(), c.cuda()
        else:
            return h, c


class SeqGAN_G(LSTMGenerator):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu=False):
        super(SeqGAN_G, self).__init__(embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx, gpu)
        self.name = 'seqgan'

    def batchPGLoss(self, inp, target, reward):
        """
        Returns a policy gradient loss

        :param inp: batch_size x seq_len, inp should be target with <s> (start letter) prepended
        :param target: batch_size x seq_len
        :param reward: batch_size (discriminator reward for each sentence, applied to each token of the corresponding sentence)
        :return loss: policy loss
        """

        batch_size, seq_len = inp.size()
        hidden = self.init_hidden(batch_size)

        out = self.forward(inp, hidden).view(batch_size, self.max_seq_len, self.vocab_size)
        target_onehot = F.one_hot(target, self.vocab_size).float()  # batch_size * seq_len * vocab_size
        pred = torch.sum(out * target_onehot, dim=-1)  # batch_size * seq_len
        loss = -torch.sum(pred * reward)

        return loss


        
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]


class CNNDiscriminator(nn.Module):
    def __init__(self, embed_dim, vocab_size, filter_sizes, num_filters, padding_idx, gpu=False,
                 dropout=0.2):
        super(CNNDiscriminator, self).__init__()
        self.embedding_dim = embed_dim
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.feature_dim = sum(num_filters)
        self.gpu = gpu

        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n, (f, embed_dim)) for (n, f) in zip(num_filters, filter_sizes)
        ])
        self.highway = nn.Linear(self.feature_dim, self.feature_dim)
        self.feature2out = nn.Linear(self.feature_dim, 2)
        self.dropout = nn.Dropout(dropout)

        self.init_params()

    def forward(self, inp):
        """
        Get final predictions of discriminator
        :param inp: batch_size * seq_len
        :return: pred: batch_size * 2
        """
        feature = self.get_feature(inp)
        pred = self.feature2out(self.dropout(feature))

        return pred

    def get_feature(self, inp):
        """
        Get feature vector of given sentences
        :param inp: batch_size * max_seq_len
        :return: batch_size * feature_dim
        """
        emb = self.embeddings(inp).unsqueeze(1)  # batch_size * 1 * max_seq_len * embed_dim
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.convs]  # [batch_size * num_filter * length]
        pools = [F.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in convs]  # [batch_size * num_filter]
        pred = torch.cat(pools, 1)  # tensor: batch_size * feature_dim
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * F.relu(highway) + (1. - torch.sigmoid(highway)) * pred  # highway

        return pred

    def init_params(self):
        for param in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / math.sqrt(param.shape[0])
                if cfg.dis_init == 'uniform':
                    torch.nn.init.uniform_(param, a=-0.05, b=0.05)
                elif cfg.dis_init == 'normal':
                    torch.nn.init.normal_(param, std=stddev)
                elif cfg.dis_init == 'truncated_normal':
                    truncated_normal_(param, std=stddev)


class SeqGAN_D(CNNDiscriminator):
    def __init__(self, embed_dim, vocab_size, padding_idx, gpu=False, dropout=0.25):
        super(SeqGAN_D, self).__init__(embed_dim, vocab_size, dis_filter_sizes, dis_num_filters, padding_idx, gpu,
                                       dropout)


def truncated_normal_(tensor, mean=0, std=1):
    """
    Implemented by @ruotianluo
    See https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

class ROLLOUT:
    def __init__(self, gen, gpu=True):
        self.gen = gen
        self.old_model = copy.deepcopy(gen)
        self.max_seq_len = gen.max_seq_len
        self.vocab_size = gen.vocab_size
        self.step_size = gen.step_size if gen.name == 'leakgan' else 0
        self.goal_out_size = gen.goal_out_size if gen.name == 'leakgan' else 0
        self.gpu = gpu

    def rollout_mc_search(self, sentences, given_num):
        """
        fill up remain tokens with MC search
        :param sentences: size of batch_size * max_seq_len
        :param given_num:
        :return:
        """
        batch_size = sentences.size(0)

        # get current state
        hidden = self.gen.init_hidden(batch_size)
        # for i in range(given_num):
        inp = sentences[:, :given_num]
        out, hidden = self.gen.forward(inp, hidden, need_hidden=True)
        out = out.view(batch_size, -1, self.vocab_size)[:, -1]

        samples = torch.zeros(batch_size, self.max_seq_len).long()
        samples[:, :given_num] = sentences[:, :given_num]

        if self.gpu:
            samples = samples.cuda()

        # MC search
        for i in range(given_num, self.max_seq_len):
            out = torch.multinomial(torch.exp(out), 1)
            samples[:, i] = out.view(-1).data
            inp = out.view(-1)

            out, hidden = self.gen.forward(inp, hidden, need_hidden=True)

        return samples

    def rollout_mc_search_leakgan(self, sentences, dis, given_num):

        batch_size, seq_len = sentences.size()

        goal_array = torch.zeros((batch_size, seq_len + 1, self.goal_out_size))

        work_hidden = self.gen.init_hidden(batch_size)
        mana_hidden = self.gen.init_hidden(batch_size)
        real_goal = self.gen.goal_init[:batch_size, :]
        out = 0

        if self.gpu:
            goal_array = goal_array.cuda()
            real_goal = real_goal.cuda()

        # get current state
        for i in range(given_num):
            # Get feature.
            dis_inp = torch.zeros(batch_size, seq_len).long()
            dis_inp[:, :i + 1] = sentences[:, :i + 1]  # cut sentences
            leak_inp = sentences[:, i]
            if self.gpu:
                dis_inp = dis_inp.cuda()
                leak_inp = leak_inp.cuda()
            feature = dis.get_feature(dis_inp).unsqueeze(0)

            # Get output of one token
            # cur_goal: batch_size * 1 * goal_out_size
            out, cur_goal, work_hidden, mana_hidden = self.gen(i, leak_inp, work_hidden, mana_hidden,
                                                               feature, real_goal, train=True)

            # Save goal and update last_goal
            goal_array[:, i, :] = cur_goal.squeeze(1)
            if i > 0 and i % self.step_size == 0:
                real_goal = torch.sum(goal_array[:, i - 3:i + 1, :], dim=1)
                if i / self.step_size == 1:
                    real_goal += self.gen.goal_init[:batch_size, :]

        samples = torch.zeros(batch_size, self.max_seq_len).long()
        samples[:, :given_num] = sentences[:, :given_num]

        # MC search
        for i in range(given_num, self.max_seq_len):
            # Sample one token
            out = torch.multinomial(torch.exp(out), 1).view(-1)  # [num_samples] (sampling from each row)
            samples[:, i] = out.data

            # Get feature
            dis_inp = samples
            if self.gpu:
                dis_inp = dis_inp.cuda()
            feature = dis.get_feature(dis_inp).unsqueeze(0)
            leak_inp = out

            # Get output of one token
            # cur_goal: batch_size * 1 * goal_out_size
            out, cur_goal, work_hidden, mana_hidden = self.gen(i, leak_inp, work_hidden, mana_hidden,
                                                               feature, real_goal, train=True)

            # Save goal and update last_goal
            goal_array[:, i, :] = cur_goal.squeeze(1)
            if i > 0 and i % self.step_size == 0:
                real_goal = torch.sum(goal_array[:, i - 3:i + 1, :], dim=1)
                if i / self.step_size == 1:
                    real_goal += self.gen.goal_init[:batch_size, :]

        if self.gpu:
            samples = samples.cuda()

        return samples

    def get_reward(self, sentences, rollout_num, dis, current_k=0):
        """
        get reward via Monte Carlo search
        :param sentences: size of batch_size * max_seq_len
        :param rollout_num:
        :param dis:
        :param current_k: current training gen
        :return: reward: [batch_size]
        """
        with torch.no_grad():
            batch_size = sentences.size(0)
            rewards = torch.zeros([rollout_num * self.max_seq_len, batch_size]).float()
            if self.gpu:
                rewards = rewards.cuda()
            idx = 0
            for i in range(rollout_num):
                for given_num in range(1, self.max_seq_len + 1):
                    samples = self.rollout_mc_search(sentences, given_num)
                    out = dis.forward(samples)
                    out = F.softmax(out, dim=-1)
                    reward = out[:, current_k + 1]
                    rewards[idx] = reward
                    idx += 1

        # rewards = torch.mean(rewards, dim=0)
        rewards = torch.mean(rewards.view(batch_size, self.max_seq_len, rollout_num), dim=-1)
        return rewards

    def get_reward_leakgan(self, sentences, rollout_num, dis, current_k):
        """
        get reward via Monte Carlo search for LeakGAN
        :param sentences: size of batch_size * max_seq_len
        :param rollout_num:
        :param dis:
        :param current_k: current training gen

        :return: reward: batch_size * (max_seq_len / step_size)
        """
        with torch.no_grad():
            batch_size = sentences.size(0)
            rewards = torch.zeros([rollout_num * (self.max_seq_len // self.step_size), batch_size]).float()
            if self.gpu:
                rewards = rewards.cuda()
            idx = 0
            for i in range(rollout_num):
                for t in range(self.max_seq_len // self.step_size):
                    given_num = t * self.step_size + 1  # 1, 5, 9, ..
                    samples = self.rollout_mc_search_leakgan(sentences, dis, given_num)
                    out = dis(samples)
                    out = F.softmax(out, dim=-1)
                    reward = out[:, current_k + 1]
                    rewards[idx] = reward
                    idx += 1

        rewards = rewards.view(batch_size, self.max_seq_len // self.step_size, rollout_num)
        rewards = torch.mean(rewards, dim=-1)
        return rewards

    def get_token_reward(self, sentences, rollout_num, dis, current_k, given_num):
        """
        get reward of each token in sequence via Monte Carlo search
        """
        with torch.no_grad():
            batch_size = sentences.size(0)
            rewards = torch.zeros([rollout_num, batch_size]).float()
            idx = 0
            for i in range(rollout_num):
                samples = self.rollout_mc_search(sentences, given_num)
                out = dis(samples)
                out = F.softmax(out, dim=-1)
                reward = out[:, current_k + 1]
                rewards[idx] = reward
                idx += 1

        rewards = torch.Tensor(rewards).cuda()
        rewards = torch.sum(rewards, dim=0) / rollout_num
        return rewards

    def get_reward_csgan(self, target, rollout_num, csgan_clas):
        pass

# generator, discriminator
gen = SeqGAN_G(cfg.gen_embed_dim, cfg.gen_hidden_dim, vocab_size, max_len,
                    cfg.padding_idx, gpu=cfg.cuda).to(device)
dis = SeqGAN_D(cfg.dis_embed_dim, vocab_size, cfg.padding_idx, gpu=cfg.cuda).to(device)
# init_model()
log.info(f'Untrained gen sample: "{to_text(gen.sample(1, batch_size).squeeze(), inv_vocab)}"')

# Optimizer
gen_opt = torch.optim.Adam(gen.parameters(), lr=cfg.gen_lr)
gen_adv_opt = torch.optim.Adam(gen.parameters(), lr=cfg.gen_lr)
dis_opt = torch.optim.Adam(dis.parameters(), lr=cfg.dis_lr)

# ===PRE-TRAINING===
# TRAIN GENERATOR
log.info('Starting Generator MLE Training...')
losses = pretrain_generator(cfg.mle_epoch)
log.info(f'Pretrained gen sample: "{to_text(gen.sample(1, batch_size).squeeze(), inv_vocab)}"')

# ===TRAIN DISCRIMINATOR====
log.info('Starting Discriminator Training...')
train_discriminator(cfg.d_step, cfg.d_epoch)

# ===ADVERSARIAL TRAINING===
log.info('Starting Adversarial Training...')
log.info('Initial generator: %s' % (cal_metrics(fmt_str=True)))

for adv_epoch in range(cfg.adv_epoch):
    log.info('-----\nADV EPOCH %d\n-----' % adv_epoch)
    adv_train_generator(cfg.adv_g_step)  # Generator
    train_discriminator(cfg.adv_d_step, cfg.adv_d_epoch, 'ADV')  # Discriminator
    log.info(f'Current gen sample: "{to_text(gen.sample(1, batch_size).squeeze(), inv_vocab)}"')
