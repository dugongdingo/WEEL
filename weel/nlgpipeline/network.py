# mostly copied from the torch tutorial.
import itertools
import math
import random
import time

import tqdm
import torch
import torch.nn.functional as functional

from .preprocess import SOS, EOS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 50

class EncoderRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(torch.nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = torch.nn.Embedding(self.output_size, self.hidden_size)
        self.attn = torch.nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = torch.nn.Dropout(self.dropout_p)
        self.gru = torch.nn.GRU(self.hidden_size, self.hidden_size)
        self.out = torch.nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = functional.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = functional.relu(output)
        output, hidden = self.gru(output, hidden)

        output = functional.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class Seq2SeqModel() :
    def __init__(
            self,
            input_size,
            hidden_size,
            output_size,
            encoder_vocab,
            decoder_vocab,
            dropout_p=0.1,
            max_length=MAX_LENGTH,
            teacher_forcing_ratio=0.5,
            learning_rate=0.001,
            criterion=torch.nn.NLLLoss()
    ) :
        self.max_length = max_length
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.encoder = EncoderRNN(
            input_size,
            hidden_size
        ).to(device)
        self.encoder_optimizer = torch.optim.SGD(self.encoder.parameters(), lr=learning_rate)
        self.encoder_vocab = encoder_vocab
        self.decoder = AttnDecoderRNN(
            hidden_size,
            output_size,
            dropout_p=dropout_p,
            max_length=max_length
        ).to(device)
        self.decoder_optimizer = torch.optim.SGD(self.decoder.parameters(), lr=learning_rate)
        self.decoder_vocab = decoder_vocab
        self.criterion = criterion
        self.sos = self.decoder_vocab.encrypt((SOS,))
        self.eos = self.decoder_vocab[EOS]

    def _train_one(self, ipt, opt):
        #def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
        encoder_hidden = self.encoder.initHidden()

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length = ipt.size(0)
        target_length = opt.size(0)

        encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=device)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(ipt[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([self.sos], device=device)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output, opt[di])
                decoder_input = opt[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += self.criterion(decoder_output, opt[di])
                if decoder_input.item() == self.eos:
                    break

        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item() / target_length

    def train(self, ipts, opts, n_iters, print_every=1, plot_every=100) :
        start = time.time()
        losses = []

        training_pairs = itertools.cycle(map(lambda p:map(Seq2SeqModel.to_tensor, p), zip(ipts, opts)))

        with tqdm.tqdm(total=n_iters, desc="Training", leave=False) as pbar :
            for iter in range(n_iters):
                training_pair = list(next(training_pairs))
                input_tensor = training_pair[0]
                target_tensor = training_pair[1]

                losses.append(self._train_one(input_tensor, target_tensor))

                pbar.update(1)
        return losses

    def run(self, input):
        with torch.no_grad():
            input_tensor = Seq2SeqModel.to_tensor(input)
            input_length = input_tensor.size()[0]
            encoder_hidden = self.encoder.initHidden()

            encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=device)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([self.sos], device=device)  # SOS

            decoder_hidden = encoder_hidden

            decoded_words = []
            decoder_attentions = torch.zeros(self.max_length, self.max_length)

            for di in range(self.max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == self.eos:
                    decoded_words.append(EOS)
                    break
                else:
                    decoded_words.append(self.decoder_vocab.index2vocab[topi.item()])

                decoder_input = topi.squeeze().detach()

            return decoded_words, decoder_attentions[:di + 1]

    @staticmethod
    def to_tensor(seq) :
        return torch.tensor(seq, dtype=torch.long, device=device).view(-1, 1)
