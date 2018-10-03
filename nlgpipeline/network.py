# mostly copied from the torch tutorial.
import itertools
import random
import time

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
            criterion=torch.nn.NLLLoss(),
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
        sos = self.decoder_vocab.encrypt((SOS,))
        eos = self.decoder_vocab[EOS]
        decoder_input = torch.tensor([sos], device=device)

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
                if decoder_input.item() == eos:
                    break

        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item() / target_length

    def train(self, ipts, opts, n_iters) :
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        training_pairs = itertools.cycle(map(lambda p:map(Seq2SeqModel.to_tensor, p), zip(ipts, opts)))

        for iter in range(n_iters):
            training_pair = list(next(training_pairs))
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = self._train_one(input_tensor, target_tensor)
            print_loss_total += loss
            plot_loss_total += loss

            """if (iter+1) % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0"""


    @staticmethod
    def to_tensor(seq) :
        return torch.tensor(seq, dtype=torch.long, device=device).view(-1, 1)

if __name__ == "__main__" :
    from .preprocess import from_file, Vocab, SOS, EOS

    input, output = zip(*from_file("data/wn_english_entries.csv"))
    enc_voc, input = Vocab.process(input, preprocess=lambda seq: list(seq) + [EOS])
    dec_voc, output = Vocab.process(output, preprocess=lambda seq:[SOS] + seq.split() + [EOS])
    model = Seq2SeqModel(len(enc_voc), 256, len(dec_voc), enc_voc, dec_voc)
    model.train(input, output, 5000)
