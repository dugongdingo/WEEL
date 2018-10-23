# mostly copied from the torch tutorial.
import itertools
import math
import random
import time

import tqdm
import torch
import torch.nn.functional as functional

from ..utils import to_tensor
from ..settings import DEVICE, MAX_LENGTH

class EncoderParams():
    def __init__(self,hidden_size=None,retrain=False,):
        self.hidden_size = hidden_size
        self.retrain = retrain

class EncoderRNN(torch.nn.Module):
    def __init__(self, fasttext_embeddings, **params):
        super(EncoderRNN, self).__init__()
        self.params = EncoderParams(**params)

        self.embedding = torch.nn.Embedding(*fasttext_embeddings.shape)
        self.embedding.weight.data.copy_(torch.from_numpy(fasttext_embeddings))
        self.embedding.requires_grad = self.params.retrain

        self.gru = torch.nn.GRU(
            fasttext_embeddings.shape[1],
            self.params.hidden_size
        )

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.params.hidden_size, device=DEVICE)


class DecoderParams():
    def __init__(self, hidden_size=None, output_size=None, dropout_p=0.01,
        max_length=MAX_LENGTH, retrain=True,):
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.retrain = retrain

class AttnDecoderRNN(torch.nn.Module):
    def __init__(self,fasttext_embeddings,**params):
        super(AttnDecoderRNN, self).__init__()
        self.params = DecoderParams(**params)

        self.embedding = torch.nn.Embedding(*fasttext_embeddings.shape)
        self.embedding.weight.data.copy_(torch.from_numpy(fasttext_embeddings))
        self.embedding.requires_grad = self.params.retrain

        attention_size = fasttext_embeddings.shape[1] + self.params.hidden_size
        self.attn = torch.nn.Linear(attention_size, self.params.max_length)
        self.attn_combine = torch.nn.Linear(attention_size, self.params.hidden_size)

        self.dropout = torch.nn.Dropout(self.params.dropout_p)

        self.gru = torch.nn.GRU(self.params.hidden_size, self.params.hidden_size)

        self.out = torch.nn.Linear(self.params.hidden_size, self.params.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = functional.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = functional.relu(output)
        output, hidden = self.gru(output, hidden)
        output = functional.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.params.hidden_size, device=DEVICE)

class Seq2SeqParams():
    def __init__(self, learning_rate=0.001, sequence_start=None, end_signal=None, teacher_forcing_ratio=0.5, max_length = MAX_LENGTH,):
        self.learning_rate=learning_rate
        self.sequence_start = sequence_start
        self.end_signal = end_signal
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.max_length = max_length


class Seq2SeqModel() :
    def __init__(self, encoder, decoder, **params) :
        self.params = Seq2SeqParams(**params)

        self.encoder = encoder
        self.encoder_optimizer = torch.optim.SGD(self.encoder.parameters(), lr=self.params.learning_rate)

        self.decoder = decoder
        self.decoder_optimizer = torch.optim.SGD(self.decoder.parameters(), lr=self.params.learning_rate)

        self.criterion = torch.nn.NLLLoss()

    def _train_one(self, ipt, opt):
        encoder_hidden = self.encoder.initHidden()

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length = ipt.size(0)
        target_length = opt.size(0)

        encoder_outputs = torch.zeros(self.params.max_length, self.encoder.params.hidden_size, device=DEVICE)

        loss = 0

        for ei in range(min(input_length, self.params.max_length)):
            encoder_output, encoder_hidden = self.encoder(ipt[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([self.params.sequence_start], device=DEVICE)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < self.params.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output, opt[di])
                decoder_input = opt[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += self.criterion(decoder_output, opt[di])
                if decoder_input.item() == self.params.end_signal:
                    break

        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item() / target_length

    def train(self, ipts, opts, epoch_number=None) :
        start = time.time()
        losses = []
        n_iters = len(ipts)

        with tqdm.tqdm(total=n_iters, desc="Training epoch #" + epoch_number, ascii=True) as pbar :
            for input_tensor, target_tensor in zip(
                (to_tensor(i) for i in ipts),
                (to_tensor(o) for o in opts),
            ) :
                losses.append(self._train_one(input_tensor, target_tensor))
                pbar.update(1)
        return losses

    def run(self, input, opt):
        with torch.no_grad():
            loss = 0
            input_tensor = to_tensor(input)
            opt = to_tensor(opt)
            input_length = input_tensor.size()[0]
            encoder_hidden = self.encoder.initHidden()

            encoder_outputs = torch.zeros(self.params.max_length, self.encoder.params.hidden_size, device=DEVICE)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            decoder_input = torch.tensor([self.params.sequence_start], device=DEVICE)

            decoder_hidden = encoder_hidden

            decoded_words = []
            decoder_attentions = torch.zeros(self.params.max_length, self.params.max_length)

            for di in range(self.params.max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                decoded_words.append(topi.item())
                if di < opt.size(0) :
                    loss += self.criterion(decoder_output, opt[di])
                if topi.item() == self.params.end_signal:
                    break

                decoder_input = topi.squeeze().detach()

            return decoded_words, loss.item() / opt.size(0)
