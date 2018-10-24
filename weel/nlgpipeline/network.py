# mostly copied from the torch tutorials.
import itertools
import math
import random
import time

import tqdm
import torch
import torch.nn.functional as functional

from ..utils import to_tensor
from ..settings import DEVICE, MAX_LENGTH, HIDDEN_SIZE, N_LAYERS, CLIP

class EncoderParams():
    def __init__(self,hidden_size=HIDDEN_SIZE,retrain=False, n_layers=N_LAYERS):
        self.hidden_size = hidden_size
        self.retrain = retrain
        self.n_layers = n_layers


class EncoderRNN(torch.nn.Module):
    def __init__(self, fasttext_embeddings, **params):
        super(EncoderRNN, self).__init__()
        self.params = EncoderParams(**params)

        self.embedding = torch.nn.Embedding(*fasttext_embeddings.shape)
        self.embedding.weight.data.copy_(torch.from_numpy(fasttext_embeddings))
        self.embedding.requires_grad = self.params.retrain

        self.gru = torch.nn.GRU(
            fasttext_embeddings.shape[1],
            self.params.hidden_size,
            self.params.n_layers,
            bidirectional=True,
        )

    def forward(self, inputs, input_lengths, hidden=None):
        embedded = self.embedding(inputs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        return outputs, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.params.hidden_size, device=DEVICE)


class DecoderParams():
    def __init__(self, hidden_size=None, output_size=None, dropout_p=0.01,
        max_length=MAX_LENGTH, retrain=True, attn_method="general", n_layers=1,):
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.retrain = retrain
        self.attn_method = attn_method
        self.n_layers = n_layers

class AttentionLayer(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError()
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return functional.softmax(attn_energies, dim=1).unsqueeze(1)

class AttnDecoderRNN(torch.nn.Module):
    def __init__(self,fasttext_embeddings,**params):
        super(AttnDecoderRNN, self).__init__()
        self.params = DecoderParams(**params)
        # embedding
        self.embedding = torch.nn.Embedding(*fasttext_embeddings.shape)
        self.embedding.weight.data.copy_(torch.from_numpy(fasttext_embeddings))
        self.embedding.requires_grad = self.params.retrain
        # dropout
        self.dropout = torch.nn.Dropout(self.params.dropout_p)
        # reccurent cell
        self.gru = torch.nn.GRU(fasttext_embeddings.shape[1], self.params.hidden_size, self.params.n_layers)
        # attention layer
        self.attn = AttentionLayer(self.params.attn_method, self.params.hidden_size)
        # concatenation layer
        self.concat = torch.nn.Linear(self.params.hidden_size * 2, self.params.hidden_size)
        # output vocabulary mapping layer
        self.out = torch.nn.Linear(self.params.hidden_size, self.params.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # embed
        embedded = self.embedding(input)
        # drop
        embedded = self.dropout(embedded)
        # recur
        rnn_output, hidden = self.gru(embedded, hidden)
        # attend
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # concat
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # map to output
        output = self.out(concat_output)
        output = functional.softmax(output, dim=1)

        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.params.hidden_size, device=DEVICE)

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()

class Seq2SeqParams():
    def __init__(self, learning_rate=0.001, sequence_start=None, end_signal=None, teacher_forcing_ratio=0.5, max_length = MAX_LENGTH,):
        self.learning_rate = learning_rate
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

        self.criterion = maskNLLLoss

    def _train_one(self, ipt, lengths, opt, mask, device=DEVICE, clip=CLIP):
        encoder_hidden = self.encoder.initHidden()


        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()


        ipt, lengths, mask, opt = ipt.to(device), lengths.to(device), opt.to(device), mask.to(device)

        encoder_outputs, encoder_hidden = self.encoder(ipt, lengths)

        loss = 0
        n_totals = 0

        decoder_input = torch.LongTensor([[self.params.sequence_start] * opt.size(1)]).to(device)

        decoder_hidden = encoder_hidden[:self.decoders.params.n_layers]

        use_teacher_forcing = bool(random.random() < self.params.teacher_forcing_ratio)

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(opt.size(0)):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_input = opt[di].view(1, -1)
                mask_loss, n_total = self.criterion(decoder_output, opt[di], mask[di])
                loss += mask_loss
                n_totals += n_total
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(opt.size(0)):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([topi[i][0] for i in range(opt.size(1))]).to(device)
                mask_loss, n_total = self.criterion(decoder_output, opt[di], mask[di])
                loss += mask_loss
                n_totals += n_total

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item() / n_totals

    def train(self, batches, epoch_number=None) :
        losses = []
        n_iters = len(ipts)

        with tqdm.tqdm(total=n_iters, desc="Training epoch #" + epoch_number, ascii=True) as pbar :
            for input_tensor, lengths, target_tensor, mask in batches :
                losses.append(self._train_one(input_tensor, lengths, target_tensor, mask))
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
