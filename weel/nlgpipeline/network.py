# mostly copied from the torch tutorials.
import itertools
import math
import random
import time

import tqdm
import torch
import torch.nn.functional as functional

from ..utils import to_tensor, to_device, to_batch_tensor
from ..settings import DEVICE, MAX_LENGTH, HIDDEN_SIZE, N_LAYERS, CLIP, BATCH_SIZE

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
            batch_first=False,
        )

    def forward(self, inputs, input_lengths, hidden=None):
        embedded = self.embedding(inputs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.params.hidden_size] + outputs[:, : ,self.params.hidden_size:]
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
        super(AttentionLayer, self).__init__()
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
    def __init__(self,fasttext_embeddings, hollistic_word_embeddings, **params):
        super(AttnDecoderRNN, self).__init__()
        self.params = DecoderParams(**params)
        # embedding
        self.embedding = torch.nn.Embedding(*fasttext_embeddings.shape)
        self.embedding.weight.data.copy_(torch.from_numpy(fasttext_embeddings))
        self.embedding.requires_grad = self.params.retrain
        # dropout
        self.dropout = torch.nn.Dropout(self.params.dropout_p)

        #hollistic word vectors
        self.hollistic_embedding = torch.nn.Embedding(*hollistic_word_embeddings.shape)
        self.hollistic_embedding.weight.data.copy_(torch.from_numpy(hollistic_word_embeddings))
        self.hollistic_embedding.requires_grad = False

        #embeddings concatenation layer
        self.embeddings_concat = torch.nn.Linear(fasttext_embeddings.shape[1] + hollistic_word_embeddings.shape[1], self.params.hidden_size)

        # reccurent cell
        self.gru = torch.nn.GRU(
            self.params.hidden_size,
            self.params.hidden_size,
            self.params.n_layers,
            batch_first=False,
        )
        # attention layer
        self.attn = AttentionLayer(self.params.attn_method, self.params.hidden_size)
        # attention concatenation layer
        self.attention_concat = torch.nn.Linear(self.params.hidden_size * 2, self.params.hidden_size)
        # output vocabulary mapping layer
        self.out = torch.nn.Linear(self.params.hidden_size, self.params.output_size)

    def forward(self, input, hidden, encoder_outputs, hollistic_indices):
        # embed
        embedded = self.embedding(input)
        hollistic_word_vectors = self.hollistic_embedding(hollistic_indices)
        # drop
        embedded = self.dropout(embedded)
        # concat with hollistic word vector
        embedded = torch.cat((embedded, hollistic_word_vectors), 2)
        embedded = torch.tanh(self.embeddings_concat(embedded))
        # recur
        rnn_output, hidden = self.gru(embedded, hidden)
        # attend
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # concat
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.attention_concat(concat_input))
        # map to output
        output = self.out(concat_output)
        output = functional.softmax(output, dim=1)

        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.params.hidden_size, device=DEVICE)

def maskNLLLoss(inp, target, mask, device=DEVICE):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()

class Seq2SeqParams():
    def __init__(self, learning_rate=0.0001, sequence_start=None, end_signal=None, teacher_forcing_ratio=1., max_length = MAX_LENGTH,):
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

    def _train_one(self, ipt, lengths, opt, mask, hollistic_indices, max_target_length, device=DEVICE, clip=CLIP, batch_size=BATCH_SIZE):
        encoder_hidden = self.encoder.initHidden()


        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        ipt, lengths, opt, hollistic_indices, mask = to_device(ipt, lengths, opt, hollistic_indices, mask, device=device)
        hollistic_indices = hollistic_indices.transpose(0,1)
        encoder_outputs, encoder_hidden = self.encoder(ipt, lengths)

        loss = 0
        n_totals = 0
        decoder_input = torch.LongTensor([[self.params.sequence_start] * batch_size]).to(device)

        decoder_hidden = encoder_hidden[:self.decoder.params.n_layers]

        use_teacher_forcing = True#bool(random.random() < self.params.teacher_forcing_ratio)

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for timestep in range(max_target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs, hollistic_indices)
                decoder_input = opt[timestep].view(1, -1)

                mask_loss, n_total = maskNLLLoss(decoder_output, opt[timestep], mask[timestep])
                loss += mask_loss
                n_totals += n_total
        else:
            # Without teacher forcing: use its own predictions as the next input
            for timestep in range(max_target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs, hollistic_indices)
                _, topi = decoder_output.topk(1)
                if len({i.item() for j in topi for i in j}) == 1 :
                    if next((i.item() for j in topi for i in j)) == 0 :
                        #import pdb; pdb.set_trace()
                        print(timestep)
                decoder_input = torch.LongTensor([topi[i][0] for i in range(batch_size)]).to(device).view(1, -1)
                mask_loss, n_total = maskNLLLoss(decoder_output, opt[timestep], mask[timestep])
                loss += mask_loss
                n_totals += n_total

        loss.backward()

        _ = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
        _ = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item() / n_totals, []

    def train(self, batches, n_iters, epoch_number=None) :
        losses, sentences = [], []

        with tqdm.tqdm(total=n_iters, desc="Training epoch #" + epoch_number, ascii=True) as pbar :
            for input_tensor, lengths, target_tensor, mask, hollistic_indices, max_target_length in batches :
                chunk_size = input_tensor.size(1)
                batch_loss, batched_sentences = self._train_one(input_tensor, lengths, target_tensor, mask, hollistic_indices, max_target_length, batch_size=chunk_size)
                losses.append(batch_loss)
                sentences.extend(batched_sentences)
                pbar.update(chunk_size)
        return losses

    def run(self, ipt, lengths, opt, mask, hollistic_indices, max_target_length, device=DEVICE, batch_size=1):
        with torch.no_grad() :
            words = []

            ipt, lengths, opt, hollistic_indices, mask = to_device(ipt, lengths, opt, hollistic_indices, mask, device=device)
            hollistic_indices = hollistic_indices.transpose(0,1)

            encoder_outputs, encoder_hidden = self.encoder(ipt, lengths)

            decoder_input = torch.LongTensor([[self.params.sequence_start] * batch_size]).to(device)

            words.append(self.params.sequence_start)

            decoder_hidden = encoder_hidden[:self.decoder.params.n_layers]

            # Without teacher forcing: use its own predictions as the next input
            for timestep in range(MAX_LENGTH):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs, hollistic_indices)
                _, topi = decoder_output.topk(1)
                word = topi[0][0].item()# for i in range(batch_size)
                words.append(word)
                if word == self.params.end_signal :
                    break
                decoder_input = torch.LongTensor([topi[i][0] for i in range(batch_size)]).to(device).view(1, -1)

            return words, 0

        """with torch.no_grad():
            loss = 0
            length = torch.tensor([len(input)]).to(DEVICE)
            hollistic_index = torch.LongTensor([hollistic_index]).transpose(0, 1).to(DEVICE)
            input_tensor = torch.LongTensor([input]).transpose(0, 1).to(DEVICE)
            opt = torch.LongTensor([opt]).transpose(0, 1).to(DEVICE)
            input_length = input_tensor.size()[0]

            encoder_outputs, encoder_hidden = self.encoder(input_tensor, length)
            decoder_input = decoder_input = torch.ones(1, 1, device=DEVICE, dtype=torch.long) * self.params.sequence_start

            decoder_hidden = encoder_hidden[:self.decoder.params.n_layers]

            all_tokens = torch.zeros([0], device=DEVICE, dtype=torch.long)
            all_scores = torch.zeros([0], device=DEVICE)
            # Iteratively decode one word token at a time
            for timestep in range(max_length):
                # Forward pass through decoder
                decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden, encoder_outputs, hollistic_index)
                # Obtain most likely word token and its softmax score
                decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
                # Record token and score
                all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
                all_scores = torch.cat((all_scores, decoder_scores), dim=0)
                # Prepare current token to be next decoder input (add a dimension)
            # Return collections of word tokens and scores
            if all_tokens.sum() == 0 :
                import pdb; pdb.set_trace()
            return all_tokens, all_scores"""
