import os
import math
import time
import spacy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from typing import List
import config
from dataset import Dataset

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, dropout):
        super().__init__()
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.input_dim = input_dim
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, n_layers, dropout=dropout,
                          bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

    def forward(self, src_batch):
        embedded = self.embedding(src_batch)
        outputs, hidden = self.rnn(embedded)
        concated = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden = torch.tanh(self.fc(concated))
        return outputs, hidden


class Attention(nn.Module):
    
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        # enc_hid_dim multiply by 2 due to bidirectional
        self.fc1 = nn.Linear(enc_hid_dim * 2 + dec_hid_dim, dec_hid_dim)
        self.fc2 = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, encoder_outputs, hidden):
        src_len = encoder_outputs.shape[0]
        batch_size = encoder_outputs.shape[1]
        
        # repeat encoder hidden state src_len times [batch size, sent len, dec hid dim]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        # reshape/permute the encoder output, so that the batch size comes first
        # [batch size, sent len, enc hid dim * 2], times 2 because of bidirectional
        outputs = encoder_outputs.permute(1, 0, 2)

        # the attention mechanism receives a concatenation of the hidden state
        # and the encoder output
        concat = torch.cat((hidden, outputs), dim=2)
        
        # fully connected layer and softmax layer to compute the attention weight
        # [batch size, sent len, dec hid dim]
        energy = torch.tanh(self.fc1(concat))

        # attention weight should be of [batch size, sent len]
        attention = self.fc2(energy).squeeze(dim=2)        
        attention_weight = torch.softmax(attention, dim=1)
        return attention_weight

class Decoder(nn.Module):
    
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers,
                 dropout, attention):
        super().__init__()
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(enc_hid_dim * 2 + emb_dim, dec_hid_dim, n_layers, dropout=dropout)
        self.linear = nn.Linear(dec_hid_dim, output_dim)

    def forward(self, trg, encoder_outputs, hidden):
        attention = self.attention(encoder_outputs, hidden).unsqueeze(1)

        outputs = encoder_outputs.permute(1, 0, 2)

        context = torch.bmm(attention, outputs).permute(1, 0, 2)

        embedded = self.embedding(trg.unsqueeze(0))
        rnn_input = torch.cat((embedded, context), dim=2)

        outputs, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        prediction = self.linear(outputs.squeeze(0))
        return prediction, hidden.squeeze(0)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src_batch, trg_batch, teacher_forcing_ratio=0.5):
        max_len, batch_size = trg_batch.shape
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src_batch)

        trg = trg_batch[0]
        for i in range(1, max_len):
            prediction, hidden = self.decoder(trg, encoder_outputs, hidden)
            outputs[i] = prediction

            if random.random() < teacher_forcing_ratio:
                trg = trg_batch[i]
            else:
                trg = prediction.argmax(1)

        return outputs

if __name__ == "__main__":
    dataset = Dataset()
    train_data, valid_data, test_data, INPUT_DIM, OUTPUT_DIM = dataset.get_data()
    
    print(config.ENC_HID_DIM)
    
    attention = Attention(config.ENC_HID_DIM, config.DEC_HID_DIM)
    encoder = Encoder(INPUT_DIM, config.ENC_EMB_DIM, config.ENC_HID_DIM, config.DEC_HID_DIM, config.N_LAYERS, config.ENC_DROPOUT)
    decoder = Decoder(OUTPUT_DIM, config.DEC_EMB_DIM, config.ENC_HID_DIM, config.DEC_HID_DIM, config.N_LAYERS, config.DEC_DROPOUT, attention)
    seq2seq = Seq2Seq(encoder, decoder, config.device).to(config.device)
    print(seq2seq)