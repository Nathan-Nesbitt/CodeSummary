"""
Modification of the lamner code originally written by Rishab Sharma 
to be run for single line use instead of files. 

To clarify, the code was originally written in a functional style but to 
simplify the usage, I altered LAMNER to be a class.

Nathan Nesbitt ~ 2021
"""

import torch
from rouge import FilesRouge
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.data import Field, BucketIterator
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import random
import math
import time
from torchtext import data
import torchtext.vocab as vocab
import nltk
from rouge import Rouge
from six.moves import map
from nltk.tokenize import word_tokenize

from pathlib import Path

from .lamnerCodePreProcess import tokenize_code

"""
    A little explanation: This is where all of the models are stored on the
    system, this is done because we cannot store 100+mb files on GitHub.  
"""
import os

models_path = Path(os.path.expanduser("~"), "Documents", ".models", "lamner")


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):

        # src = [src len, batch size]
        # src_len = [batch size]

        embedded = self.dropout(self.embedding(src))

        # embedded = [src len, batch size, emb dim]

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len)

        packed_outputs, hidden = self.rnn(packed_embedded)

        # packed_outputs is a packed sequence containing all hidden states
        # hidden is now from the final non-padded element in the batch

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        # outputs is now a non-packed sequence, all hidden states obtained
        #  when the input is a pad token are all zeros

        # outputs = [src len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]

        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer

        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(
            self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        )

        # outputs = [src len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):

        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        # repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, src len, dec hid dim]
        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)

        # attention = [batch size, src len]

        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(
        self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention
    ):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask):

        # input = [batch size]
        # hidden = [batch size, dec hid dim]
        # encoder_outputs = [src len, batch size, enc hid dim * 2]
        # mask = [batch size, src len]

        input = input.unsqueeze(0)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        a = self.attention(hidden, encoder_outputs, mask)

        # a = [batch size, src len]

        a = a.unsqueeze(1)

        # a = [batch size, 1, src len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # encoder_outputs = [batch size, src len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)

        # weighted = [batch size, 1, enc hid dim * 2]

        weighted = weighted.permute(1, 0, 2)

        # weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim=2)

        # rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        # output = [seq len, batch size, dec hid dim * n directions]
        # hidden = [n layers * n directions, batch size, dec hid dim]

        # seq len, n layers and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, dec hid dim]
        # hidden = [1, batch size, dec hid dim]
        # this also means that output == hidden
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))

        # prediction = [batch size, output dim]

        return prediction, hidden.squeeze(0), a.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device

    def create_mask(self, src):
        mask = (src != self.src_pad_idx).permute(1, 0)
        return mask

    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):

        # src = [src len, batch size]
        # src_len = [batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src, src_len)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        mask = self.create_mask(src)

        # mask = [batch size, src len]

        for t in range(1, trg_len):

            # insert input token embedding, previous hidden state, all encoder hidden states
            #  and mask
            # receive output tensor (predictions) and new hidden state
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs


def init_weights(m):
    for name, param in m.named_parameters():
        if "weight" in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(model, iterator, optimizer, criterion, clip):

    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):

        src, src_len = batch.code
        trg = batch.summary

        optimizer.zero_grad()

        output = model(src, src_len, trg)

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for i, batch in enumerate(iterator):

            src, src_len = batch.code
            trg = batch.summary

            output = model(src, src_len, trg, 0)  # turn off teacher forcing

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class Node:
    def __init__(self, prob, word, attention):

        self.prob = prob
        self.word = word
        self.attention = attention
        self.children = []

    def addChild(self, node):
        self.children.append(node)


def get_valid_words(output, beam_size, trg_indexes, fullstop):

    if len(trg_indexes) <= 1:
        probs, words = output.topk(beam_size, 1)
        fin_probs = []
        fin_words = []
        for i in range(beam_size):
            fin_words.append(words.cpu().detach().numpy()[0][i])
            fin_probs.append(probs.cpu().detach().numpy()[0][i])

        return fin_probs, fin_words

    else:
        all_combinations = []
        output[0][trg_indexes[-1]] = 0
        bigrams = zip(trg_indexes, trg_indexes[1:])
        for i in bigrams:
            all_combinations.append(i)

        for i in all_combinations:
            if not i[0] == fullstop and i[0] == trg_indexes[-1]:
                output[0][i[1]] = 0

        probs, words = output.topk(beam_size, 1)
        fin_probs = []
        fin_words = []
        for i in range(beam_size):
            fin_words.append(words.cpu().detach().numpy()[0][i])
            fin_probs.append(probs.cpu().detach().numpy()[0][i])

        return fin_probs, fin_words


def start_decode(
    leng,
    model,
    root,
    trg_indexes,
    hidden,
    encoder_outputs,
    mask,
    beam_size,
    eos_token,
    fullstop,
    device,
):
    if leng <= 64:
        # trg_tensor(trg_indexes), hidden, encoder_outputs(nochange), mask(nochange), attention
        # trg_tensor
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

        with torch.no_grad():
            output, hidden, attention = model.decoder(
                trg_tensor, hidden, encoder_outputs, mask
            )

        # probs, words = output.topk(beam_size,1)

        # check if words are valid with no reps

        probs, words = get_valid_words(
            F.softmax(output, dim=1), beam_size, trg_indexes, fullstop
        )

        for i in range(len(words)):
            if words[i] == eos_token:
                node = Node(probs[i], words[i], attention)
                root.addChild(node)
                return

            else:
                node = Node(probs[i], words[i], attention)
                root.addChild(node)
                trg_indexes.append(words[i])
                start_decode(
                    leng + 1,
                    model,
                    node,
                    trg_indexes,
                    hidden,
                    encoder_outputs,
                    mask,
                    beam_size,
                    eos_token,
                    fullstop,
                    device,
                )

        if leng == 0:
            return root


def getSequences(root, prob, maxi, path, paths, costs, eos, leng):
    while True:
        path = (path + " " + str(root.word)).strip(" ")
        prob = prob + root.prob
        if prob > maxi:
            maxi = prob
        for child in root.children:
            getSequences(child, prob, maxi, path, paths, costs, eos, leng + 1)

        if leng == 64 or root.word == eos:
            # print("Path "+path+"\n Maximun "+ str(maxi))
            paths.append(path)
            costs.append(maxi)
        return paths, costs


def translate_sentence(sentence, src_field, trg_field, model, device, max_len=64):

    model.eval()
    tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    src_len = torch.LongTensor([len(src_indexes)]).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)

    mask = model.create_mask(src_tensor)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    targetss = []
    node = Node(1, trg_field.vocab.stoi[trg_field.init_token], 1)
    rootOfTree = start_decode(
        0,
        model,
        node,
        trg_indexes,
        hidden,
        encoder_outputs,
        mask,
        1,
        trg_field.vocab.stoi[trg_field.eos_token],
        trg_field.vocab.stoi["."],
        device,
    )
    sequences, costs = getSequences(
        rootOfTree, 0, 0, "", [], [], trg_field.vocab.stoi[trg_field.eos_token], 0
    )

    for seq in sequences:
        seq = seq.split(" ")
        targetss.append([trg_field.vocab.itos[int(i)] for i in seq])
    return targetss, costs


def get_max_lens(train_data, test_data, valid_data, code=True):

    encoder_max = -1

    if code:
        for i in range(len(train_data)):
            if encoder_max < len(vars(train_data.examples[i])["code"]):
                encoder_max = len(vars(train_data.examples[i])["code"])

        for i in range(len(test_data)):
            if encoder_max < len(vars(test_data.examples[i])["code"]):
                encoder_max = len(vars(test_data.examples[i])["code"])

        for i in range(len(valid_data)):
            if encoder_max < len(vars(valid_data.examples[i])["code"]):
                encoder_max = len(vars(valid_data.examples[i])["code"])

    else:
        for i in range(len(train_data)):
            if encoder_max < len(vars(train_data.examples[i])["summary"]):
                encoder_max = len(vars(train_data.examples[i])["summary"])

        for i in range(len(test_data)):
            if encoder_max < len(vars(test_data.examples[i])["summary"]):
                encoder_max = len(vars(test_data.examples[i])["summary"])

        for i in range(len(valid_data)):
            if encoder_max < len(vars(valid_data.examples[i])["summary"]):
                encoder_max = len(vars(valid_data.examples[i])["summary"])
    return encoder_max


def check_ends_with(st):
    splitted = st.strip().split(" ")[-1]
    if splitted == "is" or splitted == "are" or splitted == "am":
        return False
    text = word_tokenize(st)
    last = nltk.pos_tag(text)[-1][-1][0]
    if last.lower() == "n" or last.lower() == "v":
        return True
    else:
        # print("Rejected: " + st)
        return False


def clean_translate(st):
    st = st.replace("<sos>", "")
    st = st.strip(" ")
    st = st.replace("<eos>", "")
    st = st.strip(" ")
    st = st.strip("\n").strip(" ")
    st = st.strip(".")
    st = st.strip(" ")
    return st


class Lamner:
    """
    This class initializes the lamner model so it can be used for
    code summarization.
    """

    def __init__(self):
        """
        Initializes the model as an object
        """
        SEED = 1234

        """ This sets the variables used in the model """
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True

        self.BATCH_SIZE = 16
        self.ENC_EMB_DIM = 512
        self.DEC_EMB_DIM = 512
        self.ENC_HID_DIM = 512
        self.DEC_HID_DIM = 512
        self.ENC_DROPOUT = 0.5
        self.DEC_DROPOUT = 0.5
        self.N_EPOCHS = 200
        self.CLIP = 1
        self.best_valid_loss = float("inf")
        self.cur_rouge = -float("inf")
        self.best_rouge = -float("inf")
        self.best_epoch = -1
        self.make_weights_static = False
        self.MIN_LR = 0.0000001
        self.MAX_VOCAB_SIZE = 50000
        self.LEARNING_RATE = 0.1
        self.early_stop = False
        self.cur_lr = self.LEARNING_RATE
        self.num_of_epochs_not_improved = 0

        # Commented out as I cannot run GPU on my machine
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = torch.device("cpu")

        # Initialize the model
        self.initialize_model()

    def initialize_model(self):
        """
        This does a set of things:
            1. It starts by defining how the source/target will be defined
            2. Then it loads all of the embeddings for the model
            3. It then creates the model, this includes the
        """

        # This defines the
        self.SRC = Field(
            init_token="<sos>", eos_token="<eos>", lower=True, include_lengths=True
        )
        self.TRG = Field(init_token="<sos>", eos_token="<eos>", lower=True)

        """ This creates a dataset pipline from a table (CSV in our case) """
        self.train_data, self.valid_data, self.test_data = data.TabularDataset.splits(
            path=Path(__file__).parent / Path(models_path, "data_seq2seq/"),
            train="train_seq.csv",
            skip_header=True,
            validation="valid_seq.csv",
            test="test_seq.csv",
            format="CSV",
            fields=[("code", self.SRC), ("summary", self.TRG)],
        )

        # This loads the custom embeddings from pre-training
        """This gets the embeddings for the encoder and decoder"""
        self.custom_embeddings_semantic_encoder = vocab.Vectors(
            name=Path(__file__).parent
            / Path(models_path, "custom_embeddings/semantic_embeds.txt"),
            cache=Path(__file__).parent
            / Path(models_path, "custom_embeddings_semantic_encoder"),
            unk_init=torch.Tensor.normal_,
        )

        self.custom_embeddings_syntax_encoder = vocab.Vectors(
            name=Path(__file__).parent
            / Path(models_path, "custom_embeddings/syntax_embeds.txt"),
            cache=Path(__file__).parent
            / Path(models_path, "custom_embeddings_syntax_encoder"),
            unk_init=torch.Tensor.normal_,
        )

        self.custom_embeddings_decoder = vocab.Vectors(
            name=Path(__file__).parent
            / Path(models_path, "custom_embeddings/decoder_embeddings.txt"),
            cache=Path(__file__).parent
            / Path(models_path, "custom_embeddings_decoder"),
            unk_init=torch.Tensor.normal_,
        )

        # This creates the vocabulary
        self.SRC.build_vocab(
            self.train_data,
            max_size=self.MAX_VOCAB_SIZE,
            vectors=self.custom_embeddings_semantic_encoder,
        )

        self.TRG.build_vocab(
            self.train_data,
            max_size=self.MAX_VOCAB_SIZE,
            vectors=self.custom_embeddings_decoder,
        )

        """
        These are some params for the model. These include
        - Size of the input vocabulary
        - Size of the output vocabulary
        - IDS of the source vocab to their associated ID's
        - Attention for the model, as defined in the Attention class
        - The encoder which takes in the input dims, encoder dims, encoder hidden layers, decoder hidden layers, and dropout
        - The decoder which takes the output dimensions, the decoder embedding dims, ... and attention object
        - It then transforms this into a SEQ2SEQ model
        """
        self.INPUT_DIM = len(self.SRC.vocab)
        self.OUTPUT_DIM = len(self.TRG.vocab)
        self.SRC_PAD_IDX = self.SRC.vocab.stoi[self.SRC.pad_token]
        self.attn = Attention(self.ENC_HID_DIM, self.DEC_HID_DIM)
        self.enc = Encoder(
            self.INPUT_DIM,
            self.ENC_EMB_DIM,
            self.ENC_HID_DIM,
            self.DEC_HID_DIM,
            self.ENC_DROPOUT,
        )
        self.dec = Decoder(
            self.OUTPUT_DIM,
            self.DEC_EMB_DIM,
            self.ENC_HID_DIM,
            self.DEC_HID_DIM,
            self.DEC_DROPOUT,
            self.attn,
        )
        self.model = Seq2Seq(self.enc, self.dec, self.SRC_PAD_IDX, self.device).to(
            self.device
        )
        self.model.apply(init_weights)

        """ We need to create the vocabulary for the input training data """
        self.SRC.build_vocab(
            self.train_data,
            max_size=self.MAX_VOCAB_SIZE,
            vectors=self.custom_embeddings_semantic_encoder,
        )
        embeddings_enc1 = self.SRC.vocab.vectors

        self.SRC.build_vocab(
            self.train_data,
            max_size=self.MAX_VOCAB_SIZE,
            vectors=self.custom_embeddings_syntax_encoder,
        )
        embeddings_enc2 = self.SRC.vocab.vectors
        embeddings_enc3 = torch.cat([embeddings_enc1, embeddings_enc2], dim=1)

        """Using the concatenated two vocabs (semantic and syntax) into the model"""
        self.model.encoder.embedding.weight.data.copy_(embeddings_enc3)

        embeddings_trg = self.TRG.vocab.vectors
        self.model.decoder.embedding.weight.data.copy_(embeddings_trg)

        del embeddings_trg, embeddings_enc1, embeddings_enc3, embeddings_enc2

        optimizer = optim.SGD(
            self.model.parameters(), lr=self.LEARNING_RATE, momentum=0.9
        )

        self.TRG_PAD_IDX = self.TRG.vocab.stoi[self.TRG.pad_token]

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.TRG_PAD_IDX)

        self.model.load_state_dict(
            torch.load(
                Path(__file__).parent / Path(models_path, "best-seq2seq.pt"),
                map_location="cpu",
            )
        )

    def translate(self, value):
        """
        Takes in an array of type
        """

        value = tokenize_code(value)

        value = value.split(" ")

        translation, attention = translate_sentence(
            value, self.SRC, self.TRG, self.model, self.device
        )

        fin_costs = []

        for j in range(len(translation)):
            a = attention[j] / (
                len(translation[j]) + (len(translation[j]) - len(set(translation[j])))
            )
            fin_costs.append(a)

        # finding top 50 indices with highest cost
        tpk = 50
        if len(fin_costs) < 50:
            tpk = len(fin_costs)
        cos, indices = torch.tensor(fin_costs).topk(tpk)

        sampToStri = " ".join(translation[indices[0]]).strip(" ")  # string

        fin = sampToStri.replace("<sos>", "")
        fin = fin.replace("<eos>", "")
        fin = fin.strip(" ")
        return fin


def main():
    """
    This is a test main script to be sure that the code is running.
    """
    print("Loading...")
    lamner = Lamner()
    while True:
        val = input("\nCode Input > ")
        val = tokenize_code(val)
        result = lamner.translate(val)
        print("\nOutput:", result)


if __name__ == "__main__":
    main()
