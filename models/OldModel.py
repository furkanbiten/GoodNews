# This file contains ShowAttendTell and AllImg model

# ShowAttendTell is from Show, Attend and Tell: Neural Image Caption Generation with Visual Attention
# https://arxiv.org/abs/1502.03044

# AllImg is a model where
# img feature is concatenated with word embedding at every time step as the input of lstm
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
import misc.utils as utils

from .CaptionModel import CaptionModel

class OldModel(CaptionModel):
    def __init__(self, opt):
        super(OldModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        if 'sentence_embed_att' in opt:
            self.sentence_embed_att = opt.sentence_embed_att
        else:
            self.sentence_embed_att = False

        self.ss_prob = 0.0 # Schedule sampling probability
        # if opt.sentence_embed:
        #     self.sentence_embed_size = opt.sentence_embed_size
        #     # self.lda = nn.Linear(self.sentence_embed_size, self.rnn_size)
        #     self.lda = nn.Sequential(nn.Linear(self.sentence_embed_size, self.rnn_size),
        #                   nn.ReLU(),
        #                   nn.Dropout(self.drop_prob_lm))

        self.linear = nn.Linear(self.fc_feat_size, self.num_layers * self.rnn_size) # feature to rnn_size
        self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.dropout = nn.Dropout(self.drop_prob_lm)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embed.weight.data.uniform_(-initrange, initrange)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, fc_feats):
        image_map = self.linear(fc_feats).view(-1, self.num_layers, self.rnn_size).transpose(0, 1)
        if self.rnn_type == 'lstm':
            return (image_map, image_map)
        else:
            return image_map

    def forward(self, fc_feats, att_feats, seq, sen_embed=None, return_attention=False):
        batch_size = fc_feats.size(0)
        state = self.init_hidden(fc_feats)
        outputs = []
        if return_attention: coverage, cov_loss = torch.Tensor([]).cuda(), torch.zeros(batch_size).cuda()

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    #prob_prev = torch.exp(outputs[-1].data.index_select(0, sample_ind)) # fetch prev distribution: shape Nx(M+1)
                    #it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1))
                    prob_prev = torch.exp(outputs[-1].data) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                    it = Variable(it, requires_grad=False)
            else:
                it = seq[:, i].clone()          
            # break if all the sequences end
            if i >= 1 and seq[:, i].data.sum() == 0:
                break

            xt = self.embed(it)
            if return_attention:
                output, state, atts = self.core(xt, fc_feats, att_feats, state, sen_embed, return_attention)
                atts = torch.from_numpy(atts[1].squeeze(2)).cuda()
                if i != 0:
                    cov_loss += torch.sum(torch.min(atts, coverage), 1)
                    coverage += atts
                else:
                    coverage = torch.cat((coverage, atts), 0)
            else: output, state = self.core(xt, fc_feats, att_feats, state, sen_embed)
            output = F.log_softmax(self.logit(self.dropout(output)))
            outputs.append(output)
        if return_attention:
            return torch.cat([_.unsqueeze(1) for _ in outputs], 1), torch.sum(cov_loss)/batch_size
        else:
            return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def get_logprobs_state(self, it, tmp_fc_feats, tmp_att_feats, state):
        # 'it' is Variable contraining a word index
        xt = self.embed(it)

        output, state = self.core(xt, tmp_fc_feats, tmp_att_feats, state)
        logprobs = F.log_softmax(self.logit(self.dropout(output)))

        return logprobs, state

    def sample_beam(self, fc_feats, att_feats, opt={}, sen_embed=None, return_attention=False):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            tmp_fc_feats = fc_feats[k:k+1].expand(beam_size, self.fc_feat_size)
            tmp_att_feats = att_feats[k:k+1].expand(*((beam_size,)+att_feats.size()[1:])).contiguous()
            
            state = self.init_hidden(tmp_fc_feats)

            beam_seq = torch.LongTensor(self.seq_length, beam_size).zero_()
            beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size).zero_()
            beam_logprobs_sum = torch.zeros(beam_size) # running sum of logprobs for each beam
            done_beams = []
            for t in range(1):
                if t == 0: # input <bos>
                    it = fc_feats.data.new(beam_size).long().zero_()
                    xt = self.embed(Variable(it, requires_grad=False))

                if sen_embed is not None:
                    if return_attention:
                        output, state, att = self.core(xt, fc_feats, att_feats, state, sen_embed, return_attention)
                    else:
                        output, state = self.core(xt, fc_feats, att_feats, state, sen_embed)
                else:
                    output, state = self.core(xt, fc_feats, att_feats, state)
                # output, state = self.core(xt, tmp_fc_feats, tmp_att_feats, state)
                logprobs = F.log_softmax(self.logit(self.dropout(output)))

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def sample(self, fc_feats, att_feats, opt={}, sen_embed=None, return_attention=False):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        if beam_size > 1:
            return self.sample_beam(fc_feats, att_feats, opt, sen_embed, return_attention)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(fc_feats)

        if return_attention: atts = []

        seq = []
        seqLogprobs = []
        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.data.new(batch_size).long().zero_()
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data).cpu() # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature)).cpu()
                it = torch.multinomial(prob_prev, 1).cuda()
                sampleLogprobs = logprobs.gather(1, Variable(it, requires_grad=False)) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            xt = self.embed(Variable(it, requires_grad=False))

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq.append(it) #seq[t] the input of t+2 time step
                seqLogprobs.append(sampleLogprobs.view(-1))

            if sen_embed is not None:
                # sen_embed = self.lda(lda)
            #     output, state = self.core(xt, fc_feats, att_feats, state, sen_embed)
            # elif self.sentence_embed_att:
            #     sen_embed = self.lda(lda)
                if return_attention:
                    output, state, att = self.core(xt, fc_feats, att_feats, state, sen_embed, return_attention)
                    atts.append(att)
                else:
                    output, state = self.core(xt, fc_feats, att_feats, state, sen_embed)
            else:
                output, state = self.core(xt, fc_feats, att_feats, state)
            # output, state = self.core(xt, fc_feats, att_feats, state)
            logprobs = F.log_softmax(self.logit(self.dropout(output)))

        if return_attention:
            return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1), atts
        else:
            return torch.cat([_.unsqueeze(1) for _ in seq], 1), torch.cat([_.unsqueeze(1) for _ in seqLogprobs], 1)


class ShowAttendTellCore(nn.Module):
    def __init__(self, opt):
        super(ShowAttendTellCore, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        
        # sentence embedding parameters
        self.sentence_embed_method = vars(opt).get('sentence_embed_method', '')
        self.sentence_embed_att = vars(opt).get('sentence_embed_att', False)
        self.sentence_length = vars(opt).get('sentence_length', None)
        self.sentence_embed_size = vars(opt).get('sentence_embed_size', None)
        self.sentence_embed = vars(opt).get('sentence_embed', False)

        if self.sentence_embed_method == 'conv':
            self.sen_conv_ch = 32
            self.ctx2att_sen = []
            self.ctx2att_sen += [utils.LeakyReLUConv2d(1, self.sen_conv_ch, [self.sentence_embed_size, 5], 1, [0, 2])]
            self.ctx2att_sen += [nn.Dropout(self.drop_prob_lm)]
            self.ctx2att_sen = nn.Sequential(*self.ctx2att_sen)
            self.h2att_sen = nn.Linear(self.rnn_size, self.sentence_embed_size)
            self.ch_embed = nn.Sequential(nn.Linear(self.sen_conv_ch, 1),
                          # nn.ReLU(),
                          nn.Dropout(self.drop_prob_lm))

        elif self.sentence_embed_method == 'bnews':
            self.sen_conv_ch = 256
            self.ctx2att_sen = []
            self.ctx2att_sen += [nn.Conv2d(1, self.sen_conv_ch, [self.sentence_embed_size, 5], 1, [0, 0])]
            self.ctx2att_sen += [nn.MaxPool2d((1, self.sentence_length - 4), 1)]

            self.ctx2att_sen_lin = []
            self.ctx2att_sen_lin += [nn.Linear(self.sen_conv_ch, 64)]
            self.ctx2att_sen_lin += [nn.ReLU(inplace=True)]
            self.ctx2att_sen_lin += [nn.Dropout(p=0.1)]

            self.ctx2att_sen = nn.Sequential(*self.ctx2att_sen)
            self.ctx2att_sen_lin = nn.Sequential(*self.ctx2att_sen_lin)

        elif self.sentence_embed_method == 'conv_deep':
            self.sen_conv_ch = 128
            self.ctx2att_sen = []
            self.ctx2att_sen += [utils.LeakyReLUConv2d(1, self.sen_conv_ch, [self.sentence_embed_size, 5], 1, [0, 2])]
            self.ctx2att_sen += [utils.INSResBlock(self.sen_conv_ch, self.sen_conv_ch, [1, 5], 1, [0, 2])]
            self.ctx2att_sen += [utils.INSResBlock(self.sen_conv_ch, self.sen_conv_ch, [1, 5], 1, [0, 2])]

            self.ctx2att_sen += [nn.Dropout(self.drop_prob_lm)]
            self.ctx2att_sen = nn.Sequential(*self.ctx2att_sen)
            self.h2att_sen = nn.Linear(self.rnn_size, self.sentence_length)
            self.ch_embed = nn.Sequential(nn.Linear(self.sen_conv_ch, 1),
                                          # nn.ReLU(),
                                          nn.Dropout(self.drop_prob_lm))

        elif self.sentence_embed_method == 'fc' or self.sentence_embed_method == 'fc_max':
            self.sentence_att = nn.Linear(self.sentence_embed_size, self.att_hid_size)
            self.h2att_sen = nn.Linear(self.rnn_size, self.att_hid_size)
            self.alpha_net_sen = nn.Linear(self.att_hid_size, 1)



        if self.sentence_embed_att and (self.sentence_embed_method =='fc' or self.sentence_embed_method == 'fc_max'
                                        or self.sentence_embed_method == 'conv'):
            self.rnn = getattr(nn, self.rnn_type.upper())(self.input_encoding_size + self.att_feat_size
                                                          + self.sentence_embed_size, self.rnn_size,
                                                          self.num_layers, bias=False, dropout=self.drop_prob_lm)
        elif self.sentence_embed_method == 'bnews':
            self.rnn = getattr(nn, self.rnn_type.upper())(
                self.input_encoding_size + self.att_feat_size + 64,
                self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)

        elif self.sentence_embed_method =='conv_deep':
            self.rnn = getattr(nn, self.rnn_type.upper())(self.input_encoding_size + self.att_feat_size
                                                          + self.sen_conv_ch, self.rnn_size,
                                                          self.num_layers, bias=False, dropout=self.drop_prob_lm)
        elif self.sentence_embed and not self.sentence_embed_att:
            self.rnn = getattr(nn, self.rnn_type.upper())(
                self.input_encoding_size + self.att_feat_size + self.att_hid_size,
                self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)
        else:
            self.rnn = getattr(nn, self.rnn_type.upper())(self.input_encoding_size + self.att_feat_size,
                                                          self.rnn_size, self.num_layers, bias=False,
                                                          dropout=self.drop_prob_lm)



        if self.att_hid_size > 0:
            self.ctx2att = nn.Linear(self.att_feat_size, self.att_hid_size)
            self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
            self.alpha_net = nn.Linear(self.att_hid_size, 1)
        # else:
            # self.ctx2att = nn.Linear(self.att_feat_size, 1)
            # self.h2att = nn.Linear(self.rnn_size, 1)

    def forward(self, xt, fc_feats, att_feats, state, sen_embed=None, return_attention=False):
        att_size = att_feats.numel() // att_feats.size(0) // self.att_feat_size
        att = att_feats.contiguous().view(-1, self.att_feat_size)
        if self.att_hid_size > 0:
            att = self.ctx2att(att)                             # (batch * att_size) * att_hid_size
            att = att.view(-1, att_size, self.att_hid_size)     # batch * att_size * att_hid_size
            att_h = self.h2att(state[0][-1])                    # batch * att_hid_size
            att_h = att_h.unsqueeze(1).expand_as(att)           # batch * att_size * att_hid_size
            dot = att + att_h                                   # batch * att_size * att_hid_size
            dot = F.tanh(dot)                                   # batch * att_size * att_hid_size
            dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
            dot = self.alpha_net(dot)                           # (batch * att_size) * 1
            dot = dot.view(-1, att_size)                        # batch * att_size
            weight = F.softmax(dot)
            att_feats_ = att_feats.view(-1, att_size, self.att_feat_size)  # batch * att_size * att_feat_size
            att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size
        else:
            att_res = fc_feats
            # att = self.ctx2att(att)(att)                        # (batch * att_size) * 1
            # att = att.view(-1, att_size)                        # batch * att_size
            # att_h = self.h2att(state[0][-1])                    # batch * 1
            # att_h = att_h.expand_as(att)                        # batch * att_size
            # dot = att_h + att                                   # batch * att_size


        if self.sentence_embed_att:
            if self.sentence_embed_method == 'fc' or  self.sentence_embed_method =='fc_max':
                att_size_sen = self.sentence_length + 1
                att_sen = sen_embed.view(-1, self.sentence_embed_size).float()
                att_sen = self.sentence_att(att_sen)  # (batch * att_size) * att_hid_size
                att_sen = att_sen.view(-1, att_size_sen, self.att_hid_size)  # batch * att_size * att_hid_size
                att_h_sen = self.h2att_sen(state[0][-1])  # batch * att_hid_size
                att_h_sen = att_h_sen.unsqueeze(1).expand_as(att_sen)  # batch * att_size * att_hid_size
                dot = att_sen + att_h_sen  # batch * att_size * att_hid_size
                dot = F.tanh(dot)  # batch * att_size * att_hid_size
                # dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
                dot = self.alpha_net(dot)  # (batch * att_size) * 1
                # dot = dot.view(-1, att_size)  # batch * att_size

                weight_sen = F.softmax(dot)
                # att_feats_sen = att_feats.view(-1, att_size_sen, self.sentence_embed_size)  # batch * att_size * att_feat_size
                if self.sentence_embed_method == 'fc':
                    att_res_sen = torch.bmm(sen_embed.permute(0,2,1).float(), weight_sen).squeeze(2)  # batch * att_feat_size
                elif self.sentence_embed_method == 'fc_max':
                    # fancy indexing, we are taking the max of the attention values and choosing the sen_embed index accordingly.
                    att_res_sen = sen_embed[torch.arange(0, sen_embed.size()[0]).long(), weight_sen.argmax(1).squeeze(1), :]

            elif self.sentence_embed_method == 'conv':
                att_h_sen = self.h2att_sen(state[0][-1])
                sen = sen_embed + att_h_sen.unsqueeze(1)
                sen = sen.permute(0,2,1).unsqueeze(1)
                att_sen = self.ctx2att_sen(sen)
                dot = F.tanh(att_sen)
                dot = dot.squeeze(2).permute(0, 2, 1)
                weight_sen = F.softmax(self.ch_embed(dot).squeeze(2))
                att_res_sen = torch.bmm(sen_embed.permute(0,2,1), weight_sen.unsqueeze(2))
                att_res_sen = att_res_sen.squeeze(2)

            elif self.sentence_embed_method == 'conv_deep':
                att_h_sen = self.h2att_sen(state[0][-1])
                # sen = sen_embed + att_h_sen.unsqueeze(1)
                # sen = sen.permute(0,2,1).unsqueeze(1)
                att_sen = self.ctx2att_sen(sen_embed.permute(0,2,1).unsqueeze(1))
                att_sen_combined = att_h_sen.unsqueeze(1) + att_sen.squeeze(2)
                dot = F.tanh(self.ch_embed(att_sen_combined.permute(0,2,1)))
                # dot = dot.squeeze(2).permute(0, 2, 1)
                weight_sen = F.softmax(dot.squeeze(2))
                att_res_sen = torch.bmm(att_sen.squeeze(2), weight_sen.unsqueeze(2))
                att_res_sen = att_res_sen.squeeze(2)
        if self.sentence_embed_method == 'bnews':
            intermediate = self.ctx2att_sen(sen_embed.permute(0,2,1).unsqueeze(1))
            final = self.ctx2att_sen_lin(intermediate.squeeze(2).squeeze(2))

        if self.sentence_embed_method == 'bnews':
            output, state = self.rnn(torch.cat([xt, final, att_res], 1).unsqueeze(0), state)
        elif self.sentence_embed_att and (self.sentence_embed_method == 'conv' or self.sentence_embed_method =='fc'
                                          or self.sentence_embed_method =='fc_max'):
            output, state = self.rnn(torch.cat([xt, att_res, att_res_sen.float()], 1).unsqueeze(0), state)
        elif sen_embed is not None:
            output, state = self.rnn(torch.cat([xt, sen_embed, att_res], 1).unsqueeze(0), state)
        else:
            output, state = self.rnn(torch.cat([xt, att_res], 1).unsqueeze(0), state)

        if return_attention:
            return output.squeeze(0), state, [weight.data.cpu().numpy(), weight_sen.data.cpu().numpy()]
        else:
            return output.squeeze(0), state

class AllImgCore(nn.Module):
    def __init__(self, opt):
        super(AllImgCore, self).__init__()
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_type = opt.rnn_type
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        
        self.rnn = getattr(nn, self.rnn_type.upper())(self.input_encoding_size + self.fc_feat_size, 
                self.rnn_size, self.num_layers, bias=False, dropout=self.drop_prob_lm)

    def forward(self, xt, fc_feats, att_feats, state):
        output, state = self.rnn(torch.cat([xt, fc_feats], 1).unsqueeze(0), state)
        return output.squeeze(0), state

class ShowAttendTellModel(OldModel):
    def __init__(self, opt):
        super(ShowAttendTellModel, self).__init__(opt)
        self.core = ShowAttendTellCore(opt)

class AllImgModel(OldModel):
    def __init__(self, opt):
        super(AllImgModel, self).__init__(opt)
        self.core = AllImgCore(opt)

