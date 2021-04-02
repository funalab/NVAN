import torch
import torch.nn as nn
import torch.nn.functional as F

class NVAN(nn.Module):
    """
    Multi-view Attention Network for Multivariate Temporal Data

    Parameters
    ----------
    input_dim : dimention of multivariable
    num_classes : number of the output classes
    num_layers : number of hidden layers
    hidden_dim : number of unit on the hidden layer
    dropout : dropout rate
    lossfun : loss function
    phase : specify phase of training [train, validation, test]

    Inputs
    ------
    input: tensor, shaped [batch, time, view]

    Outputs
    -------
    output: tensor, shaped [batch, num_classes]
    """

    def __init__(
            self,
            input_dim,
            num_classes,
            num_layers,
            hidden_dim,
            dropout,
            lossfun,
            phase='train'
            ):
        super(NVAN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.bgru = nn.LSTM(1, hidden_dim, num_layers, dropout=dropout, bidirectional=True)

        # self.multi_view_attention = self.location_based_attention
        self.multi_view_attention = self.context_based_attention

        # location-based attention
        if self.multi_view_attention == self.location_based_attention:
            self.w_e = nn.Linear(hidden_dim * 2, 1)

        # context-based attention
        if self.multi_view_attention == self.context_based_attention:
            self.k = 128
            self.w_a = nn.Linear(self.k, input_dim)
            self.w_b = nn.Linear(self.k, input_dim)
            self.w_c = nn.Linear(self.k, input_dim)
            self.w_d = nn.Linear(input_dim, input_dim)
            # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            w_sc_i, w_sc_t = {}, {}
            for v in range(input_dim):
                w_sc_i[str(v)] = nn.Conv2d(1, 1, (1, hidden_dim * 2 - self.k + 1), 1, 0)
                w_sc_t[str(v)] = nn.Conv2d(1, 1, (1, hidden_dim * 2 - self.k + 1), 1, 0)
            self.w_sc_i = nn.ModuleDict(w_sc_i)
            self.w_sc_t = nn.ModuleDict(w_sc_t)
            self.w_cc_1 = nn.Conv2d(1, 1, (input_dim, hidden_dim * 2 - self.k + 1), 1, 0)
            self.w_cc_2 = nn.Conv2d(1, 1, (input_dim, hidden_dim * 2 - self.k + 1), 1, 0)

        # hybrid_focus_procedure
        self.eps = 0.00001
        self.sharpening_factor = 2.0

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.attn_fusion_1 = nn.Conv2d(2, 16, 5, 1, 2)
        self.attn_fusion_2 = nn.Conv2d(16, 32, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        if isinstance(lossfun, nn.CrossEntropyLoss):  # Multi-class classification
            self.affine = nn.Linear(int(hidden_dim * 2 / 4) * int(input_dim / 4) * 32, num_classes)
        elif isinstance(lossfun, nn.BCEWithLogitsLoss):
            self.affine = nn.Linear(int(hidden_dim * 2 / 4) * int(input_dim / 4) * 32, 1)
        self.softmax = nn.Softmax(dim=1)
        self.loss = lossfun
        self.phase = phase


    def location_based_attention(self, hidden_matrix):
        # hidden_matrix: [batch, view, time, dim]
        _, _, time, _ = hidden_matrix.shape
        e_v = []
        for v in range(self.input_dim):
            e_t = []
            for t in range(time-1):
                e_t.append(self.w_e(hidden_matrix[:,v,t]).squeeze(1))
            e_v.append(torch.stack(e_t).permute(1, 0))
        return torch.stack(e_v).permute(1, 0, 2)

    def context_based_attention(self, hidden_matrix):
        # hidden_matrix: [batch, view, time, dim]
        batch_size, _, time_point, _ = hidden_matrix.shape
        ''' local self-context '''
        h_t = []
        for v in range(self.input_dim):
            h_t.append(self.w_sc_t[str(v)](hidden_matrix[:,v,-1].view(hidden_matrix.shape[0],1,1,-1)))
        # h_t: [batch, 1, 1, k]
        h_t = torch.sum(torch.stack(h_t), dim=0)

        e_ti = []
        for t in range(time_point - 1):
            ''' target self-context '''
            h_i = []
            for v in range(self.input_dim):
                h_i.append(self.w_sc_i[str(v)](hidden_matrix[:,v,t].view(batch_size,1,1,-1)))
            h_i = torch.sum(torch.stack(h_i), dim=0)

            ''' cross-context '''
            h_ti = \
                self.w_cc_1(hidden_matrix[:,:,-1].unsqueeze(1)) + \
                self.w_cc_2(hidden_matrix[:,:,t].unsqueeze(1))

            ''' previous score information from attention matrix '''
            # hidden_matrix: [batch, view, time, dim]
            if t == 0:
                attn_weights = torch.sum(hidden_matrix[:,:,0] * hidden_matrix[:,:,-1], 2)
            else:
                attn_weights = torch.sum(hidden_matrix[:,:,t-1] * hidden_matrix[:,:,-1], 2)
            soft_attn_weights = self.softmax(attn_weights).unsqueeze(1).unsqueeze(2)

            ''' merge '''
            e_ti.append(torch.tanh(self.w_a(h_t) + self.w_b(h_i) + self.w_c(h_ti) + self.w_d(soft_attn_weights)))

        # e_ti: [batch, view, time]
        e_ti = torch.stack(e_ti).view(time_point-1, batch_size, self.input_dim).permute(1, 2, 0)
        return e_ti

    def hybrid_focus_procedure(self, energy_matrix):
        batch, view, time = energy_matrix.shape
        # e_sig: [batch, view, time]
        e_sig = torch.sigmoid(energy_matrix)

        beta = []
        for t in range(time):
            beta_t = e_sig[:,:,t]
            beta_t_min = torch.min(beta_t, dim=1)[0].unsqueeze(1)
            beta_t_max = torch.max(beta_t, dim=1)[0].unsqueeze(1)
            beta.append(torch.div(beta_t - beta_t_min, beta_t_max - beta_t_min + self.eps))
        beta = torch.stack(beta).permute(1, 2, 0)

        # e_hat: [batch, view, time]
        e_hat = []
        for v in range(self.input_dim):
            e_sig_v = e_sig[:,v]
            e_sig_v_min = torch.min(e_sig_v, dim=1)[0].unsqueeze(1)
            e_sig_v_max = torch.max(e_sig_v, dim=1)[0].unsqueeze(1)
            e_hat.append(torch.div(e_sig_v - e_sig_v_min, e_sig_v_max - e_sig_v_min + self.eps))
        e_hat = torch.stack(e_hat).permute(1, 0, 2)
        e_hat = torch.mul(e_hat, beta)
        e_hat = torch.mul(e_hat, self.sharpening_factor)

        # attention_matrix: [batch, view, time]
        attention_matrix = self.softmax(e_hat.reshape(batch, view*time)).view(batch, view, time)
        return attention_matrix

    def hybrid_focus_procedure_original(self, energy_matrix):
        batch, view, time = energy_matrix.shape
        # beta_top: [batch, view]
        beta_top = torch.sum(self.relu(energy_matrix), dim=2)
        # beta: [batch, view]
        beta = torch.div(beta_top - torch.min(beta_top, dim=1)[0].unsqueeze(1),
                         torch.max(beta_top, dim=1)[0].unsqueeze(1) - torch.min(beta_top, dim=1)[0].unsqueeze(1) + self.eps)

        # e_sig: [batch, view, time]
        e_sig = torch.sigmoid(energy_matrix)
        # e_hat: [batch, view, time]
        e_hat = []
        for v in range(self.input_dim):
            e_sig_v = e_sig[:,v]
            e_sig_v_min = torch.min(e_sig_v, dim=1)[0].unsqueeze(1)
            e_sig_v_max = torch.max(e_sig_v, dim=1)[0].unsqueeze(1)
            e_hat.append(torch.div(e_sig_v - e_sig_v_min, e_sig_v_max - e_sig_v_min + self.eps))
        e_hat = torch.stack(e_hat).permute(1, 0, 2)
        e_hat = torch.mul(e_hat, beta.unsqueeze(2)) # [batch, view, time] x [batch, view, 1]
        e_hat = torch.mul(e_hat, self.sharpening_factor)

        # attention_matrix: [batch, view, time]
        attention_matrix = self.softmax(e_hat.reshape(batch, view*time)).view(batch, view, time)
        return attention_matrix

    def attentional_feature_fusion(self, hidden_matrix, attention_matrix):
        # context_matrix: [batch, view, dim]
        context_matrix = torch.matmul(attention_matrix.unsqueeze(2), hidden_matrix[:,:,:-1]).squeeze(2)
        # cat_matrix: [batch, channel, view, dim]
        cat_matrix = torch.stack([hidden_matrix[:,:,-1,:], context_matrix]).permute(1, 0, 2, 3)
        logit = self.pool(self.relu(self.bn1(self.attn_fusion_1(cat_matrix))))
        logit = self.pool(self.relu(self.bn2(self.attn_fusion_2(logit))))
        #logit = self.pool(self.relu(self.attn_fusion_1(cat_matrix)))
        #logit = self.pool(self.relu(self.attn_fusion_2(logit)))
        logit = self.affine(logit.view(logit.size()[0], -1))
        return logit

    def forward(self, input):
        hidden_matrix = []
        for v in range(self.input_dim):
            # lstm_out: [batch, time, dim]
            lstm_out, _ = self.bgru(input[:,:,v].unsqueeze(2))
            hidden_matrix.append(lstm_out)

        # hidden_matrix: [batch, view, time, dim]
        hidden_matrix = torch.stack(hidden_matrix).permute(1, 0, 2, 3)
        # energy_matrix: [batch, view, time]
        energy_matrix = self.multi_view_attention(hidden_matrix)
        # attention_matrix: [batch, view, time]
        attention_matrix = self.hybrid_focus_procedure(energy_matrix)
        # print('min: {}, max: {}'.format(attention_matrix.min(), attention_matrix.max()))
        # context_matrix: [batch, view, dim]
        logit = self.attentional_feature_fusion(hidden_matrix, attention_matrix)

        if self.phase == 'test':
            return logit, attention_matrix
        else:
            return logit
