import torch 
import torch.nn as nn 
import torch.nn.functional as F


class minLSTM(nn.Module):
    def __init__(self, input_size:int,hidden_size:int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_f = nn.Linear(input_size,hidden_size)
        self.linear_i = nn.Linear(input_size,hidden_size)
        self.linear_h = nn.Linear(input_size,hidden_size)
    
    @staticmethod
    def g(x:torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, x+0.5, torch.sigmoid(x))
    
    @staticmethod
    def log_g(x:torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, (F.relu(x)+0.5).log(),-F.softplus(-x))
    
    @staticmethod
    def parallel_scan_log(log_coeffs:torch.Tensor, log_values:torch.Tensor) -> torch.Tensor:
        # log_coeffs: (batch_size, seq_len, input_size)
        # log_values: (batch_size, seq_len + 1, input_size)
        a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0))
        log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
        log_h = a_star + log_h0_plus_b_star
        return torch.exp(log_h)[:, 1:]

    def forward(self, x:torch.Tensor, h_0:torch.Tensor=None) -> torch.Tensor:
        # x: (batch_size, seq_len, input_size)
        # h_0: (batch_size, 1, hidden_size)
        if(h_0 is None):
            h_0 = torch.zeros((x.size(0),1,self.hidden_size),device=x.device)

        diff = F.softplus(-self.linear_f(x)) - F.softplus(-self.linear_i(x))
        log_f = -F.softplus(diff)
        log_i = -F.softplus(-diff)
        log_h_0 = self.log_g(h_0) 
        log_tilde_h = self.log_g(self.linear_h(x))
        h = self.parallel_scan_log(log_f,torch.cat([log_h_0, log_i + log_tilde_h], dim=1))
        return h

    def sequential_forward(self, x_t:torch.Tensor, h_prev:torch.Tensor=None) -> torch.Tensor:
        # x_t: (batch_size, input_size)
        # h_prev: (batch_size, hidden_size)
        if(h_prev is None):
            h_prev = self.g(torch.zeros((x_t.size(0),self.hidden_size),device=x_t.device))

    
        f_t = torch.sigmoid(self.linear_f(x_t))
        i_t = torch.sigmoid(self.linear_i(x_t))
        tilde_h_t = self.g(self.linear_h(x_t))
        f_prime_t = f_t / (f_t + i_t)
        i_prime_t = i_t / (f_t + i_t)
        h_t = f_prime_t * h_prev + i_prime_t * tilde_h_t
        return h_t
    

if __name__ == "__main__":
    batch_size = 3
    seq_size = 2
    input_size = 5 
    hidden_size = 6
    x = torch.randn(batch_size,seq_size,input_size)

    model = minLSTM(input_size,6)
    print(model(x))

    output_list = []
    ht = None
    for i in range(x.size(1)):
        ht = model.sequential_forward(x[:,i,:],ht)
        output_list.append(ht.unsqueeze(dim=1))
    
    print(torch.cat(output_list,dim=1))
