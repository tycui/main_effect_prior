import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def seperating(x_train, y_train):
    """
    seperate the training set into a training set and a validation set
    """
    x = x_train.numpy()
    y = y_train.numpy()
    np.random.seed(129)
    msk = np.random.rand(len(x)) < 0.8
    x_train = torch.tensor(x[msk, :], dtype=torch.float)
    x_val = torch.tensor(x[~msk, :], dtype=torch.float)
    y_train = torch.tensor(y[msk, :].reshape(-1, 1), dtype=torch.float)
    y_val = torch.tensor(y[~msk, :].reshape(-1, 1), dtype=torch.float)
    return x_train, y_train, x_val, y_val

class Linearlayer(nn.Module):
    """
    Class of linear layer for constructing a NN
    """
    def __init__(self, input_dim, output_dim):
        super(Linearlayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        scale = 1. * np.sqrt(6. / (input_dim + output_dim))
        # approximated posterior
        self.w = nn.Parameter(torch.Tensor(self.input_dim, self.output_dim).uniform_(-scale, scale))
        self.bias = nn.Parameter(torch.Tensor(self.output_dim).uniform_(-scale, scale))

    def forward(self, x, training):
        if training:
            # forward passing
            output = torch.mm(x, self.w) + self.bias
            return output
        else:
            output = torch.mm(x, self.w) + self.bias
            return output


class NNLRModule(nn.Module):
    def __init__(self, num_feature, num_nodes=[20,10]):
        super(NNLRModule, self).__init__()
        self.num_feature = num_feature
        self.num_nodes = num_nodes
        self.num_layers = len(num_nodes)

        list_of_layers = [Linearlayer(num_feature, num_nodes[0])]
        list_of_layers += [Linearlayer(num_nodes[i], num_nodes[i + 1]) for i in range(len(num_nodes) - 1)]
        list_of_layers += [Linearlayer(num_nodes[-1], 1)]
        
        self.skip_layer = Linearlayer(num_feature, 1)
        self.layers = nn.ModuleList(list_of_layers)
        self.activation_fn = nn.ReLU()

    def forward(self, x):
        current_layer = x
        y_lr = self.skip_layer(x, self.training)

        for i, j in enumerate(self.layers):
            if i != self.num_layers:
                current_layer = self.activation_fn(j(current_layer, self.training))
            else:
                current_layer = j(current_layer, self.training)
        
        return current_layer + y_lr, self.reg_skip(), self.reg_layers()
    
    def reg_skip(self):
        return torch.norm(self.skip_layer.w, 1)
    
    def reg_layers(self):
        reg = 0
        for i in self.layers:
            reg += torch.norm(i.w, 1)
        return reg    



class NNtraining(object):
    def __init__(self, 
                 model, 
                 learning_rate=0.001, 
                 batch_size=10000, 
                 num_epoch=200, 
                 early_stop_patience = 20,
                 reg_weight_main = 0.0,
                 reg_weight_nn = 0.0,
                 reg_weight_info = 0.0,
                 use_cuda=False,
                 use_early_stopping = False):
        
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.best_val = 1e5
        self.early_stop_patience = early_stop_patience
        self.epochs_since_update = 0  # used for early stopping
        self.reg_weight_main = reg_weight_main
        self.reg_weight_nn = reg_weight_nn
        self.reg_weight_info = reg_weight_info
        self.use_early_stopping = use_early_stopping
        
        self.use_cuda = use_cuda
        if use_cuda:
            self.model.cuda()
        
    def training(self, x, y, xval, yval,
                 main_effect = 0,
                 main_effect_std = 1.,
                 marginal = False):
        main_effect = torch.tensor(main_effect, dtype = torch.float32)
        
        if type(main_effect_std) == float:
            main_effect_std = torch.tensor(main_effect_std, dtype = torch.float32)
            print('std is not used in training')
        else:
            main_effect_std = torch.tensor(10 * main_effect_std, dtype = torch.float32) # in cross-validation we increased the std
            
        print('Compute the gram matrix')    
        gram = torch.matmul(torch.transpose(x, 0, 1), x); eps = 1e-3
        if marginal:
            gram = torch.diag(torch.diag(gram))
        self.multiplier = torch.matmul(torch.inverse(gram+torch.diag(eps * torch.ones(gram.shape[0]))), torch.transpose(x, 0, 1))
        if self.use_cuda:
            self.multiplier = self.multiplier.cuda()
        del gram
        print('Compute the gram matrix done!')  
     
        parameters = set(self.model.parameters())
        optimizer = optim.Adam(parameters, lr=self.learning_rate, eps=1e-3)
        criterion = nn.MSELoss()
        train_dl = DataLoader(TensorDataset(x, y), batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.num_epoch):
            for x_batch, y_batch in train_dl:
                if self.use_cuda:
                    x_batch = x_batch.cuda()
                    y_batch = y_batch.cuda()
                optimizer.zero_grad()
                self.model.train()
                # calculate the training loss
                output, reg_main, reg_NN = self.model(x_batch)
                total_reg = self.reg_weight_main * reg_main + self.reg_weight_nn * reg_NN
                if self.reg_weight_info < 1e-8:
                    loss = criterion(y_batch, output) + total_reg
                else:
                    loss = criterion(y_batch, output) + self.reg_weight_info * self.main_effect_reg_l1(x, main_effect, main_effect_std) + total_reg
                # backpropogate the gradient
                loss.backward()
                # optimize with SGD
                optimizer.step()
            
            train_mse, train_pve = self.build_evaluation(x, y)
            val_mse, val_pve = self.build_evaluation(xval, yval)

            if self.use_early_stopping:
                early_stop = self._early_stop(val_mse)
                if early_stop:
                    break
                    
        print('>>> Epoch {:5d}/{:5d} | train_mse={:.5f} | val_mse={:.5f} | train_pve={:.5f} | val_pve={:.5f}'.format(epoch,
                                                                                                                         self.num_epoch, 
                                                                                                                         train_mse, 
                                                                                                                         val_mse, 
                                                                                                                         train_pve, 
                                                                                                                         val_pve))
        
        return epoch + 1 - self.early_stop_patience
    
    
    def training_continue(self, x, y, xval, yval, 
                          target_pve = 0,
                          main_effect = 0,
                          main_effect_std = 1.,
                          marginal = False):
        main_effect = torch.tensor(main_effect, dtype = torch.float32)
        
        if type(main_effect_std) == float:
            main_effect_std = torch.tensor(main_effect_std, dtype = torch.float32)
            print('std is not used in training')
        else:
            main_effect_std = torch.tensor(10 * main_effect_std, dtype = torch.float32) # in cross-validation we increased the std

            
        parameters = set(self.model.parameters())
        optimizer = optim.Adam(parameters, lr=self.learning_rate, eps=1e-3)
        criterion = nn.MSELoss()
        train_dl = DataLoader(TensorDataset(x, y), batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.num_epoch):
            for x_batch, y_batch in train_dl:
                if self.use_cuda:
                    x_batch = x_batch.cuda()
                    y_batch = y_batch.cuda()
                optimizer.zero_grad()
                self.model.train()
                # calculate the training loss
                output, reg_main, reg_NN = self.model(x_batch)
                total_reg = self.reg_weight_main * reg_main + self.reg_weight_nn * reg_NN
                if self.reg_weight_info < 1e-8:
                    loss = criterion(y_batch, output) + total_reg
                else:
                    loss = criterion(y_batch, output) + self.reg_weight_info * self.main_effect_reg_l1(x, main_effect, main_effect_std) + total_reg
                # backpropogate the gradient
                loss.backward()
                # optimize with SGD
                optimizer.step()
            
            train_mse, train_pve = self.build_evaluation(x, y)
            val_mse, val_pve = self.build_evaluation(xval, yval)

            
        # Train the model until validation PVE is higher than the target PVE (train PVE)
            if val_pve > target_pve or epoch > 5000:
                break
        print('>>> Epoch {:5d}/{:5d} | train_mse={:.5f} | val_mse={:.5f} | train_pve={:.5f} | val_pve={:.5f}'.format(epoch,
                                                                                                                         self.num_epoch, 
                                                                                                                         train_mse, 
                                                                                                                         val_mse, 
                                                                                                                         train_pve, 
                                                                                                                         val_pve))
                
    def build_evaluation(self, x_test, y_test):
        criterion = nn.MSELoss()
        if self.use_cuda:
            x_test = x_test.cuda()
            y_test = y_test.cuda()
        self.model.eval()
        y_pred, _, _ = self.model(x_test)
        mse_eval = criterion(y_test, y_pred).detach()
        
        pve = (1. - torch.var(y_pred.view(-1) - y_test.view(-1)) / torch.var(y_test.view(-1))).detach() 
        return mse_eval, pve
    
    def _early_stop(self, val_loss):
        updated = False # flag
        current = val_loss
        best = self.best_val
        improvement = (best - current) / best
        
        if improvement > 0.00:
            self.best_val = current
            updated = True
        
        if updated:
            self.epochs_since_update = 0
        else:
            self.epochs_since_update += 1
            
        return self.epochs_since_update > self.early_stop_patience
  

    def main_effect_reg_l1(self, x_batch, main_effect, main_effect_std):

        y_batch_pred, _, _ = self.model(x_batch)
        self.coef = torch.matmul(self.multiplier, y_batch_pred)
        reg = torch.sum(torch.abs(self.coef - main_effect) / main_effect_std)
        return reg  
