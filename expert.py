import torch 
from torch import nn
from typing import Dict, Optional, Any

## Expert base class 

class ExpertBase(nn.Module):
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[Any] = None):
        
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.hidden_dim = hidden_dim or 4 * input_dim # dense 
        
        self.network = nn.Sequential(
            
            nn.Linear(input_dim, self.hidden_dim),  # the hidden_dim can be hidden_dim or 4 * input_dim so that's why self
            nn.ReLU(),
            nn.Linear(self.hidden_dim, output_dim)
        )
        
        
    def forward(self, x): # how this fucntin work internally ? 
        return self.network(x)
    
    def get_config(self):
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_dim': self.hidden_dim
        }
        
  
        
## Extended class cause we want multiple expert 


class ExtenedExpert(ExpertBase):
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: Optional[Any] = None, num_layers=2, dropout=0.2):
        
        super().__init__(input_dim, output_dim, hidden_dim)
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        ## this we create multiple layer 
        
        layers = []
        current_dim = input_dim 
        
        
        for _ in range(num_layers-1):
            
            layers.extend([
                nn.Linear(current_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            
            current_dim = self.hidden_dim # we want whatever the current dim it have 
            
        
        ## add final layer
        
        layers.append(nn.Linear(current_dim, output_dim))
        
        ## pass to the network 
        
        self.network = nn.Sequential(*layers)
        
        
    def get_config(self):
        
        config = super().get_config()
        config.update({
            'num_layers': self.num_layers,
            'dropout': self.dropout 
        })
            
        return config 
        
           
# check = ExtenedExpert(input_dim=10, output_dim=4, hidden_dim=10)

# x = torch.randn(1, 10)
# out = check(x)

# print(out.shape)