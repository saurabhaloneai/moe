import torch 
from torch import nn
from typing import Optional, Any , dict 
import torch.nn.functional as F  



class Router(nn.Module):
    
    def __init__(self, input_dim: int , num_experts: int, k: int= 1, 
                 capacity_factor: float = 1.0 , noise_eps: float = 1e-2):
        
        
        super().__init__()
        
        self.input_dim = input_dim 
        self.num_experts = num_experts
        self.k = k
        self.capacity_factor = capacity_factor
        self.noise_eps = noise_eps 
        
        
        ## let write the logic here 
        ## router is basically the NN which will 
        ## select the expert acc to what porb it gets for given experts and it will 
        ## selecet acc k = 2 then it will select the 2 experts 
        
        self.router = nn.Linear(input_dim,num_experts,bias=False)
        
        # expert capcity : this will handle that expert should have some 
        # caopacity it should not get more thant that otherwise it will overfit 
        # math : batch_size * capacity_factor * (k / num_experts)
        
        ## why we take the batch_size and why we using lambda here 
        ## when we use this  
        self.capacity = lambda batch_size : int(batch_size * capacity_factor * (k / num_experts))
        
        
    def _compute_routing_score(self,x: torch.Tensor)-> torch.Tensor:
        
        ## why use func name with _ ? 
        ## adding noise 
        
        if self.trainig:
            noise = torch.randn_like(x) * self.noise_eps
            x = x + noise
        
        return self.rounter(x)
            
            
    def foward(self, x: torch.Tensor) :
        
        batch_size = x.shape[0]
        
        routing_scores = self._compute_routing_score(x)
        
        ## gettting top-k weights and indices 
        
        routing_weights, routing_indices = torch.topk(routing_scores, self.k, dim = -1) # don't why dim = -1 (last dim)
        
        # print(routing_weights, routing_indices)
        # print(routing_indices.shape)
        
        ## compute load balancing loss during trianing 
        
        aux_loss = None 
        
        if self.training:
            
            expert_counts = torch.zeros(self.num_experts, device=x.device)
            
            for idx in routing_indices.view(-1):
                expert_counts[idx] += 1 
                
            target_count = torch.ones_like(expert_counts) * (batch_size * self.k / self.num_experts)
            
            aux_loss = F.mse_loss(expert_counts, target_count)
            
        
        return routing_weights, routing_indices, aux_loss
        
            
            
        
        
        