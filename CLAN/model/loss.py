import torch
import numpy as np
import torch
import torch.nn as nn

def get_similarity_two_matrix(output1, output2, chunk=2, multi_gpu=False):
    '''
        Compute similarity matrix
        - outputs: (B', d) tensor for B' = B * chunk
        - sim_matrix: (B', B') tensor
    '''

    sim_matrix = torch.mm(output1, output2.t())  # (B', d), (d, B') -> (B', B')

    return sim_matrix

def NT_xent(sim_matrix, a, temperature=0.5, chunk=2, eps=1e-8):
    '''
    Compute NT_xent loss with groups and individual pairings.
    - sim_matrix: (2*B, 2*B) tensor
    - a: group size
    '''
    device = sim_matrix.device
    B = sim_matrix.size(0) // 2  # B = (2*B) / 2

    eye = torch.eye(2 * B).to(device)
    sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)

    # Create mask for positive pairs between each sample and its counterpart
    basic_mask = torch.zeros_like(sim_matrix).to(device)
    basic_mask[:B, B:] = torch.eye(B).to(device)  # [0:B]와 [B:2*B] 사이의 양의 쌍
    basic_mask[B:, :B] = torch.eye(B).to(device)  # 대칭적으로 양의 쌍 설정

    # Create mask for each group
    group_mask = torch.zeros_like(sim_matrix).to(device)
    for i in range(0, B, a):
        group_mask[i:i + a, B + i:B + i + a] = 1  # 그룹 내에서 양의 쌍으로 설정
        group_mask[B + i:B + i + a, i:i + a] = 1  # 대칭적으로 설정

    # Combine the masks to include both types of positive pairings
    combined_mask = basic_mask + group_mask

    # Calculate the denominator (normalization)
    denom = torch.sum(sim_matrix, dim=1, keepdim=True)

    # Apply combined mask to consider both individual and group positive pairs
    sim_matrix = -torch.log((sim_matrix * combined_mask) / (denom + eps) + eps)

    # Compute the loss only for the masked positive pairs
    loss = torch.sum(sim_matrix * combined_mask) / combined_mask.sum()

    return loss

def NT_xent_TF(sim_matrix, temperature=0.5, chunk=2, eps=1e-8):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''
    device = sim_matrix.device    

    B = sim_matrix.size(0) // chunk  # B = B' / chunk
    eye = torch.eye(B * chunk).to(device)  # (B', B')
    sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)  # remove diagonal

    #print("B", B)
    
    denom = torch.sum(sim_matrix, dim=1, keepdim=True)
    sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)  # loss matrix
    

    # normal & orginal closer and shifted & shifted closer / negative pairs are not calculated 
    loss = torch.sum(sim_matrix[:].diag())  / (B)

    return loss

def Supervised_NT_xent(sim_matrix, labels, temperature=0.5, chunk=2, eps=1e-8, multi_gpu=False):
    '''
        Compute NT_xent loss
        - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
    '''

    device = sim_matrix.device

    labels = labels.repeat(2)

    logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
    sim_matrix = sim_matrix - logits_max.detach()

    B = sim_matrix.size(0) // chunk  # B = B' / chunk

    eye = torch.eye(B * chunk).to(device)  # (B', B')
    sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)  # remove diagonal

    denom = torch.sum(sim_matrix, dim=1, keepdim=True)
    sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)  # loss matrix

    labels = labels.contiguous().view(-1, 1)
    Mask = torch.eq(labels, labels.t()).float().to(device)
    #Mask = eye * torch.stack([labels == labels[i] for i in range(labels.size(0))]).float().to(device)
    Mask = Mask / (Mask.sum(dim=1, keepdim=True) + eps)

    loss = torch.sum(Mask * sim_matrix) / (2 * B)

    return loss

def get_similarity_matrix(outputs, chunk=2, multi_gpu=False):
    sim_matrix = torch.mm(outputs, outputs.t())  # (B', d), (d, B') -> (B', B')
    return sim_matrix


# def NT_xent(sim_matrix, temperature=0.5, chunk=2, eps=1e-8):
#     '''
#         Compute NT_xent loss
#         - sim_matrix: (B', B') tensor for B' = B * chunk (first 2B are pos samples)
#     '''

#     device = sim_matrix.device

#     B = sim_matrix.size(0) // chunk  # B = B' / chunk

#     eye = torch.eye(B * chunk).to(device)  # (B', B')
#     sim_matrix = torch.exp(sim_matrix / temperature) * (1 - eye)  # remove diagonal

#     denom = torch.sum(sim_matrix, dim=1, keepdim=True)
#     sim_matrix = -torch.log(sim_matrix / (denom + eps) + eps)  # loss matrix

#     loss = torch.sum(sim_matrix[:B, B:].diag() + sim_matrix[B:, :B].diag()) / (2 * B)

#     return loss
    
