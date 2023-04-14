import torch
import torch.nn as nn
from torch.nn.functional import softmax
import logging
from crowd_nav.policy.cadrl import mlp
from crowd_nav.policy.multi_human_rl import MultiHumanRL


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims, attention_dims, with_global_state,
                 cell_size, cell_num, empty_peds_stub: bool):
        super().__init__()
        self.self_state_dim = self_state_dim
        self.global_state_dim = mlp1_dims[-1]
        self.mlp1 = mlp(input_dim, mlp1_dims, last_relu=True)
        self.mlp2 = mlp(mlp1_dims[-1], mlp2_dims)
        self.with_global_state = with_global_state
        if with_global_state:
            self.attention = mlp(mlp1_dims[-1] * 2, attention_dims)
        else:
            self.attention = mlp(mlp1_dims[-1], attention_dims)
        self.cell_size = cell_size
        self.cell_num = cell_num
        mlp3_input_dim = mlp2_dims[-1] + self.self_state_dim
        self.mlp3 = mlp(mlp3_input_dim, mlp3_dims)
        self.attention_weights = None

        self.empty_peds_stub = nn.Parameter(torch.zeros(mlp2_dims[1])) if empty_peds_stub else None
        self.mlp1_dim = mlp1_dims[1]
        self.mlp2_dim = mlp2_dims[1]

    def forward(self, state):
        """
        First transform the world coordinates to self-centric coordinates and then do forward computation

        :param state: tensor of shape (batch_size, # of humans, length of a rotated state)
        :return:
        """
        self_state = state[:, 0, :self.self_state_dim]

        # TODO: May not work with proper vectorization
        has_peds = True
        # if self.empty_peds_stub is not None:
        #     state = state[state[:, :, -3] > 0.]
        #     if len(state) == 0:
        #         has_peds = False
        #     else:
        #         state = state.unsqueeze(0)

        visibility_mask = state[:, :, -3] > 0
        visibility_mask_flatten_1 = torch.tile(visibility_mask.reshape(-1).unsqueeze(1), (1, self.mlp1_dim))
        visibility_mask_flatten_2 = torch.tile(visibility_mask.reshape(-1).unsqueeze(1), (1, self.mlp2_dim))

        if has_peds:
            size = state.shape
            mlp1_output = self.mlp1(state.view((-1, size[2])))
            mlp2_output = self.mlp2(mlp1_output)

            mlp1_output = mlp1_output * visibility_mask_flatten_1
            mlp2_output = mlp2_output * visibility_mask_flatten_2

            if self.with_global_state:
                # compute attention scores
                global_state = torch.mean(mlp1_output.view(size[0], size[1], -1), 1, keepdim=True)
                global_state = global_state.expand((size[0], size[1], self.global_state_dim)).\
                    contiguous().view(-1, self.global_state_dim)
                attention_input = torch.cat([mlp1_output, global_state], dim=1)
            else:
                attention_input = mlp1_output
            scores = self.attention(attention_input).view(size[0], size[1], 1).squeeze(dim=2)

            # masked softmax
            # weights = softmax(scores, dim=1).unsqueeze(2)
            scores_exp = torch.exp(scores) * (scores != 0).float()
            weights = (scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)).unsqueeze(2)
            self.attention_weights = weights[0, :, 0].data.cpu().numpy()

            # output feature is a linear combination of input features
            features = mlp2_output.view(size[0], size[1], -1)
            # for converting to onnx
            # expanded_weights = torch.cat([torch.zeros(weights.size()).copy_(weights) for _ in range(50)], dim=2)
            weighted_feature = torch.sum(torch.mul(weights, features), dim=1)
        else:
            weighted_feature = torch.tile(self.empty_peds_stub, (self_state.shape[0], 1))

        # concatenate agent's state with global weighted humans' state
        joint_state = torch.cat([self_state, weighted_feature], dim=1)
        value = self.mlp3(joint_state)
        return value


class SARL(MultiHumanRL):
    def __init__(self):
        super().__init__()
        self.name = 'SARL'

    def configure(self, config):
        self.set_common_parameters(config)
        mlp1_dims = [int(x) for x in config.get('sarl', 'mlp1_dims').split(', ')]
        mlp2_dims = [int(x) for x in config.get('sarl', 'mlp2_dims').split(', ')]
        mlp3_dims = [int(x) for x in config.get('sarl', 'mlp3_dims').split(', ')]
        attention_dims = [int(x) for x in config.get('sarl', 'attention_dims').split(', ')]
        self.with_om = config.getboolean('sarl', 'with_om')
        with_global_state = config.getboolean('sarl', 'with_global_state')
        empty_peds_stub = config.getboolean('sarl', 'empty_peds_stub')
        self.model = ValueNetwork(self.input_dim(), self.self_state_dim, mlp1_dims, mlp2_dims, mlp3_dims,
                                  attention_dims, with_global_state, self.cell_size, self.cell_num,
                                  empty_peds_stub)
        self.multiagent_training = config.getboolean('sarl', 'multiagent_training')
        if self.with_om:
            self.name = 'OM-SARL'
        logging.info('Policy: {} {} global state'.format(self.name, 'w/' if with_global_state else 'w/o'))

    def get_attention_weights(self):
        return self.model.attention_weights
