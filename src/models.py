import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# ===== Basic temporal forecasting model =====
class TSFModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, model_type='lstm', num_layers=4, dropout=0.3):
        super(TSFModel, self).__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        if model_type == 'lstm':
            self.model = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0  # no dropout if only one layer
            )
        elif model_type == 'cnn':
            # implement multiple cnn layers
            cnn_layers = []
            for _ in range(num_layers):
                cnn_layers.extend([
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
            self.model = nn.Sequential(*cnn_layers)
        elif model_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            self.model = nn.TransformerEncoder(encoder_layer, num_layers)

        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_length, features]
        x = self.input_proj(x)  # [batch_size, seq_length, hidden_dim]

        if self.model_type == 'lstm':
            lstm_out, _ = self.model(x)  # lstm_out: [batch_size, seq_length, hidden_dim]
            x = lstm_out[:, -1]  # use last sequence output [batch_size, hidden_dim]
        elif self.model_type == 'cnn':
            # transform sequence to channel dimension
            x = x.transpose(1, 2)  # [batch_size, hidden_dim, seq_length]
            x = self.model(x)  # pass through all cnn layers
            # global average pooling over time dimension
            x = x.mean(dim=2)  # [batch_size, hidden_dim]
        else:  # transformer
            x = self.model(x)  # [batch_size, seq_length, hidden_dim]
            # use only the last sequence element
            x = x[:, -1]  # [batch_size, hidden_dim]

        out = self.output_projection(x)  # [batch_size, output_dim]

        if self.output_dim == 1:
            return out.unsqueeze(-1)  # [batch, 1]
        return out  # [batch, horizon]


# ===== Transformer =====
class SpatioTemporalTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, nhead=8, dropout=0.3):
        super(SpatioTemporalTransformer, self).__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # input feature projection with layer norm
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # spatial embedding with layer norm
        self.spatial_embedding = nn.Sequential(
            nn.Linear(2, hidden_dim),  # simpler architecture
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, 24, hidden_dim))
        nn.init.xavier_normal_(self.pos_encoding)  # changed from uniform to normal

        # transformer encoder with additional layer norm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,  # reduced from 4x to 2x
            dropout=dropout,
            batch_first=True,
            norm_first=True  # pre-norm instead of post-norm
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # output projection with layer norm
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x, spatial_info):
        # x: [batch_size, seq_length, features]
        # spatial_info: [batch_size, 2] (latitude, longitude)

        # project input features
        x = self.input_proj(x)  # [batch, seq, hidden]

        # add positional encoding
        x = x + self.pos_encoding[:, :x.size(1), :]

        # process spatial information
        spatial_enc = self.spatial_embedding(spatial_info)  # [batch, hidden]
        spatial_enc = spatial_enc.unsqueeze(1).expand(-1, x.size(1), -1)

        # combine temporal and spatial information
        x = x + spatial_enc

        # apply transformer encoder
        transformer_out = self.transformer(x)

        # use last output for prediction
        out_features = transformer_out[:, -1]

        # final output projection
        outputs = self.output_projection(out_features)

        if self.output_dim == 1:
            return outputs.unsqueeze(-1)  # [batch, 1]
        return outputs  # [batch, horizon]


# ===== GCN =====
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, dropout=0.3, temporal_model='lstm'):
        super(GCN, self).__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # edge feature processing
        self.edge_feature_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),  # 1 is the number of edge feature: height
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # GCN layers with skip connections
        self.gcn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # select temporal processing module
        if temporal_model == 'lstm':
            self.temporal_model = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                dropout=dropout
            )
        elif temporal_model == 'cnn':
            self.temporal_model = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        elif temporal_model == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            )
            self.temporal_model = nn.TransformerEncoder(encoder_layer, num_layers)

        self.temporal_type = temporal_model
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x, edge_index, edge_weights=None, edge_features=None):
        batch_size, seq_length, num_nodes, feat_dim = x.shape

        # process edge features if available
        if edge_features is not None:
            edge_attention = self.edge_feature_proj(edge_features).squeeze(-1)
            if edge_weights is not None:
                edge_weights = edge_weights * edge_attention
            else:
                edge_weights = edge_attention

        # GCN processing
        x = x.reshape(-1, feat_dim)
        x = self.input_proj(x)
        x = x.reshape(batch_size * seq_length, num_nodes, -1)

        h_list = []
        for t in range(seq_length):
            h = x[t:t + batch_size]
            h = h.reshape(-1, self.hidden_dim)

            # apply GCN layers with skip connections and edge weights
            for gcn_layer, batch_norm in zip(self.gcn_layers, self.batch_norms):
                identity = h
                h = gcn_layer(h, edge_index, edge_weight=edge_weights)
                h = batch_norm(h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                h = h + identity  # skip connection

            h = h.reshape(batch_size, num_nodes, -1)
            h_list.append(h)

        x = torch.stack(h_list, dim=1)

        # temporal processing
        if self.temporal_type == 'lstm':
            x = x.transpose(1, 2)
            x = x.reshape(-1, seq_length, self.hidden_dim)
            lstm_out, _ = self.temporal_model(x)
            x = lstm_out[:, -1]
        elif self.temporal_type == 'cnn':
            x = x.transpose(1, 2)
            x = x.reshape(-1, self.hidden_dim, seq_length)
            for layer in self.temporal_model:
                x = layer(x)
            x = x.mean(dim=2)
        else:  # transformer
            x = x.reshape(-1, seq_length, self.hidden_dim)
            transformer_out = self.temporal_model(x)
            x = transformer_out[:, -1]

        # output projection
        x = x.reshape(batch_size, num_nodes, -1)
        out = self.output_projection(x)
        if self.output_dim == 1:
            return out.unsqueeze(-1)  # [batch, nodes, 1]
        return out  # [batch, nodes, horizon]


# ===== STGCN =====
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(STGCNBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0, kernel_size // 2))
        self.gcn = GCNConv(out_channels, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (1, kernel_size), padding=(0, kernel_size // 2))
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_weights=None):
        # x: [batch, seq, nodes, channels]
        batch_size, seq_len, num_nodes, channels = x.shape

        # temporal convolution
        x = x.permute(0, 3, 2, 1)  # [batch, channels, nodes, time]
        x = self.conv1(x)

        # graph convolution with edge weights
        x = x.permute(0, 3, 2, 1)  # [batch, time, nodes, channels]
        x_gcn = []
        for t in range(x.size(1)):
            h = x[:, t]  # [batch, nodes, channels]
            h = h.reshape(-1, h.size(-1))  # [batch*nodes, channels]
            h = self.gcn(h, edge_index, edge_weight=edge_weights)  # edge_weights 사용
            h = h.view(batch_size, num_nodes, -1)  # [batch, nodes, channels]
            x_gcn.append(h)
        x = torch.stack(x_gcn, dim=1)  # [batch, time, nodes, channels]

        # final temporal convolution
        x = x.permute(0, 3, 2, 1)  # [batch, channels, nodes, time]
        x = self.conv2(x)
        x = self.batch_norm(x)
        x = self.relu(x)

        return x.permute(0, 3, 2, 1)  # [batch, time, nodes, channels]


class GatedSTGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate=2):
        super(GatedSTGCNBlock, self).__init__()
        kernel_size = 3
        padding = dilation_rate * (kernel_size - 1) // 2

        self.conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0, padding), dilation=(1, dilation_rate))
        self.gate = nn.Conv2d(in_channels, out_channels, (1, kernel_size), padding=(0, padding), dilation=(1, dilation_rate))
        self.gcn = GCNConv(out_channels, out_channels)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_weights=None, edge_features=None):
        batch_size, seq_len, num_nodes, channels = x.shape

        # temporal gated convolution
        x = x.permute(0, 3, 2, 1)
        conv_out = self.conv(x)
        gate_out = torch.sigmoid(self.gate(x))
        x = conv_out * gate_out

        # graph convolution with edge weights
        x = x.permute(0, 3, 2, 1)
        x_gcn = []
        for t in range(x.size(1)):
            h = x[:, t]
            h = h.reshape(-1, h.size(-1))
            h = self.gcn(h, edge_index, edge_weight=edge_weights)  # edge_weights 추가
            h = h.view(batch_size, num_nodes, -1)
            x_gcn.append(h)
        x = torch.stack(x_gcn, dim=1)

        # normalization and activation
        x = x.permute(0, 3, 2, 1)
        x = self.batch_norm(x)
        x = self.relu(x)

        return x.permute(0, 3, 2, 1)


class STGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.3, block_type='base'):
        super(STGCN, self).__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # select block type
        if block_type == 'base':
            Block = STGCNBlock
        elif block_type == 'gated':
            Block = GatedSTGCNBlock

        self.st_blocks = nn.ModuleList([
            Block(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3
            ) for _ in range(num_layers)
        ])

        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

        self.output_dim = output_dim

    def forward(self, x, edge_index, edge_weights=None, edge_features=None):
        # x: [batch_size, seq_length, num_nodes, features]
        batch_size, seq_length, num_nodes, feat_dim = x.shape

        # project input features
        x = x.reshape(-1, feat_dim)  # [batch*seq*nodes, features]
        x = self.input_proj(x)  # [batch*seq*nodes, hidden]
        x = x.reshape(batch_size, seq_length, num_nodes, -1)  # [batch, seq, nodes, hidden]

        # process with ST-GCN blocks
        for st_block in self.st_blocks:
            x = st_block(x, edge_index, edge_weights)

        # global average pooling over sequence dimension
        x = x.mean(dim=1)  # [batch, nodes, hidden]

        # final projection
        out = self.output_projection(x)

        if self.output_dim == 1:
            return out.unsqueeze(-1)  # [batch, nodes, 1]
        return out  # [batch, nodes, horizon]


def get_model(model_type, input_dim, hidden_dim, output_dim, num_layers=4, num_nodes=37, embed_dim=10, seq_length=7,
              dropout=0.3):
    if model_type == 'LSTM':
        return TSFModel(input_dim, hidden_dim, output_dim, model_type='lstm', num_layers=num_layers, dropout=dropout)
    elif model_type == 'CNN':
        return TSFModel(input_dim, hidden_dim, output_dim, model_type='cnn', num_layers=num_layers, dropout=dropout)
    elif model_type == 'Transformer':
        return TSFModel(input_dim, hidden_dim, output_dim, model_type='transformer', num_layers=num_layers, dropout=dropout)
    elif model_type == 'STTransformer':
        return SpatioTemporalTransformer(input_dim, hidden_dim, output_dim, num_layers=num_layers, nhead=8, dropout=dropout)
    elif model_type == 'GCNLSTM':
        return GCN(input_dim, hidden_dim, output_dim, num_layers, dropout=dropout, temporal_model='lstm')
    elif model_type == 'GCNCNN':
        return GCN(input_dim, hidden_dim, output_dim, num_layers, dropout=dropout, temporal_model='cnn')
    elif model_type == 'GCNTransformer':
        return GCN(input_dim, hidden_dim, output_dim, num_layers, dropout=dropout, temporal_model='transformer')
    elif model_type == 'STGCNLSTM':
        return STGCN(input_dim, hidden_dim, output_dim, num_layers, dropout=dropout, block_type='base')
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # set parameters
    batch_size = 8
    seq_length = 4
    num_nodes = 35
    input_dim = 15
    hidden_dim = 128
    output_dim = 2
    num_layers = 4
    embed_dim = 10

    # set sample data
    x = torch.randn(batch_size, seq_length, num_nodes, input_dim)
    x_tsf = x.view(batch_size * num_nodes, seq_length, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_nodes * (num_nodes - 1)))
    spatial_info = torch.randn(batch_size * num_nodes, 2)
    adj = torch.ones(batch_size, num_nodes, num_nodes)

    print("\n=== Testing Time Series Models ===")
    for model_type in ['LSTM', 'CNN', 'Transformer']:
        model = get_model(
            model_type, input_dim, hidden_dim, output_dim,
            num_layers=num_layers, num_nodes=num_nodes,
            embed_dim=embed_dim, seq_length=seq_length
        )
        print(f"\n{model_type}:")
        print(f"Input shape: {x.shape} -> Reshaped to: {x_tsf.shape}")
        y = model(x_tsf)
        print(f"Output shape: {y.shape}")  # [batch*num_nodes, output_dim]

    print("\n=== Testing Spatiotemporal Models ===")
    # STTransformer
    model = get_model(
        'STTransformer', input_dim, hidden_dim, output_dim,
        num_layers=num_layers, num_nodes=num_nodes,
        embed_dim=embed_dim, seq_length=seq_length
    )
    print(f"\nSTTransformer:")
    print(f"Input shape: {x.shape} -> Reshaped to: {x_tsf.shape}")
    y = model(x_tsf, spatial_info)
    print(f"Output shape: {y.shape}")  # [batch*num_nodes, output_dim]

    # GCN models
    for temporal in ['LSTM', 'CNN', 'Transformer']:
        model = get_model(
            f'GCN{temporal}', input_dim, hidden_dim, output_dim,
            num_layers=num_layers, num_nodes=num_nodes,
            embed_dim=embed_dim, seq_length=seq_length
        )
        print(f"\nGCN with {temporal}:")
        print(f"Input shape: {x.shape}")
        print(f"Edge index shape: {edge_index.shape}")
        y = model(x, edge_index)
        print(f"Output shape: {y.shape}")  # [batch, num_nodes, output_dim]

    # STGCN
    print("\n=== Testing STGCN ===")
    model = get_model(
        'STGCNLSTM', input_dim, hidden_dim, output_dim,
        num_layers=num_layers, num_nodes=num_nodes,
        embed_dim=embed_dim, seq_length=seq_length
    )
    print(f"Input shape: {x.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    y = model(x, edge_index)
    print(f"Output shape: {y.shape}")  # [batch, num_nodes, output_dim]