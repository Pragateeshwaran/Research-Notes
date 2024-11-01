import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, max_relative_position):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_relative_position = max_relative_position
        
        # Linear transformations for Q, K, V
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        # Relative position embeddings
        self.relative_positions_k = nn.Parameter(
            torch.randn(2 * max_relative_position + 1, self.head_dim)
        )
        self.relative_positions_v = nn.Parameter(
            torch.randn(2 * max_relative_position + 1, self.head_dim)
        )

    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.size()
        
        # Linear transformations and reshape
        q = self.q_linear(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.k_linear(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = self.v_linear(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        k = k.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        v = v.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]

        # Calculate relative positions
        relative_positions = self._get_relative_positions(seq_length)
        # print(f"--------------->{relative_positions}")
        # Get relative position embeddings for keys and values
        rel_pos_embed_k = self._get_relative_embeddings(
            self.relative_positions_k, relative_positions
        )  # [seq_len, seq_len, head_dim]
        rel_pos_embed_v = self._get_relative_embeddings(
            self.relative_positions_v, relative_positions
        )  # [seq_len, seq_len, head_dim]
        # print(f"--------------->{rel_pos_embed_k}")

        # Reshape relative position embeddings for broadcasting
        rel_pos_embed_k = rel_pos_embed_k.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len, head_dim]
        rel_pos_embed_v = rel_pos_embed_v.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len, head_dim]

        # Calculate attention scores
        content_scores = torch.matmul(q, k.transpose(-2, -1))  # [batch, num_heads, seq_len, seq_len]
        
        # Calculate relative attention scores
        q_reshaped = q.unsqueeze(-2)  # [batch, num_heads, seq_len, 1, head_dim]
        relative_scores = torch.matmul(q_reshaped, rel_pos_embed_k.transpose(-2, -1))
        relative_scores = relative_scores.squeeze(-2)  # [batch, num_heads, seq_len, seq_len]
        
        # Combine content and relative scores
        attention_scores = (content_scores + relative_scores) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float, device=x.device)
        )

        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch, num_heads, seq_len, seq_len]
        
        # Calculate content and relative outputs
        content_output = torch.matmul(attention_weights, v)  # [batch, num_heads, seq_len, head_dim]
        
        # Calculate relative output
        attention_weights_reshaped = attention_weights.unsqueeze(-2)  # [batch, num_heads, seq_len, 1, seq_len]
        relative_output = torch.matmul(attention_weights_reshaped, rel_pos_embed_v).squeeze(-2)
        
        # Combine outputs
        output = content_output + relative_output
        
        # Reshape and apply output transformation
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        output = self.out_linear(output)
        
        return output

    def _get_relative_positions(self, length):
        """Generate matrix of relative positions between inputs."""
        range_vec = torch.arange(length)
        range_mat = range_vec.unsqueeze(0).repeat(length, 1)
        distance_mat = range_mat - range_mat.transpose(0, 1)
        
        # Clip distances to max_relative_position
        distance_mat_clipped = torch.clamp(
            distance_mat,
            -self.max_relative_position,
            self.max_relative_position
        )
        
        # Shift values to be non-negative
        final_mat = distance_mat_clipped + self.max_relative_position
        return final_mat

    def _get_relative_embeddings(self, relative_embeddings, relative_positions):
        """Look up relative embeddings."""
        flat_relative_positions = relative_positions.view(-1)
        one_hot = F.one_hot(
            flat_relative_positions,
            num_classes=2 * self.max_relative_position + 1
        ).float()
        embeddings = torch.matmul(one_hot, relative_embeddings)
        embeddings = embeddings.view(*relative_positions.size(), -1)
        return embeddings

# Example usage:
if __name__ == "__main__":
    # Parameters
    batch_size = 2
    seq_length = 10
    d_model = 64
    num_heads = 8
    max_relative_position = 4

    # Create model
    model = RelativeMultiHeadAttention(
        d_model=d_model,
        num_heads=num_heads,
        max_relative_position=max_relative_position
    )

    # Create input
    x = torch.randn(batch_size, seq_length, d_model)

    # Forward pass
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")