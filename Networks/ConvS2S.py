import torch
import torch.nn as nn
import sys
sys.path.append("..")
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack
from Networks.ConvS2S_parts import *
from config import arg_parse
from config import PAD_IDX, SOURCE_VOCAB_SIZE, TARGET_VOCAB_SIZE

config = arg_parse()

class FConvEncoder(nn.Module):
    """
    Convolutional encoder consisting of `len(convolutions)` layers.
    Args:
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_dim (int, optional): embedding dimension
        embed_dict (str, optional): filename from which to load pre-trained
            embeddings
        max_positions (int, optional): maximum supported input sequence length
        convolutions (list, optional): the convolutional layer structure. Each
            list item `i` corresponds to convolutional layer `i`. Layers are
            given as ``(out_channels, kernel_width, [residual])``. Residual
            connections are added between layers when ``residual=1`` (which is
            the default behavior).
        dropout (float, optional): dropout to be applied before each conv layer
    """

    def __init__(self, config):
        super().__init__()
        self.dropout_module = Dropout(
            config.fconv_dropout, module_name=self.__class__.__name__
        )
        self.num_attention_layers = None

        num_embeddings = SOURCE_VOCAB_SIZE
        self.padding_idx = PAD_IDX
        self.embed_tokens = Embedding(num_embeddings, config.embedding_size , self.padding_idx)


        self.embed_positions = PositionalEmbedding(
            config.max_positions,
            config.embedding_size,
            self.padding_idx,
        )

        convolutions = extend_conv_spec(config.convolutions)
        in_channels = convolutions[0][0]
        self.fc1 = Linear(config.embedding_size, in_channels, dropout=config.fconv_dropout)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.residuals = []

        layer_in_channels = [in_channels]
        for _, (out_channels, kernel_size, residual) in enumerate(convolutions):
            if residual == 0:
                residual_dim = out_channels
            else:
                residual_dim = layer_in_channels[-residual]
            self.projections.append(
                Linear(residual_dim, out_channels)
                if residual_dim != out_channels
                else None
            )
            if kernel_size % 2 == 1:
                padding = kernel_size // 2
            else:
                padding = 0
            self.convolutions.append(
                convTBC(
                    in_channels,
                    out_channels * 2,
                    kernel_size,
                    dropout=config.fconv_dropout,
                    padding=padding,
                )
            )
            self.residuals.append(residual)
            in_channels = out_channels
            layer_in_channels.append(out_channels)
        self.fc2 = Linear(in_channels, config.embedding_size)

    def forward(self, src_tokens, src_lengths):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): lengths of each source sentence of shape
                `(batch)`
        Returns:
            dict:
                - **encoder_out** (tuple): a tuple with two elements, where the
                  first element is the last encoder layer's output and the
                  second element is the same quantity summed with the input
                  embedding (used for attention). The shape of both tensors is
                  `(batch, src_len, embed_dim)`.
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """
        # embed tokens and positions
        x = self.embed_tokens(src_tokens) + self.embed_positions(src_tokens)
        x = self.dropout_module(x)
        input_embedding = x

        # project to size of convolution
        x = self.fc1(x)

        # used to mask padding in input
        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()  # -> T x B
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        residuals = [x]
        # temporal convolutions
        for proj, conv, res_layer in zip(
            self.projections, self.convolutions, self.residuals
        ):
            if res_layer > 0:
                residual = residuals[-res_layer]
                residual = residual if proj is None else proj(residual)
            else:
                residual = None

            if encoder_padding_mask is not None:
                x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)

            x = self.dropout_module(x)
            if conv.kernel_size[0] % 2 == 1:
                # padding is implicit in the conv
                x = conv(x)
            else:
                padding_l = (conv.kernel_size[0] - 1) // 2
                padding_r = conv.kernel_size[0] // 2
                x = F.pad(x, (0, 0, 0, 0, padding_l, padding_r))
                x = conv(x)
            x = F.glu(x, dim=2)

            if residual is not None:
                x = (x + residual) * math.sqrt(0.5)
            residuals.append(x)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # project back to size of embedding
        x = self.fc2(x)

        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.t()  # -> B x T
            x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)

        # scale gradients (this only affects backward, not forward)
        x = GradMultiply.apply(x, 1.0 / (2.0 * self.num_attention_layers))

        # add output to input embedding for attention
        y = (x + input_embedding) * math.sqrt(0.5)

        return {
            "encoder_out": (x, y),
            "encoder_padding_mask": encoder_padding_mask,  # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = (
                encoder_out["encoder_out"][0].index_select(0, new_order),
                encoder_out["encoder_out"][1].index_select(0, new_order),
            )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.embed_positions.max_positions






class AttentionLayer(nn.Module):
    def __init__(self, conv_channels, embed_dim, bmm=None):
        super().__init__()
        # projects from output of convolution to embedding dimension
        self.in_projection = Linear(conv_channels, embed_dim)
        # projects from embedding dimension to convolution size
        self.out_projection = Linear(embed_dim, conv_channels)

        self.bmm = bmm if bmm is not None else torch.bmm

    def forward(self, x, target_embedding, encoder_out, encoder_padding_mask):
        residual = x

        # attention
        x = (self.in_projection(x) + target_embedding) * math.sqrt(0.5)
        x = self.bmm(x, encoder_out[0])

        # don't attend over padding
        if encoder_padding_mask is not None:
            x = (
                x.float()
                .masked_fill(encoder_padding_mask.unsqueeze(1), float("-inf"))
                .type_as(x)
            )  # FP16 support: cast to float and back

        # softmax over last dim
        sz = x.size()
        x = F.softmax(x.view(sz[0] * sz[1], sz[2]), dim=1)
        x = x.view(sz)
        attn_scores = x

        x = self.bmm(x, encoder_out[1])

        # scale attention output (respecting potentially different lengths)
        s = encoder_out[1].size(1)
        if encoder_padding_mask is None:
            x = x * (s * math.sqrt(1.0 / s))
        else:
            s = s - encoder_padding_mask.type_as(x).sum(
                dim=1, keepdim=True
            )  # exclude padding
            s = s.unsqueeze(-1)
            x = x * (s * s.rsqrt())

        # project back
        x = (self.out_projection(x) + residual) * math.sqrt(0.5)
        return x, attn_scores

    def make_generation_fast_(self, beamable_mm_beam_size=None, **kwargs):
        """Replace torch.bmm with BeamableMM."""
        if beamable_mm_beam_size is not None:
            del self.bmm
            self.add_module("bmm", BeamableMM(beamable_mm_beam_size))




@with_incremental_state
class FConvDecoder(nn.Module):
    """Convolutional decoder"""

    def __init__(
        self, config,
        attention=True,
        share_embed=False,
        positional_embeddings=True,
        adaptive_softmax_cutoff=None,
        adaptive_softmax_dropout=0.0,
    ):
        super().__init__()
        self.register_buffer("version", torch.Tensor([2]))
        self.dropout_module = Dropout(
            config.fconv_dropout, module_name=self.__class__.__name__
        )
        self.need_attn = True

        convolutions = extend_conv_spec(config.convolutions)
        in_channels = convolutions[0][0]
        if isinstance(attention, bool):
            # expand True into [True, True, ...] and do the same with False
            attention = [attention] * len(convolutions)
        if not isinstance(attention, list) or len(attention) != len(convolutions):
            raise ValueError(
                "Attention is expected to be a list of booleans of "
                "length equal to the number of layers."
            )

        num_embeddings = TARGET_VOCAB_SIZE
        padding_idx = PAD_IDX
        self.embed_tokens = Embedding(num_embeddings, config.embedding_size, padding_idx)

        self.embed_positions = (
            PositionalEmbedding(
                config.max_positions,
                config.embedding_size,
                padding_idx,
            )
            if positional_embeddings
            else None
        )

        self.fc1 = Linear(config.embedding_size, in_channels, dropout=config.fconv_dropout)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.residuals = []

        layer_in_channels = [in_channels]
        for i, (out_channels, kernel_size, residual) in enumerate(convolutions):
            if residual == 0:
                residual_dim = out_channels
            else:
                residual_dim = layer_in_channels[-residual]
            self.projections.append(
                Linear(residual_dim, out_channels)
                if residual_dim != out_channels
                else None
            )
            self.convolutions.append(
                LinearizedConv1d(
                    in_channels,
                    out_channels * 2,
                    kernel_size,
                    padding=(kernel_size - 1),
                    dropout=config.fconv_dropout,
                )
            )
            self.attention.append(
                AttentionLayer(out_channels, config.embedding_size) if attention[i] else None
            )
            self.residuals.append(residual)
            in_channels = out_channels
            layer_in_channels.append(out_channels)

        self.adaptive_softmax = None
        self.fc2 = self.fc3 = None

        if adaptive_softmax_cutoff is not None:
            assert not share_embed
            self.adaptive_softmax = AdaptiveSoftmax(
                num_embeddings,
                in_channels,
                adaptive_softmax_cutoff,
                dropout=adaptive_softmax_dropout,
            )
        else:
            self.fc2 = Linear(in_channels, config.out_embedding_size)
            if share_embed:
                assert config.out_embedding_size == config.embedding_size, (
                    "Shared embed weights implies same dimensions "
                    " out_embed_dim={} vs embed_dim={}".format(config.out_embedding_size, config.embedding_size)
                )
                self.fc3 = nn.Linear(config.out_embedding_size, num_embeddings)
                self.fc3.weight = self.embed_tokens.weight
            else:
                self.fc3 = Linear(config.out_embedding_size, num_embeddings, dropout=config.fconv_dropout)

    def forward(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused
    ):
        if encoder_out is not None:
            encoder_padding_mask = encoder_out["encoder_padding_mask"]
            encoder_out = encoder_out["encoder_out"]

            # split and transpose encoder outputs
            encoder_a, encoder_b = self._split_encoder_out(
                encoder_out, incremental_state
            )

        if self.embed_positions is not None:
            pos_embed = self.embed_positions(prev_output_tokens, incremental_state)
        else:
            pos_embed = 0

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        x = self._embed_tokens(prev_output_tokens, incremental_state)

        # embed tokens and combine with positional embeddings
        x += pos_embed
        x = self.dropout_module(x)
        target_embedding = x

        # project to size of convolution
        x = self.fc1(x)

        # B x T x C -> T x B x C
        x = self._transpose_if_training(x, incremental_state)

        # temporal convolutions
        avg_attn_scores = None
        num_attn_layers = len(self.attention)
        residuals = [x]
        for proj, conv, attention, res_layer in zip(
            self.projections, self.convolutions, self.attention, self.residuals
        ):
            if res_layer > 0:
                residual = residuals[-res_layer]
                residual = residual if proj is None else proj(residual)
            else:
                residual = None

            x = self.dropout_module(x)
            x = conv(x, incremental_state)
            x = F.glu(x, dim=2)

            # attention
            if attention is not None:
                x = self._transpose_if_training(x, incremental_state)

                x, attn_scores = attention(
                    x, target_embedding, (encoder_a, encoder_b), encoder_padding_mask
                )

                if not self.training and self.need_attn:
                    attn_scores = attn_scores / num_attn_layers
                    if avg_attn_scores is None:
                        avg_attn_scores = attn_scores
                    else:
                        avg_attn_scores.add_(attn_scores)

                x = self._transpose_if_training(x, incremental_state)

            # residual
            if residual is not None:
                x = (x + residual) * math.sqrt(0.5)
            residuals.append(x)

        # T x B x C -> B x T x C
        x = self._transpose_if_training(x, incremental_state)

        # project back to size of vocabulary if not using adaptive softmax
        if self.fc2 is not None and self.fc3 is not None:
            x = self.fc2(x)
            x = self.dropout_module(x)
            x = self.fc3(x)

        return x, avg_attn_scores

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        encoder_out = get_incremental_state(
            self, incremental_state, "encoder_out"
        )
        if encoder_out is not None:
            encoder_out = tuple(eo.index_select(0, new_order) for eo in encoder_out)
            set_incremental_state(
                self, incremental_state, "encoder_out", encoder_out
            )

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return (
            self.embed_positions.max_positions
            if self.embed_positions is not None
            else float("inf")
        )

    def upgrade_state_dict(self, state_dict):
        if item(state_dict.get("decoder.version", torch.Tensor([1]))[0]) < 2:
            # old models use incorrect weight norm dimension
            for i, conv in enumerate(self.convolutions):
                # reconfigure weight norm
                nn.utils.remove_weight_norm(conv)
                self.convolutions[i] = nn.utils.weight_norm(conv, dim=0)
            state_dict["decoder.version"] = torch.Tensor([1])
        return state_dict

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

    def _embed_tokens(self, tokens, incremental_state):
        if incremental_state is not None:
            # keep only the last token for incremental forward pass
            tokens = tokens[:, -1:]
        return self.embed_tokens(tokens)

    def _split_encoder_out(self, encoder_out, incremental_state):
        """Split and transpose encoder outputs.
        This is cached when doing incremental inference.
        """
        cached_result = get_incremental_state(
            self, incremental_state, "encoder_out"
        )
        if cached_result is not None:
            return cached_result

        # transpose only once to speed up attention layers
        encoder_a, encoder_b = encoder_out
        encoder_a = encoder_a.transpose(1, 2).contiguous()
        result = (encoder_a, encoder_b)

        if incremental_state is not None:
            set_incremental_state(self, incremental_state, "encoder_out", result)
        return result

    def _transpose_if_training(self, x, incremental_state):
        if incremental_state is None:
            x = x.transpose(0, 1)
        return x
    
    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""
        vocab = net_output.size(-1)
        net_output1 = net_output.view(-1, vocab)
        if log_probs:
            return F.log_softmax(net_output1, dim=1).view_as(net_output)
        else:
            return F.softmax(net_output1, dim=1).view_as(net_output)



class ConvS2S(nn.Module):
    """
    ConvS2S:
    Input:
        encoder:
        decoder:
        attention:
        generator:
    return:
    """
    def __init__(self, config):
        super(ConvS2S, self).__init__()
        self.config = config
        self.encoder = FConvEncoder(self.config)
        self.decoder = FConvDecoder(self.config)
        self.encoder.num_attention_layers = sum(
            layer is not None for layer in self.decoder.attention
        )

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Run the forward pass for an encoder-decoder model.
        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs::
            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out, _ = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )

        
        # 标准化概率 softmax 使用交叉熵不需要进行标准化
        # decoder_out = self.decoder.get_normalized_probs(decoder_out, log_probs=False)
        
        lprobs = decoder_out.view(-1, decoder_out.size(-1))

        return decoder_out , lprobs