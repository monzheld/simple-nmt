import torch
import torch.nn as nn

import simple_nmt.data_loader as data_loader
from simple_nmt.search import SingleBeamSearchBoard


class Attention(nn.Module):
    """ 
    Attention
    """

    def __init__(self):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None, dk=64):
        # |Q| = (batch_size, m, hidden_size)
            # -> 실제로 들어오는 사이즈: (n_splits * batch_size, m, hidden_size / n_splits)
        # |K| = |V| = (batch_size, n, hidden_size)
            # -> 실제로 들어오는 사이즈: (n_splits * batch_size, n, hidden_size / n_splits)
        # |mask| = (batch_size, m, n)
            # -> 실제로 들어오는 사이즈: (n_splits * batch_size, m, n)

        # weight 구하기
            # K에 transpose 
                # 원래 |K| = (batch_size, n, hidden_size)
                # K에 transpose(KT) = (batch_size, hidden_size, n)
        w = torch.bmm(Q, K.transpose(1, 2))
        # |Q| = (batch_size, m, hidden_size) * KT = (batch_size, hidden_size, n)
        # => |w| = (batch_size, m, n)
            # -> normalize되지 않은 weight 값

        # mask 구하기
        # -무한대를 넣어준 후, softmax 취하기 => 0이 들어감
        if mask is not None:
            assert w.size() == mask.size()
            w.masked_fill_(mask, -float('inf'))

        # gradient를 안정적으로 만들기 위해 (dk**.5)로 나눠줌
        w = self.softmax(w / (dk**.5))
            # -> 확률분포처럼 다 더했을 때 1이 되는 normalize된 weight 값이 나옴
            # size는 그대로 |w| = (batch_size, m, n)
        
        # normalize된 w에 V를 Batch matrix multiplication
        c = torch.bmm(w, V)
        # |w| = (batch_size, m, n) * |V| = (batch_size, n, hidden_size)
        # => |c| = (batch_size, m, hidden_size)
            # -> 실제로 return 될 사이즈: (n_splits * batch_size, m, hidden_size / n_splits)

        return c


class MultiHead(nn.Module):
    """ 
    Multi-Head Attention
    """

    def __init__(self, hidden_size, n_splits):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_splits = n_splits

        # Note that we don't have to declare each linear layer, separately.
        # Q, K, V에 대해 각각 linear transformation한 것
            # head 하나의 dimension = hidden_size / n_splits
                # (hidden_size / n_splits) * n_splits => hidden_size * hidden_size
        self.Q_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.K_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        # 마지막으로 통과할 linear layer 하나
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

        # Attention 선언
        self.attn = Attention()

    def forward(self, Q, K, V, mask=None):
        # |Q|    = (batch_size, m, hidden_size) -> m = 디코더의 time-step 개수
        # |K|    = (batch_size, n, hidden_size) -> n = 인코더의 time-step 개수
        # |V|    = |K|
            # -> 인코더의 최종 output: |K|, |V|
            # 디코더의 현재 layer의 이전 layer의 output: |Q|
        # |mask| = (batch_size, m, n)

        # 맨 마지막 dimension(dim=-1)에 대해 split()
        # -> QWs, KWs, VWs는 리스트가 됨
        QWs = self.Q_linear(Q).split(self.hidden_size // self.n_splits, dim=-1)
        KWs = self.K_linear(K).split(self.hidden_size // self.n_splits, dim=-1)
        VWs = self.V_linear(V).split(self.hidden_size // self.n_splits, dim=-1)
        # |QW_i| = (batch_size, m, hidden_size / n_splits)
        # |KW_i| = |VW_i| = (batch_size, n, hidden_size / n_splits)
        # -> 그 리스트 안에 들어있는 각 원소

        # parallel 하게 연산하기 위해 QWs, KWs, VWs 재정의
        # By concatenating splited linear transformed results,
        # we can remove sequential operations,
        # like mini-batch parallel operations.
        QWs = torch.cat(QWs, dim=0)
        KWs = torch.cat(KWs, dim=0)
        VWs = torch.cat(VWs, dim=0)
        # |QWs| = (batch_size * n_splits, m, hidden_size / n_splits)
        # |KWs| = |VWs| = (batch_size * n_splits, n, hidden_size / n_splits)

        # mask가 들어오는 경우, mask도 확장해 줌
            # 원래 mask 사이즈 => |mask| = (batch_size, m, n)
        if mask is not None:
            mask = torch.cat([mask for _ in range(self.n_splits)], dim=0)
            # |mask| = (batch_size * n_splits, m, n)

        c = self.attn(
            QWs, KWs, VWs,
            mask=mask,
            dk=self.hidden_size // self.n_splits,
        )
        # |c| = (batch_size * n_splits, m, hidden_size / n_splits)

        # We need to restore temporal mini-batchfied multi-head attention results.
        # batch_size로 쪼개기
        c = c.split(Q.size(0), dim=0)
        # |c_i| = (batch_size, m, hidden_size / n_splits)

        # 먼저, cat을 마지막 dimension(dim=-1)에 붙이고 linear layer 통과
        c = self.linear(torch.cat(c, dim=-1))
        # |c| = (batch_size, m, hidden_size)

        return c


class EncoderBlock(nn.Module):
    """ 
    Encoder Block

    - 2개의 Layer Normalization
        1) Attention을 위한 Layer Normalization
        2) Feed Forward 다음에 있는 Layer Normalization
    """

    def __init__(
        self,
        hidden_size,
        n_splits,
        dropout_p=.1,
        use_leaky_relu=False,
    ):
        super().__init__()

        # Multi-Head Attention
        self.attn = MultiHead(hidden_size, n_splits)
        # 1) Attention을 위한 Layer Normalization
        self.attn_norm = nn.LayerNorm(hidden_size)
        # Dropout
        self.attn_dropout = nn.Dropout(dropout_p)

        # Feed Forward(FFN)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4), # hidden_size로 들어온 것을 hidden_size의 4배로 늘림
            nn.LeakyReLU() if use_leaky_relu else nn.ReLU(), # LeakyReLU 또는 ReLU에 통과
            nn.Linear(hidden_size * 4, hidden_size), # hidden_size의 4배에서 다시 hidden_size로 돌아감
        )
        # 2) Feed Forward 다음에 있는 Layer Normalization
        self.fc_norm = nn.LayerNorm(hidden_size)
        # Dropout
        self.fc_dropout = nn.Dropout(dropout_p)

    def forward(self, x, mask):
        # |x|    = (batch_size, n, hidden_size)
        # |mask| = (batch_size, n, n)

        # (논문에 나와있는 방식)
        # Post-LN(Post Layer Normalization):

        # 순서: Muti-Head Attention(self.attn) -> dropout(self.attn_dropout) -> residual connection(이전 입력인 x 더하기) -> Layer Normalization(self.attn_norm) 
            # Q, K, V에는 각각 x가 들어감(self-attention이기 때문)
        # z = self.attn_norm(x + self.attn_dropout(self.attn(Q=x,
        #                                                    K=x,
        #                                                    V=x,
        #                                                    mask=mask)))

        # 순서: FFN(self.fc) -> dropout(self.fc_dropout) -> residual connection(이전 입력인 z 더하기) -> Layer Normalization(self.fc_norm)
        # z = self.fc_norm(z + self.fc_dropout(self.fc(z)))


        # (모델에 적용한 방식)
        # Pre-LN(Pre Layer Normalization):

        # 먼저, Layer Normalization(self.attn_norm)
        z = self.attn_norm(x)

        # 순서: Muti-Head Attention(self.attn) -> dropout(self.attn_dropout) -> residual connection(이전 입력인 x 더하기)
            # Q, K, V에는 각각 z가 들어감(self-attention이기 때문)
        z = x + self.attn_dropout(self.attn(Q=z,
                                            K=z,
                                            V=z,
                                            mask=mask))

        # 순서: Layer Normalization(self.fc_norm) -> FFN(self.fc) -> dropout(self.fc_dropout) -> residual connection(이전 입력인 z 더하기)
        z = z + self.fc_dropout(self.fc(self.fc_norm(z)))
        # |z| = (batch_size, n, hidden_size)

        return z, mask
        # -> 입력과 출력 인터페이스가 같음


class DecoderBlock(nn.Module):
    """ 
    Decoder Block

    - 2개의 Attention
        1) Decoder 자기 자신에 대한 self-attention
        2) Encoder에 대한 attention
    """

    def __init__(
        self,
        hidden_size,
        n_splits,
        dropout_p=.1,
        use_leaky_relu=False,
    ):
        super().__init__()

        # 1) Decoder 자기 자신에 대한 self-attention
        self.masked_attn = MultiHead(hidden_size, n_splits)
        # masked_attn에 대한 Layer Normalization
        self.masked_attn_norm = nn.LayerNorm(hidden_size) 
        # masked_attn에 대한 Dropout
        self.masked_attn_dropout = nn.Dropout(dropout_p) 

        # 2) Encoder에 대한 attention
        self.attn = MultiHead(hidden_size, n_splits)
        # attn에 대한 Layer Normalization
        self.attn_norm = nn.LayerNorm(hidden_size)
        # attn에 대한 Dropout
        self.attn_dropout = nn.Dropout(dropout_p)

        # Feed Forward(FFN)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4), # hidden_size로 들어온 것을 hidden_size의 4배로 늘림
            nn.LeakyReLU() if use_leaky_relu else nn.ReLU(), # LeakyReLU 또는 ReLU에 통과
            nn.Linear(hidden_size * 4, hidden_size), # hidden_size의 4배에서 다시 hidden_size로 돌아감
        )
        # Feed Forward 다음에 있는 Layer Normalization
        self.fc_norm = nn.LayerNorm(hidden_size)
        # Dropout
        self.fc_dropout = nn.Dropout(dropout_p)

    def forward(self, x, key_and_value, mask, prev, future_mask):
        # |key_and_value| = (batch_size, n, hidden_size) # -> Encoder의 output
        # |mask|          = (batch_size, m, n) # -> Encoder에서 비어있는 time-step(pad가 들어있는 곳)을 마스킹 해놓은 mask

        # In case of inference, we don't have to repeat same feed-forward operations.
        # Thus, we save previous feed-forward results.

        # prev: 각 layer 별로 이전 time-step까지의 모든 출력값
        # -> prev가 주어지면 추론, 아니면 학습 모드 

        # 학습 시에는 모든 time-step가 한 번에 들어감
        if prev is None: # Training mode
            # |x|           = (batch_size, m, hidden_size) # -> 전체 time-step에 다 들어옴
            # |prev|        = None
            # |future_mask| = (batch_size, m, m) # -> self-attention 시 미래의 time-step을 보지 못하게 하는 mask
            # |z|           = (batch_size, m, hidden_size) # -> 아래에서 실행한 결과값

            # Post-LN:
            # z = self.masked_attn_norm(x + self.masked_attn_dropout(
            #     self.masked_attn(x, x, x, mask=future_mask)
            # ))

            # Pre-LN:
            # 먼저, Layer Normalization 
            z = self.masked_attn_norm(x)
            z = x + self.masked_attn_dropout(
                self.masked_attn(z, z, z, mask=future_mask)
            )
            # => Encoder와 같은 방식 
        
        # 추론 시에는 하나의 time-step씩 들어감
        else: # Inference mode
            # |x|           = (batch_size, 1, hidden_size) # -> 하나의 time-step만 들어옴
            # |prev|        = (batch_size, t - 1, hidden_size) # -> 각 layer 별로 이전 time-step까지의 모든 출력값
            # |future_mask| = None # -> 미래의 time-step이 없기 때문에 mask 필요 없음 
            # |z|           = (batch_size, 1, hidden_size) # -> 아래에서 실행한 결과값

            # prev에 대해 Attention 수행
            # Post-LN:
            # z = self.masked_attn_norm(x + self.masked_attn_dropout(
            #     self.masked_attn(x, prev, prev, mask=None)
            # ))
                               # -> Q = x, K = prev, V = prev

            # Pre-LN:
            # 먼저, Layer Normalization 
            normed_prev = self.masked_attn_norm(prev)
            z = self.masked_attn_norm(x)
            z = x + self.masked_attn_dropout(
                self.masked_attn(z, normed_prev, normed_prev, mask=None) # mask 생략 (추론 모드에서는 미래의 time-step이 없기 때문)
            )
                            # -> Q = z, K = normed_prev, V = normed_prev

        # Post-LN:
        # z = self.attn_norm(z + self.attn_dropout(self.attn(Q=z, # -> 위에서 나온 z
        #                                                    K=key_and_value, # -> Encoder의 output
        #                                                    V=key_and_value, # -> Encoder의 output
        #                                                    mask=mask))) # -> Encoder에서 비어있는 time-step(pad가 들어있는 곳)을 마스킹 해놓은 mask

        # Pre-LN:
        # 먼저, Layer Normalization 
        normed_key_and_value = self.attn_norm(key_and_value)
        z = z + self.attn_dropout(self.attn(Q=self.attn_norm(z), # -> 위에서 나온 z
                                            K=normed_key_and_value, 
                                            V=normed_key_and_value, 
                                            mask=mask)) # -> Encoder에서 비어있는 time-step(pad가 들어있는 곳)을 마스킹 해놓은 mask
        # |z| = (batch_size, m, hidden_size)


        # Feed Forward(FFN) layer 통과

        # Post-LN:
        # z = self.fc_norm(z + self.fc_dropout(self.fc(z)))

        # Pre-LN:
        z = z + self.fc_dropout(self.fc(self.fc_norm(z)))
        # |z| = (batch_size, m, hidden_size)

        return z, key_and_value, mask, prev, future_mask
        # 이 출력값들을 Sequential에 넣을 것이기 때문에 다음 layer에서 이 값들을 그대로 입력받음
        # -> 입력과 출력 인터페이스가 같음


class MySequential(nn.Sequential):
    """ 
    Sequential

    - nn.Sequential의 경우, 하나의 tensor만 입력으로 받는다는 문제 존재
    - for문을 통해 tuple 같이 여러 개의 입력이 주어졌을 때도 입력과 출력 interface가 같도록 만들어줌
    """

    def forward(self, *x):
        # nn.Sequential class does not provide multiple input arguments and returns.
        # Thus, we need to define new class to solve this issue.
        # Note that each block has same function interface.

        for module in self._modules.values():
            x = module(*x)

        return x


class Transformer(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        n_splits,
        n_enc_blocks=6,
        n_dec_blocks=6,
        dropout_p=.1,
        use_leaky_relu=False,
        max_length=512,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_splits = n_splits
        self.n_enc_blocks = n_enc_blocks
        self.n_dec_blocks = n_dec_blocks
        self.dropout_p = dropout_p
        self.max_length = max_length

        super().__init__()

        # Encoder Embedding
        self.emb_enc = nn.Embedding(input_size, hidden_size) # -> source language의 vocab size를 입력으로 받아서 hidden_size로 뱉어줌
        # Decoder Embedding
        self.emb_dec = nn.Embedding(output_size, hidden_size) # -> target language의 vocab size를 입력으로 받아서 hidden_size로 뱉어줌 
        # Dropout
        self.emb_dropout = nn.Dropout(dropout_p)

        # Positional Encoding
        # -> positional encoding을 하는 matrix를 미리 max_length 만큼 하나 만들어놓고 필요할 때마다 필요한 만큼만 잘라서 사용
        self.pos_enc = self._generate_pos_enc(hidden_size, max_length)

        # Encoder, Decoder -> MySequential 사용
        self.encoder = MySequential(
            *[EncoderBlock(
                hidden_size,
                n_splits,
                dropout_p,
                use_leaky_relu,
              ) for _ in range(n_enc_blocks)], # for문으로 필요한 encoder blocks의 개수만큼을 선언
        )
        self.decoder = MySequential(
            *[DecoderBlock(
                hidden_size,
                n_splits,
                dropout_p,
                use_leaky_relu,
              ) for _ in range(n_dec_blocks)], # for문으로 필요한 decoder blocks의 개수만큼을 선언
        )
        # generator -> nn.Sequential 사용 
        self.generator = nn.Sequential(
            nn.LayerNorm(hidden_size), # Only for Pre-LN Transformer. (Post-LN인 경우, 이 부분 필요 x)
            nn.Linear(hidden_size, output_size),
            nn.LogSoftmax(dim=-1), # 각 time-step 별, 단어 별 로그 확률 값을 return 
        )

    @torch.no_grad()
    # positional encoding을 하는 전체 matrix를 생성하는 함수
    # -> max_length 만큼의 matrix를 만들고, 그 안에 값들을 모두 채움
    def _generate_pos_enc(self, hidden_size, max_length):
        enc = torch.FloatTensor(max_length, hidden_size).zero_()
        # |enc| = (max_length, hidden_size)

        pos = torch.arange(0, max_length).unsqueeze(-1).float()
        dim = torch.arange(0, hidden_size // 2).unsqueeze(0).float()
        # |pos| = (max_length, 1)
        # |dim| = (1, hidden_size // 2)

        # 짝수 dimension -> sin()
        enc[:, 0::2] = torch.sin(pos / 1e+4**dim.div(float(hidden_size)))
        # 홀수 dimension -> cos()
        enc[:, 1::2] = torch.cos(pos / 1e+4**dim.div(float(hidden_size)))

        return enc

    # matrix에서 필요한 만큼 자르는 함수 
    def _position_encoding(self, x, init_pos=0):
        # init_pos(= initial position): 추론 시 Decoder의 positional encoding은 time-step이 하나씩 들어가기 때문에 항상 같은 position이 들어가는 것을 막기 위해 필요
            # (학습 시에는 항상 전체 time-step이 다 들어가기 때문에 상관 없음)
        # |x| = (batch_size, n, hidden_size)
        # |self.pos_enc| = (max_length, hidden_size)
        assert x.size(-1) == self.pos_enc.size(-1)
        assert x.size(1) + init_pos <= self.max_length

        pos_enc = self.pos_enc[init_pos:init_pos + x.size(1)].unsqueeze(0)
            # [init_pos:init_pos + x.size(1)] -> 0부터 n까지 
            # => self.pos_enc을 0부터 (0 + n)까지 자름
        # |pos_enc| = (1, n, hidden_size)
        x = x + pos_enc.to(x.device)
            # -> |pos_enc| = (1, n, hidden_size)에서의 1이 broadcasting돼서 batch_size로 자동으로 늘어남
            # => x의 원래 size인 (batch_size, n, hidden_size)에 맞게 잘라짐

        return x

    @torch.no_grad()
    # mask를 생성하는 함수
    def _generate_mask(self, x, length):
        mask = []

        max_length = max(length)
        for l in length:
            if max_length - l > 0:
                # If the length is shorter than maximum length among samples,
                # set last few values to be 1s to remove attention weight.
                mask += [torch.cat([x.new_ones(1, l).zero_(), # 0이 l 개만큼
                                    x.new_ones(1, (max_length - l)) # 1이 (max_length - l) 개만큼
                                    ], dim=-1)]
            else:
                # If length of sample equals to maximum length among samples,
                # set every value in mask to be 0.
                mask += [x.new_ones(1, l).zero_()] # 0이 l 개만큼

        mask = torch.cat(mask, dim=0).bool()
        # |mask| = (batch_size, max_length)

        return mask

    def forward(self, x, y):
        # |x[0]| = (batch_size, n)
        # |y|    = (batch_size, m)

        # Mask Generation
        # Mask to prevent having attention weight on padding position.
        with torch.no_grad(): # 학습 시 gradient를 받을 필요가 없으므로 no_grad()를 준 상태에서 mask generation 
            mask = self._generate_mask(x[0], x[1]) # x[0] : 원핫 인코딩 벡터(tensor), x[1] : mini-batch의 각 sample 별 length(=time-step)
            # |mask| = (batch_size, n)
            x = x[0]

            mask_enc = mask.unsqueeze(1).expand(*x.size(), mask.size(-1))
                # mask.unsqueeze(1) -> (batch_size, 1, n)
                # *x.size(): (batch_size, n)
                # mask.size(-1): n (|mask| = (batch_size, n)의 제일 마지막 원소인 n)
            mask_dec = mask.unsqueeze(1).expand(*y.size(), mask.size(-1))
            # |mask_enc| = (batch_size, n, n) # -> Encoder에서의 mask
                # => Encoder에서 self-attention할 때, 빈 time-step에 masking된 것
            # |mask_dec| = (batch_size, m, n) # -> Decoder에서의 mask
                # => Decoder에서 Encoder에 attention할 때, Encoder의 빈 time-step에 masking된 것

        # 순서: x를 Encoder Embedding layer에 통과 -> positional encoding(여기에서는 init_pos 없음) -> Dropout
        z = self.emb_dropout(self._position_encoding(self.emb_enc(x))) # -> word embedding vector(tensor)가 나옴
        # z를 encoder에 통과
        z, _ = self.encoder(z, mask_enc)
            # -> z만 필요
        # |z| = (batch_size, n, hidden_size)
            # => z는 Encoder의 최종 layer의 output
            # (Decoder에서 이 z에 대해 attention 수행)

        # Generate future mask
        # future mask: 미래의 time-step을 보지 못하게 하는 mask
        # -> 따로 함수로 구현하지 않고 매번 구함
        with torch.no_grad():
            future_mask = torch.triu(x.new_ones((y.size(1), y.size(1))), diagonal=1).bool()
                # triu: upper triangle
                # y.size(1): m
                # => 1로 가득 찬 m x m 사이즈의 triu matrix
                # 주의) diagonal에 0이 아닌 1을 줌
            # |future_mask| = (m, m)

            # future_mask를 batch_size 만큼 똑같이 복사
            future_mask = future_mask.unsqueeze(0).expand(y.size(0), *future_mask.size())
                # future_mask.unsqueeze(0) -> (1, m, m)
                # y.size(0): batch_size
                # *future_mask.size(): (m, m)
            # |future_mask| = (batch_size, m, m)

        # 순서: y를 Decoder Embedding layer에 통과 -> positional encoding -> Dropout
        h = self.emb_dropout(self._position_encoding(self.emb_dec(y))) # -> word embedding vector(tensor)가 나옴
        # h를 decoder에 통과
        h, _, _, _, _ = self.decoder(h, z, mask_dec, None, future_mask) 
            # -> prev = None
            # z: Encoder의 최종 layer의 output
            # -> h만 필요
        # |h| = (batch_size, m, hidden_size)

        # h를 generator에 넣어줌
        y_hat = self.generator(h) # -> 로그 확률 값으로 나옴
        # |y_hat| = (batch_size, m, output_size)

        return y_hat

    def search(self, x, is_greedy=True, max_length=255):
        # |x[0]| = (batch_size, n)
        batch_size = x[0].size(0)

        mask = self._generate_mask(x[0], x[1])
        # |mask| = (batch_size, n)
        x = x[0]

        mask_enc = mask.unsqueeze(1).expand(mask.size(0), x.size(1), mask.size(-1))
        mask_dec = mask.unsqueeze(1)
        # |mask_enc| = (batch_size, n, n)
        # |mask_dec| = (batch_size, 1, n)

        z = self.emb_dropout(self._position_encoding(self.emb_enc(x)))
        z, _ = self.encoder(z, mask_enc)
        # |z| = (batch_size, n, hidden_size)

        # Fill a vector, which has 'batch_size' dimension, with BOS value.
        y_t_1 = x.new(batch_size, 1).zero_() + data_loader.BOS
        # |y_t_1| = (batch_size, 1)
        is_decoding = x.new_ones(batch_size, 1).bool()

        prevs = [None for _ in range(len(self.decoder._modules) + 1)]
        y_hats, indice = [], []
        # Repeat a loop while sum of 'is_decoding' flag is bigger than 0,
        # or current time-step is smaller than maximum length.
        while is_decoding.sum() > 0 and len(indice) < max_length:
            # Unlike training procedure,
            # take the last time-step's output during the inference.
            h_t = self.emb_dropout(
                self._position_encoding(self.emb_dec(y_t_1), init_pos=len(indice))
            )
            # |h_t| = (batch_size, 1, hidden_size))
            if prevs[0] is None:
                prevs[0] = h_t
            else:
                prevs[0] = torch.cat([prevs[0], h_t], dim=1)

            for layer_index, block in enumerate(self.decoder._modules.values()):
                prev = prevs[layer_index]
                # |prev| = (batch_size, len(y_hats), hidden_size)

                h_t, _, _, _, _ = block(h_t, z, mask_dec, prev, None)
                # |h_t| = (batch_size, 1, hidden_size)

                if prevs[layer_index + 1] is None:
                    prevs[layer_index + 1] = h_t
                else:
                    prevs[layer_index + 1] = torch.cat([prevs[layer_index + 1], h_t], dim=1)
                # |prev| = (batch_size, len(y_hats) + 1, hidden_size)

            y_hat_t = self.generator(h_t)
            # |y_hat_t| = (batch_size, 1, output_size)

            y_hats += [y_hat_t]
            if is_greedy: # Argmax
                y_t_1 = torch.topk(y_hat_t, 1, dim=-1)[1].squeeze(-1)
            else: # Random sampling                
                y_t_1 = torch.multinomial(y_hat_t.exp().view(x.size(0), -1), 1)
            # Put PAD if the sample is done.
            y_t_1 = y_t_1.masked_fill_(
                ~is_decoding,
                data_loader.PAD,
            )

            # Update is_decoding flag.
            is_decoding = is_decoding * torch.ne(y_t_1, data_loader.EOS)
            # |y_t_1| = (batch_size, 1)
            # |is_decoding| = (batch_size, 1)
            indice += [y_t_1]

        y_hats = torch.cat(y_hats, dim=1)
        indice = torch.cat(indice, dim=-1)
        # |y_hats| = (batch_size, m, output_size)
        # |indice| = (batch_size, m)

        return y_hats, indice

    #@profile
    def batch_beam_search(
        self,
        x,
        beam_size=5,
        max_length=255,
        n_best=1,
        length_penalty=.2,
    ):
        # |x[0]| = (batch_size, n)
        batch_size = x[0].size(0)
        n_dec_layers = len(self.decoder._modules)

        mask = self._generate_mask(x[0], x[1])
        # |mask| = (batch_size, n)
        x = x[0]

        mask_enc = mask.unsqueeze(1).expand(mask.size(0), x.size(1), mask.size(-1))
        mask_dec = mask.unsqueeze(1)
        # |mask_enc| = (batch_size, n, n)
        # |mask_dec| = (batch_size, 1, n)

        z = self.emb_dropout(self._position_encoding(self.emb_enc(x)))
        z, _ = self.encoder(z, mask_enc)
        # |z| = (batch_size, n, hidden_size)

        prev_status_config = {}
        for layer_index in range(n_dec_layers + 1):
            prev_status_config['prev_state_%d' % layer_index] = {
                'init_status': None,
                'batch_dim_index': 0,
            }
        # Example of prev_status_config:
        # prev_status_config = {
        #     'prev_state_0': {
        #         'init_status': None,
        #         'batch_dim_index': 0,
        #     },
        #     'prev_state_1': {
        #         'init_status': None,
        #         'batch_dim_index': 0,
        #     },
        #
        #     ...
        #
        #     'prev_state_${n_layers}': {
        #         'init_status': None,
        #         'batch_dim_index': 0,
        #     }
        # }

        boards = [
            SingleBeamSearchBoard(
                z.device,
                prev_status_config,
                beam_size=beam_size,
                max_length=max_length,
            ) for _ in range(batch_size)
        ]
        done_cnt = [board.is_done() for board in boards]

        length = 0
        while sum(done_cnt) < batch_size and length <= max_length:
            fab_input, fab_z, fab_mask = [], [], []
            fab_prevs = [[] for _ in range(n_dec_layers + 1)]

            for i, board in enumerate(boards): # i == sample_index in minibatch
                if board.is_done() == 0:
                    y_hat_i, prev_status = board.get_batch()

                    fab_input += [y_hat_i                 ]
                    fab_z     += [z[i].unsqueeze(0)       ] * beam_size
                    fab_mask  += [mask_dec[i].unsqueeze(0)] * beam_size

                    for layer_index in range(n_dec_layers + 1):
                        prev_i = prev_status['prev_state_%d' % layer_index]
                        if prev_i is not None:
                            fab_prevs[layer_index] += [prev_i]
                        else:
                            fab_prevs[layer_index] = None

            fab_input = torch.cat(fab_input, dim=0)
            fab_z     = torch.cat(fab_z,     dim=0)
            fab_mask  = torch.cat(fab_mask,  dim=0)
            for i, fab_prev in enumerate(fab_prevs): # i == layer_index
                if fab_prev is not None:
                    fab_prevs[i] = torch.cat(fab_prev, dim=0)
            # |fab_input|    = (current_batch_size, 1,)
            # |fab_z|        = (current_batch_size, n, hidden_size)
            # |fab_mask|     = (current_batch_size, 1, n)
            # |fab_prevs[i]| = (current_batch_size, length, hidden_size)
            # len(fab_prevs) = n_dec_layers + 1

            # Unlike training procedure,
            # take the last time-step's output during the inference.
            h_t = self.emb_dropout(
                self._position_encoding(self.emb_dec(fab_input), init_pos=length)
            )
            # |h_t| = (current_batch_size, 1, hidden_size)
            if fab_prevs[0] is None:
                fab_prevs[0] = h_t
            else:
                fab_prevs[0] = torch.cat([fab_prevs[0], h_t], dim=1)

            for layer_index, block in enumerate(self.decoder._modules.values()):
                prev = fab_prevs[layer_index]
                # |prev| = (current_batch_size, m, hidden_size)

                h_t, _, _, _, _ = block(h_t, fab_z, fab_mask, prev, None)
                # |h_t| = (current_batch_size, 1, hidden_size)

                if fab_prevs[layer_index + 1] is None:
                    fab_prevs[layer_index + 1] = h_t
                else:
                    fab_prevs[layer_index + 1] = torch.cat(
                        [fab_prevs[layer_index + 1], h_t],
                        dim=1,
                    ) # Append new hidden state for each layer.

            y_hat_t = self.generator(h_t)
            # |y_hat_t| = (batch_size, 1, output_size)

            # |fab_prevs[i][begin:end]| = (beam_size, length, hidden_size)
            cnt = 0
            for board in boards:
                if board.is_done() == 0:
                    begin = cnt * beam_size
                    end = begin + beam_size

                    prev_status = {}
                    for layer_index in range(n_dec_layers + 1):
                        prev_status['prev_state_%d' % layer_index] = fab_prevs[layer_index][begin:end]

                    board.collect_result(y_hat_t[begin:end], prev_status)

                    cnt += 1

            done_cnt = [board.is_done() for board in boards]
            length += 1

        batch_sentences, batch_probs = [], []

        for i, board in enumerate(boards):
            sentences, probs = board.get_n_best(n_best, length_penalty=length_penalty)

            batch_sentences += [sentences]
            batch_probs     += [probs]

        return batch_sentences, batch_probs
