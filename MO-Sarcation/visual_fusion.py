import torch
import torch.nn as nn
import torch.nn.functional as F

from context_aware_attention import ContextAwareAttention

SOURCE_MAX_LEN = 768 # 500
VISUAL_DIM = 768 # 2048
VISUAL_MAX_LEN = 1000 # 480

class MAF_visual(nn.Module):
    def __init__(self,
                dim_model,
                dropout_rate):
        super(MAF_visual, self).__init__()
        self.dropout_rate = dropout_rate

        # self.acoustic_context_transform = nn.Linear(ACOUSTIC_MAX_LEN, SOURCE_MAX_LEN, bias = False)
        self.visual_context_transform = nn.Linear(VISUAL_MAX_LEN, SOURCE_MAX_LEN, bias = False)

        # self.acoustic_context_attention = ContextAwareAttention(dim_model=dim_model,
        #                                                         dim_context=ACOUSTIC_DIM,
        #                                                         dropout_rate=dropout_rate)

        self.visual_context_attention = ContextAwareAttention(dim_model=dim_model,
                                                            dim_context=VISUAL_DIM,
                                                            dropout_rate=dropout_rate)

        # self.acoustic_gate = nn.Linear(2*dim_model, dim_model)
        self.visual_gate = nn.Linear(2*dim_model, dim_model)
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.final_layer_norm = nn.LayerNorm(dim_model)

    def forward(self,
                text_input,
                visual_context):

        if visual_context.dim() == 2:
            visual_context = visual_context.unsqueeze(1)

        # Adjust the dimensions of visual_context to match the expected input shape
        visual_context = visual_context.repeat(1, 1, VISUAL_MAX_LEN // visual_context.shape[-1] + 1)[:, :, :VISUAL_MAX_LEN]

        # print("Acoustic context shape (A) : ", acoustic_context.shape)

        # acoustic_context = acoustic_context.permute(0,2,1)
        # acoustic_context = self.acoustic_context_transform(acoustic_context.float())
        # acoustic_context = acoustic_context.permute(0,2,1)

        # audio_out = self.acoustic_context_attention(q=text_input,
        #                                             k=text_input,
        #                                             v=text_input,
        #                                             context=acoustic_context)
        # print("Audio out (A) : ", audio_out.shape)

        # print("Visual context shape : ", visual_context.shape)
        # visual_context = visual_context.permute(0,2,1)
        visual_context = visual_context.reshape(-1, VISUAL_MAX_LEN)
        visual_context = self.visual_context_transform(visual_context.float())
        visual_context = visual_context.view(text_input.size(0), -1, SOURCE_MAX_LEN)
        # visual_context = visual_context.permute(0,2,1)

        video_out = self.visual_context_attention(q=text_input,
                                                    k=text_input,
                                                    v=text_input,
                                                    context=visual_context)

        # print("Video out shape : ", video_out.shape)
        # print("Text input shape : ", text_input.shape)
        # weight_a = F.sigmoid(self.acoustic_gate(torch.cat([text_input, audio_out], dim=-1)))
        weight_v = F.sigmoid(self.visual_gate(torch.cat([text_input, video_out], dim=-1)))

        # output = self.final_layer_norm(text_input + weight_a * audio_out + weight_v * video_out)

        output = self.final_layer_norm(text_input  + weight_v * video_out)

        return output