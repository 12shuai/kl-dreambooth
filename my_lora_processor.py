from diffusers.models.cross_attention import LoRACrossAttnProcessor
import torch

class MyLoRACrossAttnProcessor(LoRACrossAttnProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

        query = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states)
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        key = attn.to_k(encoder_hidden_states) + scale * self.to_k_lora(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states) + scale * self.to_v_lora(encoder_hidden_states)

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
    



class EvalMyLoRACrossAttnProcessor(LoRACrossAttnProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, scale=1.0):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length)

        lora_query=scale * self.to_q_lora(hidden_states)
        # lora_query=0
        
        query = attn.to_q(hidden_states) + lora_query
        
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        
        lora_key=scale * self.to_k_lora(encoder_hidden_states)
        # lora_key=0
        lora_value=scale * self.to_v_lora(encoder_hidden_states) ##4, 4096, 320
        # lora_value=0
        # print(lora_value)
        # print(self.to_v_lora,type(self.to_v_lora)) ## In  and out
        # exit(0)# print(lora_value.sum())

        key = attn.to_k(encoder_hidden_states) + lora_key
        value = attn.to_v(encoder_hidden_states) + lora_value

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)


        lora_out=scale * self.to_out_lora(hidden_states)
        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + lora_out
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states