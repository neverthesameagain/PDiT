import torch #type:ignore
import torch.nn as nn #type:ignore
import torch.nn.functional as F #type:ignore
from transformers import CLIPModel, CLIPProcessor #type:ignore

class Patch_State(nn.Module):

    def __init__(self, config):
        super(Patch_State, self).__init__()

        self.config = config
        self.max_len = config['max_len']
        self.num_patches = config['max_len'] + config['img_tokens']
        self.emd_dim = config['emd_dim']
        self.pos_emd = nn.Parameter(torch.randn(self.num_patches, self.emd_dim), requires_grad=True)
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.proj = nn.Linear(768, self.config['emd_dim'])
        self.embed_image = nn.Linear(49, 768)
        self.projtext = nn.Linear(512, self.config['emd_dim'])

    def forward(self, images, prompt):
        
        #images are of shape (bsz, seq_len, 224, 224, 3) / ((bsz, seq_len, 7, 7, 3))
        #prompt are of shape (bsz, 1)
        
        bsz = images.size(0)
        seq_len = images.size(1)
        
        if self.config['use_clip']:
            
            images = images.reshape(bsz * seq_len, 224, 224, 3)
            image_inputs = self.processor(images=images, return_tensors="pt").to(self.config['device'])
            vision_outputs = self.model.vision_model(**image_inputs).last_hidden_state
            del image_inputs
        else:            
            images = images.reshape(bsz, seq_len, -1, images.size(-1)).permute([0, 1, 3, 2])
            vision_outputs = self.embed_image(images)        
        
        prompt_inputs = self.processor(text=prompt[0], return_tensors="pt", padding=True).to(self.config['device'])        
        with torch.no_grad():
            text_outputs = self.model.text_model(**prompt_inputs).last_hidden_state
        
        del prompt_inputs
        
        vision_outputs = self.proj(vision_outputs)
        if self.config['emd_dim'] != 512:
            text_outputs = self.projtext(text_outputs)
        
        image_embds = vision_outputs.reshape(bsz, seq_len, -1, self.config['emd_dim'])
        prompt_embds = text_outputs.unsqueeze(1).repeat(1, seq_len, 1, 1)
        
        # Assuming prompt_embds has shape (bsz, seq_len, num_tokens, emd_dim)
        prompt_mask = torch.stack([torch.cat([torch.ones(emb.shape[0], emb.shape[1]), torch.zeros(emb.shape[0], self.max_len - emb.shape[1])], dim=1)for emb in prompt_embds]).to(self.config['device'])
        prompt_embds = torch.stack([torch.cat([emb, torch.zeros(emb.shape[0], self.max_len - emb.shape[1], self.emd_dim).to(self.config['device'])], dim=1)for emb in prompt_embds])            
        
        # Assuming prompt_embds has shape (bsz, seq_len, num_tokens, emd_dim)
        state_embds = torch.cat([image_embds, prompt_embds], axis=2)
        
        # state_embds are of shape (bsz, seq_len, num_tokens, emd_dim)
        # pos_emd is of shape (num_tokens, emd_dim)
        state_embds += self.pos_emd
 
        return state_embds, prompt_mask
    
class Patch_RA(nn.Module):
    
    def __init__(self, config):
        super(Patch_RA, self).__init__()
        self.embd_rewards = nn.Linear(1, config['emd_dim']) 
        self.embd_actions = nn.Linear(config['act_dim'], config['emd_dim'])
        
    def forward(self, rewards, actions, time_embds):
        
        rewards = self.embd_rewards(rewards) + time_embds
        actions = self.embd_actions(actions) + time_embds
        
        return rewards, actions
    
class MultiHead(nn.Module):

    def __init__(self, emd_dim, d_model, head, is_causal):
        super(MultiHead, self).__init__()

        self.d_model = d_model
        self.head = head
        self.is_causal = is_causal
        self.qmat = nn.Linear(emd_dim, d_model)
        self.kmat = nn.Linear(emd_dim, d_model)
        self.vmat = nn.Linear(emd_dim, d_model)
        self.omat = nn.Linear(d_model, emd_dim)

    def make_heads(self, x):
        return x.view(x.size()[0], x.size()[1], self.head, self.d_model // self.head).transpose(1, 2)

    def forward(self, x, attn_mask=None):

        q, k, v = self.qmat(x), self.kmat(x), self.vmat(x)
        q, k, v = self.make_heads(q), self.make_heads(k), self.make_heads(v)
        
        x = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=self.is_causal, attn_mask=attn_mask) # flash attn go brrr
        x = x.transpose(1, 2).contiguous().view(x.size()[0], -1, self.d_model)

        return self.omat(x)

class MLP(nn.Module):
    
    def __init__(self, emd_dim):
        super(MLP, self).__init__()
        self.emd_dim = emd_dim
        self.ff = nn.Sequential(
            nn.Linear(self.emd_dim, self.emd_dim * 2),
            nn.GELU(),
            nn.Linear(self.emd_dim * 2, self.emd_dim)
        )
        
    def forward(self, input_tensor): 
        return self.ff(input_tensor)

class BLOCK(nn.Module):

    def __init__(self, emd_dim, d_model, heads, is_causal):
        super(BLOCK, self).__init__()

        self.norm1 = nn.LayerNorm(emd_dim)
        self.multihead = MultiHead(emd_dim, d_model, heads, is_causal)
        self.norm2 = nn.LayerNorm(emd_dim)
        self.ff = MLP(emd_dim)

    def forward(self, x, attn_mask=None):
        
        x = x + self.multihead(self.norm1(x), attn_mask)
        x = x + self.ff(self.norm2(x))
        
        return x
        
class PerceptionTransformer(nn.Module):
    
    def __init__(self, config):
        super(PerceptionTransformer, self).__init__()

        self.config = config
        self.emd_dim = config['emd_dim']
        self.d_model = config['d_model']
        self.heads = config['heads']
        self.block = BLOCK(emd_dim=self.emd_dim, d_model=self.d_model, heads=self.heads, is_causal=False)
        self.ln = nn.LayerNorm(self.emd_dim)
        
    def forward(self, x, attention_mask, valid_ind):
        
        bsz, seq_len = x.size(0), x.size(1)
        x = x.reshape(bsz * seq_len, -1, x.size(-1))
        x = self.ln(x)
        x = self.block(x, attention_mask)
    
        x = x.reshape(bsz, seq_len, -1, self.emd_dim).permute(0, 2, 1, 3)
        state_embeddings = torch.gather(x, 1, valid_ind).squeeze().unsqueeze(0) # shall I unsqueeze it or let it be like that????
        
        return state_embeddings, x.permute(0, 2, 1, 3)
    
class DecisionTransformer(nn.Module):
    
    def __init__(self, config):
        super(DecisionTransformer, self).__init__()
        
        self.config = config
        self.emd_dim = config['emd_dim']
        self.d_model = config['d_model']
        self.heads = config['heads']
        self.block = BLOCK(emd_dim=self.emd_dim, d_model=self.d_model, heads=self.heads, is_causal=False) #making the causal by hand
        self.embed_ln = nn.LayerNorm(config['emd_dim'])
        self.predict_action = nn.Linear(config['emd_dim'], config['act_dim']) 
        
    def forward(self, state_embds, R_embds, actions_embds, attention_mask):
                
        bsz, seq_length = state_embds.size(0), state_embds.size(1)
        
        stacked_inputs = torch.stack(
            (R_embds, state_embds, actions_embds), dim=1
        ).permute(0, 2, 1, 3).reshape(bsz, -1, self.emd_dim)
        stacked_inputs = self.embed_ln(stacked_inputs)

        x = self.block(stacked_inputs, attention_mask)
        x = x.reshape(bsz, seq_length, -1, self.emd_dim).permute(0, 2, 1, 3)

        return self.predict_action(x[:,1])   
class CPDIT(nn.Module):
    
    def __init__(self, config):
        super(CPDIT, self).__init__()
        
        self.max_ep_len = config['max_ep_len']
        self.hidden_size = config['emd_dim']
        self.max_len = config['max_len']
        self.config = config
        self.patch_state = Patch_State(config)
        self.patch_actions_rewards = Patch_RA(config)
        self.perception_blocks = nn.ModuleList([PerceptionTransformer(config) for _ in range(config['n_layers'])])
        self.decision_blocks = nn.ModuleList([DecisionTransformer(config) for _ in range(config['n_layers'])])
        self.embed_timestep = nn.Embedding(self.max_ep_len, self.hidden_size)
        
        self.proj = nn.Sequential(
         
            nn.Linear(config['n_layers'], 64),
            nn.GELU(),
            nn.Linear(64, 1)
            
        )
        
    def create_mask(self, prompt_mask, attention_mask):
        
        bsz, seq_len = prompt_mask.size(0), prompt_mask.size(1)
        image_token_num = self.config['img_tokens']
        
        filler = torch.ones_like(prompt_mask[:, :, 0]).unsqueeze(2).repeat(1, 1, image_token_num)
        attention_mask_state = torch.cat([filler, prompt_mask], dim=2).reshape(bsz * seq_len, -1)
        
        attention_mask = torch.stack(([attention_mask for _ in range(3)]), dim=1).permute(0, 2, 1).reshape(bsz, 3 * seq_len)

        # such a cheap way to do this
        attention_mask = attention_mask.unsqueeze(1)
        attention_mask = attention_mask.unsqueeze(-1)
        attention_mask = attention_mask.repeat(1, self.config['heads'], 1, seq_len * 3).to(self.config['device'])

        attention_mask_state = attention_mask_state.unsqueeze(1)
        attention_mask_state = attention_mask_state.unsqueeze(-1)
        attention_mask_state = attention_mask_state.repeat(1, self.config['heads'], 1, self.config['img_tokens'] + self.config['max_len']).to(self.config['device'])
        
        causal_mask = torch.tril(torch.ones(seq_len * 3, seq_len * 3)).to(self.config['device'])
        
        attention_mask = attention_mask * causal_mask
        
        return attention_mask, attention_mask_state
    
    def forward(self, images, prompt, actions, Gt, timesteps, mask, target=None):
        
        time_embds = self.embed_timestep(timesteps)
        
        R_embds, actions_embds = self.patch_actions_rewards(Gt, actions, time_embds)
        obs, prompt_mask = self.patch_state(images, prompt)        
        attention_mask, attention_mask_state = self.create_mask(prompt_mask, mask)
        
        filler = torch.ones_like(prompt_mask[:,:,0]).unsqueeze(-1).repeat(1, 1, self.config['img_tokens'])
        valid_ind = torch.cat(
            [filler, prompt_mask], dim=2
        ).permute(0, 2, 1).sum(dim=1, keepdim=True).long() - 1
        valid_ind = valid_ind.unsqueeze(-1).repeat([1, 1, 1, self.config['emd_dim']]).to(self.config['device'])
        
        actions_pred = []
        for i in range(self.config['n_layers']):
            
            state_embds, obs = self.perception_blocks[i](obs, attention_mask_state, valid_ind)
            state_embds += time_embds
            action = self.decision_blocks[i](state_embds.squeeze(0), R_embds, actions_embds, attention_mask)
            actions_pred.append(action)
            del state_embds
            
        actions_pred = torch.stack(actions_pred, dim=1)
        actions_pred = actions_pred.permute(0, 2, 3, 1).reshape(-1, self.config['act_dim'], self.config['n_layers'])
        
        actions_pred = self.proj(actions_pred).squeeze(-1).reshape(-1, self.config['cntxt'], self.config['act_dim'])
        
        final_preds = actions_pred.reshape(-1, self.config['act_dim'])
        
        loss = None
        if target is not None:
            target = target.max(-1)[1].reshape(-1)
            loss = F.cross_entropy(final_preds, target)
            
        return loss, final_preds