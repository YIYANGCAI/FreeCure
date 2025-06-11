import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
from einops import rearrange, repeat

def trace_attention_module(model_name, model, key_words):
	trace_result = {}
	def trace_attention_module_once(module_name, module, key_words):
		for name, sub_module in module.named_children():
			if sub_module.__class__.__name__ == 'Attention' and key_words in (module_name + '.' + name):
				trace_result[module_name + '.' + name] =len(trace_result)
			elif hasattr(sub_module, 'children'):
				trace_attention_module_once(module_name + '.' + name, sub_module, key_words)
	trace_attention_module_once(model_name, model, key_words)
	for k,v in trace_result.items():
		print(f"{k} - {v}")
	return trace_result

def register_fasa_to_model(model, fasa, net_name=None):
	"""
	Hack the original self-attention module to foundation aware self attention mechanism
	"""
	def fasa_forward(self, place_in_unet):
		def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None, **kwargs):
			"""
			The msra is similar to the original implementation of LDM CrossAttention class
			except adding some modifications on the attention
			"""
			# import pdb;pdb.set_trace()
			if encoder_hidden_states is not None:
				context = encoder_hidden_states
			if attention_mask is not None:
				mask = attention_mask

			to_out = self.to_out
			if isinstance(to_out, nn.modules.container.ModuleList):
				to_out = self.to_out[0]
			else:
				to_out = self.to_out

			h = self.heads
			q = self.to_q(x)
			is_cross = context is not None
			context = context if is_cross else x
			k = self.to_k(context)
			v = self.to_v(context)
			q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

			sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

			if mask is not None:
				mask = rearrange(mask, 'b ... -> b (...)')
				max_neg_value = -torch.finfo(sim.dtype).max
				mask = repeat(mask, 'b j -> (b h) () j', h=h)
				mask = mask[:, None, :].repeat(h, 1, 1)
				sim.masked_fill_(~mask, max_neg_value)

			attn = sim.softmax(dim=-1)
			# the only difference
			out = fasa(
				q, k, v, sim, attn, is_cross, place_in_unet,
				self.heads, scale=self.scale, **kwargs)

			return to_out(out)

		return forward
	
	def hack_attention_module(net, count, place_in_unet, net_name):
		for name, subnet in net.named_children():
			if net.__class__.__name__ == 'Attention':
				net.forward = fasa_forward(net, place_in_unet)
				print(net_name + '.' + name)
				return count + 1
			elif hasattr(net, 'children'):
				count = hack_attention_module(subnet, count, place_in_unet, net_name + '.' + name)
		return count
  
	cross_att_count = 0
	for net_name, net in model.unet.named_children():
		if "down" in net_name:
			cross_att_count += hack_attention_module(net, 0, "down", 'pipe.unet.down_blocks')
		elif "mid" in net_name:
			cross_att_count += hack_attention_module(net, 0, "mid", 'pipe.unet.mid_blocks')
		elif "up" in net_name:
			cross_att_count += hack_attention_module(net, 0, "up", 'pipe.unet.up_blocks')
	fasa.num_att_layers = cross_att_count

class FoundationAwareSelfAttention():
	def __init__(self,  start_step=0, end_step=50, step_idx=None, layer_idx=None, ref_masks=None, mask_weights=[1.0,1.0,1.0], style_fidelity=1, viz_cfg=None):
		"""
		Args:
			start_step   : the step to start transforming self-attention to multi-reference self-attention
			end_step	 : the step to end transforming self-attention to multi-reference self-attention
			step_idx	 : list of the steps to transform self-attention to multi-reference self-attention
			layer_idx	: list of the layers to transform self-attention to multi-reference self-attention
			ref_masks	: masks of the input reference images
			mask_weights : mask weights for each reference masks
			viz_cfg	  : config for visualization
		"""
		self.cur_step	   =  0
		self.num_att_layers = -1
		self.cur_att_layer  =  0

		self.start_step   = start_step
		self.end_step	 = end_step
		self.step_idx	 = step_idx if step_idx is not None else list(range(start_step, end_step))
		self.layer_idx	= layer_idx
		
		self.ref_masks	= ref_masks
		self.mask_weights = mask_weights
		
		self.style_fidelity = style_fidelity

		self.viz_cfg = viz_cfg
	   
	def __call__(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
		# import pdb;pdb.set_trace()
		out = self.mrsa_forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
		self.cur_att_layer += 1
		if self.cur_att_layer == self.num_att_layers:
			self.cur_att_layer = 0
			self.cur_step += 1
		return out
	
	def get_ref_mask(self, ref_mask, mask_weight, H, W):
		# 注意在这里由于face2diffusion使用了float16，所以这里要去掉.float()操作
		# ref_mask = ref_mask.float() * mask_weight
		# 这里 mask 自适应 attention map的大小
		ref_mask = ref_mask * mask_weight
		ref_mask = F.interpolate(ref_mask, (H, W))
		ref_mask = ref_mask.flatten()
		return ref_mask
	
	def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
		# import pdb;pdb.set_trace()
		B = q.shape[0] // num_heads
		H = W = int(np.sqrt(q.shape[1]))
		# how to rearrange the tensor
		q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads) 
		k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
		v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

		sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
		# import pdb;pdb.set_trace()
		if kwargs.get("attn_batch_type") == 'mrsa':
			sim_own, sim_refs = sim[..., :H*W], sim[..., H*W:]
			sim_or = [sim_own]
			for i, (ref_mask, mask_weight) in enumerate(zip(self.ref_masks, self.mask_weights)):
				ref_mask = self.get_ref_mask(ref_mask, mask_weight, H, W)
				sim_ref = sim_refs[..., H*W*i: H*W*(i+1)]
				# 精度对齐, 这个判断语句在使用sdxl模型时为true, 在使用sdv1.4/5则为false
				if 'float16' in str(ref_mask.dtype) and 'bfloat16' in str(sim_ref.dtype):
					ref_mask = ref_mask.to(torch.bfloat16)
				sim_ref = sim_ref + ref_mask.masked_fill(ref_mask == 0, torch.finfo(sim.dtype).min)
				# attention! fp16 minimum value: -65504.0; bf16 minimum value: 
				sim_or.append(sim_ref)
			sim = torch.cat(sim_or, dim=-1)
		attn = sim.softmax(-1)
		
		# viz attention map within MRSA module
		# if self.viz_cfg.viz_attention_map == True and \
		# 	kwargs.get("attn_batch_type") == 'mrsa' and \
		# 	self.cur_step in self.viz_cfg.viz_map_at_step and \
		# 	self.cur_att_layer // 2 in self.viz_cfg.viz_map_at_layer:
		# 	visualize_attention_map(attn, self.viz_cfg, self.cur_step, self.cur_att_layer//2)
		
		# # viz feature correspondence within MRSA module
		# if self.viz_cfg.viz_feature_correspondence == True and \
		# 	kwargs.get("attn_batch_type") == 'mrsa' and \
		# 	self.cur_step in self.viz_cfg.viz_corr_at_step and \
		# 	self.cur_att_layer // 2 in self.viz_cfg.viz_corr_at_layer:
		# 	visualize_correspondence(self.viz_cfg, attn, self.cur_step, self.cur_att_layer//2)
			
		if len(attn) == 2 * len(v):
			v = torch.cat([v] * 2)
		# import pdb;pdb.set_trace()
		out = torch.einsum("h i j, h j d -> h i d", attn, v)
		out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)
		return out  

	def sa_forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
		"""
		Original self-attention forward function
		"""
		out = torch.einsum('b i j, b j d -> b i d', attn, v)
		out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
		return out
	
	def mrsa_forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
		"""
		Mutli-reference self-attention(MRSA) forward function
		"""
		# import pdb;pdb.set_trace()
		if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
			return self.sa_forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
		
		# import pdb;pdb.set_trace()
		B = q.shape[0] // num_heads // 2 # q.shape的含义：(batch * num_head) * seq_len * hidden_state_per_head
		
		qu, qc = q.chunk(2)
		ku, kc = k.chunk(2)
		vu, vc = v.chunk(2)

		# The first batch is the q,k,v feature of $z_t$ (own feature), and the subsequent batches are the q,k,v features of $z_t^'$ (reference featrue)
		qu_o, qu_r = qu[:num_heads], qu[num_heads:] 
		qc_o, qc_r = qc[:num_heads], qc[num_heads:]
		
		ku_o, ku_r = ku[:num_heads], ku[num_heads:]
		kc_o, kc_r = kc[:num_heads], kc[num_heads:]
		
		vu_o, vu_r = vu[:num_heads], vu[num_heads:]
		vc_o, vc_r = vc[:num_heads], vc[num_heads:]
		
		ku_cat, vu_cat = torch.cat([ku_o, *ku_r.chunk(B-1)], 1), torch.cat([vu_o, *vu_r.chunk(B-1)], 1)
		kc_cat, vc_cat = torch.cat([kc_o, *kc_r.chunk(B-1)], 1), torch.cat([vc_o, *vc_r.chunk(B-1)], 1)

		out_u_target = self.attn_batch(qu_o, ku_cat, vu_cat, None, None, is_cross, place_in_unet, num_heads, attn_batch_type='mrsa', **kwargs)
		out_c_target = self.attn_batch(qc_o, kc_cat, vc_cat, None, None, is_cross, place_in_unet, num_heads, attn_batch_type='mrsa', **kwargs)
		
		# The larger the style_fidelity, the more like the reference concepts, range of values: [0,1]
		if self.style_fidelity > 0:
			out_u_target = (1 - self.style_fidelity) * out_u_target + self.style_fidelity * self.attn_batch(qu_o, ku_o, vu_o, None, None, is_cross, place_in_unet, num_heads, **kwargs)

		out = self.sa_forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
		out_u, out_c = out.chunk(2)
		out_u_ref, out_c_ref = out_u[1:], out_c[1:]
		out = torch.cat([out_u_target, out_u_ref, out_c_target, out_c_ref], dim=0)
		
		return out