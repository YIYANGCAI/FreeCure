def visualize_attention_map(attn, viz_cfg, cur_step, cur_att_layer):
	"""
	Visualize the attention map only in the multi-reference self-attention (MRSA) module.
	Input:
		viz_cfg: includes generated image and reference images info.
		cur_step: viz at current denoising step.
		cur_att_layer: viz at current attention layer.
	"""
	# set the dir to save all attention map
	attn_map_save_dir = os.path.join(viz_cfg.results_dir, 'attention_maps', f'step_{cur_step}, layer_{cur_att_layer}')
	os.makedirs(attn_map_save_dir, exist_ok=True)
	
	save_res = 512 # save attention map to the specified image size
	H = W = cur_res = int(attn.shape[1] ** 0.5) # resolution of current feature(q,k,v)

	# set image paths includes generated image and input images
	image_paths = [viz_cfg.generated_image_path]
	image_paths = image_paths + list(viz_cfg.ref_image_infos)
	
	generated_image = cv2.resize(cv2.imread(viz_cfg.generated_image_path), (save_res, save_res), interpolation=cv2.INTER_NEAREST)

	attn_mean = attn.mean(0) # Merge the attention of 8 heads, attn_mean shape: (H*W, H*W)
	# visualize the attention maps, for simplicity, we dont viz all of them
	for i in range(0, attn_mean.shape[0], 10): 
		# We have H*W query pixels and corresponding H*W attention map, the i-th is the index of the current query pixel.
		pixels = np.zeros(H * W , dtype=np.uint8)
		pixels[i] = 1
		pixels = pixels.reshape(H, W)
		pixels = cv2.resize(pixels, (save_res, save_res), interpolation=cv2.INTER_NEAREST)
		# set the query pixel within the generated image to white color
		query_pixel_on_generated_image = generated_image.copy()
		query_pixel_on_generated_image[pixels > 0] = [255, 255, 255] 
		
		# processing rgb images and attention maps
		images = []
		attention_maps = []
		for j, image_path in enumerate(image_paths):
			attn_part = attn_mean[i][H*W*j: H*W*(j+1)]
			attention_map = attn_part.reshape(cur_res, -1).clone().cpu().numpy()

			image = cv2.resize(cv2.imread(image_path), (save_res, save_res), interpolation=cv2.INTER_NEAREST)
			attention_map = cv2.resize(attention_map, (save_res, save_res), interpolation=cv2.INTER_LINEAR)
			
			images.append(image)
			attention_maps.append(attention_map)
		
		# show multi reference self attention map on the generated image and input images
		images = np.concatenate(images, axis=1)
		attention_maps = np.concatenate(attention_maps, axis=1)
		attn_maps_on_images = show_cam_on_image(images / 255., attention_maps / attention_maps.max(), use_rgb=False, colormap=cv2.COLORMAP_JET)
		/home/ycaicj/workspaces/FreeCure/freecure/attribute_master.py
		# save the generated image and attention maps
		curr_attn_map = np.concatenate((query_pixel_on_generated_image, attn_maps_on_images), axis=1)
		cv2.imwrite(os.path.join(attn_map_save_dir, f'{i}.png'), curr_attn_map)
		
def visualize_correspondence(viz_cfg, attn, cur_step, cur_att_layer):
	"""
	Visualize the feature correspondence in the multi-reference self-attention (MRSA) module.
	Input:
		viz_cfg: includes generated image and reference images info.
		cur_step: viz at current denoising step.
		cur_att_layer: viz at current attention layer.
	"""
	cur_res  = int(attn.shape[1] ** 0.5) # current resolution of q,k,v feature
	
	attn_mean = attn.mean(0).cpu().numpy() 
	attn_refs = attn_mean / attn_mean.max()
	ref_num   = attn_refs.shape[-1] // (cur_res * cur_res)  # number of input reference images + 1
	
	# providing a mask for the generated combined concept is suggested
	if viz_cfg.generated_mask_path == '':
		mask_gen = np.ones((cur_res,cur_res), dtype=int)
	else:
		mask_gen = cv2.imread(viz_cfg.generated_mask_path, -1) // 255

	def polar_color_map(image_size, mask=None):
		"""
		generate a square polar colormap.
		"""
		x, y = np.meshgrid(np.arange(image_size), np.arange(image_size))
		coords = np.stack((x, y), axis=-1)
		
		angle = (np.arctan2(coords[..., 1] - image_size // 2, coords[..., 0] - image_size // 2) + np.pi) % (2 * np.pi) # calculate angle
		normalized_angle = angle / (2 * np.pi) # normalize angle
		
		cmap = cm.get_cmap('hsv', 2056) # get the polar colormap values
		bgr_colors = (cmap(normalized_angle)[:, :, :3] * 255).astype(np.uint8)
	
		hsv_colors = cv2.cvtColor(bgr_colors, cv2.COLOR_BGR2HSV)

		saturation_factor = 0.5  # control the level of saturation
		hsv_colors[..., 1] = hsv_colors[..., 1] * saturation_factor

		bgr_colors = cv2.cvtColor(hsv_colors, cv2.COLOR_HSV2BGR)
		rgb_colors = cv2.cvtColor(bgr_colors, cv2.COLOR_BGR2RGB)

		if mask is not None:
			mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
			rgb_colors[mask == 0] = [255, 255, 255]

		return rgb_colors
	
	colormap = polar_color_map(cur_res, mask_gen) # a square polar color map

	max_attn_value_indices = np.argmax(attn_refs, axis=0)  # the index of the maximum attention value corresponding to each query feature
	corrspondence = colormap.reshape(-1, 3)[max_attn_value_indices] # core of this function, pick up pixel color from the colormap according to the the index of the maximum attention value.
	corrspondence = corrspondence.reshape(-1, cur_res, 3)
	corrspondence = np.concatenate(np.split(corrspondence, ref_num)[1:], axis=1) # [1:] means that we only visualize the correspondence betweent the generated image and the reference images.
	
	save_res = 512
	cmap_resize = cv2.resize(colormap, (save_res, save_res), interpolation=cv2.INTER_NEAREST)
	corr_resize = cv2.resize(corrspondence, (save_res*(ref_num-1), save_res), interpolation=cv2.INTER_NEAREST)
	cmap_corr_resize = np.concatenate((cmap_resize, corr_resize), axis=1)

	corr_save_dir = os.path.join(viz_cfg.results_dir, 'feature_correspondences')
	os.makedirs(corr_save_dir, exist_ok=True)
	cv2.imwrite(os.path.join(corr_save_dir, f'step_{cur_step}, layer_{cur_att_layer}.png'), cmap_corr_resize)
