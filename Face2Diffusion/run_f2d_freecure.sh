python inference_f2d.py \
	--w_map ./checkpoints/mapping.pt \
	--w_msid ./checkpoints/msid.pt\
	-i ./input/1.jpg \
	-p "f l with blonde curly hair, wearing sunglasses" \
	-g "woman" \
	-o ./outputs \
	-n 1
	