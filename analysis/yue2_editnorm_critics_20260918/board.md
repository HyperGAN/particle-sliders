# Edit-norm critic compare (sn_mlp vs gmix)
updated: Fri Sep 18 10:53:42 2026

- **metal_gmix_t8_w48_l1**: survived=True best_cos=0.9937211126089096 noise_start=4.111171194485255 edit_rms=1.1511279344558716 abort=complete
- **metal_sn_mlp_w128_l1**: survived=True best_cos=0.9916356205940247 noise_start=4.111171194485255 edit_rms=1.1511279344558716 abort=complete
- **female_sn_mlp_w128_l1**: survived=None best_cos=0.744569294154644 noise_start=5.2946077925818305 edit_rms=1.4824901819229126 abort=collapse:cos_pos=0.482<0.5 @ 212
- **female_gmix_t8_w48_l1**: survived=None best_cos=None noise_start=5.2946077925818305 edit_rms=1.4824901819229126 abort=collapse:cos_pos=0.338<0.5 @ 200
