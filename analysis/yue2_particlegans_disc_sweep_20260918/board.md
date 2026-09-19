# ParticleGAN discriminator sweep
updated: Fri Sep 18 05:50:34 2026

- **patch_p64_w48_l1** (patch): survived=True collapse@None best_cos=0.9959711804986 @1305 final_cos=0.9915312379598618 abort=complete
- **mix_t8_w48_l1** (mix): survived=True collapse@None best_cos=0.9944480881094933 @1349 final_cos=0.9785645604133606 abort=complete
- **hybrid_t16_w48_l1** (hybrid): survived=True collapse@None best_cos=0.9919231161475182 @1451 final_cos=0.9825181141495705 abort=complete
- **hybrid_t8_w48_l1** (hybrid): survived=None collapse@485 best_cos=0.9775136113166809 @368 final_cos=0.7392833866178989 abort=collapse:cos_pos=0.078<0.5 @ 485
- **lowrank_r64_t8_w48_l1** (lowrank): survived=None collapse@471 best_cos=0.9825925976037979 @440 final_cos=0.08319433778524399 abort=collapse:cos_pos=0.098<0.5 @ 472
- **query_t16_q8_w48_l1** (query): survived=None collapse@435 best_cos=0.9751767292618752 @235 final_cos=0.7882087826728821 abort=collapse:cos_pos=0.307<0.5 @ 437
- **bottleneck_t8_w48_l2** (bottleneck): survived=None collapse@341 best_cos=0.9438723996281624 @228 final_cos=0.5978220887482166 abort=collapse:cos_pos=0.467<0.5 @ 344
- **mlp** (mlp): survived=None collapse@330 best_cos=0.9929538294672966 @1305 final_cos=0.9851778447628021 abort=complete
- **bquery_t16_q8_w48_l1** (bquery): survived=None collapse@317 best_cos=0.9393423646688461 @266 final_cos=0.2441887187305838 abort=collapse:cos_pos=0.145<0.5 @ 330
- **bquery_t8_q4_w48_l1** (bquery): survived=None collapse@204 best_cos=0.6383844055235386 @201 final_cos=0.3162842314923182 abort=collapse:cos_pos=0.116<0.5 @ 205
- **bottleneck_t16_w48_l1** (bottleneck): survived=None collapse@201 best_cos=0.5749634653329849 @202 final_cos=0.5749634653329849 abort=collapse:cos_pos=0.495<0.5 @ 201
- **lowrank_r32_t8_w48_l1** (lowrank): survived=None collapse@200 best_cos=None @None final_cos=0.22506960853934288 abort=collapse:cos_pos=0.106<0.5 @ 201
