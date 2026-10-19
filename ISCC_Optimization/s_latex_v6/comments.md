I have an issue with the inference accuracy model in equatiin  (2). The proposed expression is a surrogate based on sensing SNR and feature retention, but it is not validated against the actual accuracy of an inference model. Since accuracy is one of the main contributions, I think this needs stronger justification, ideally with results from a real classification task. Is it possible without adding extra pages

The closed-form solutions in Propositions 1–3 are interesting I liked them, but their optimality holds only when the remaining variables are fixed. I think it is useful to clarify this distinction, especially since the overall problem remains nonconvex and is solved using heuristics.

I suggest that residual interference after SIC, decoding order, and feasibility of the SIC constraints should be better justified. The assumptions behind the ZF beamforming model also need clarification.

Regarding the computational complexity the additional variables have closed-form updates, the overall runtime may still increase due to the alternating optimization and repeated evaluations right ?. I suggest reporting the actual computational overhead compared with the original framework.

Finally, the numerical results would be stronger with a direct comparison against an accuracy-aware baseline without joint feature optimization, along with sensitivity to jamming power and sensing quality.