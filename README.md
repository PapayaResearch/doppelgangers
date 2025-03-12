# Contrastive Learning from Synthetic Audio Doppelgängers

![Python version](https://img.shields.io/badge/python-3.10-blue)
![Package version](https://img.shields.io/badge/version-0.1.0-green)
![GitHub license](https://img.shields.io/github/license/PapayaResearch/doppelgangers)
[![Website](https://img.shields.io/badge/website-DOPPELGANGERS-red)](https://doppelgangers.media.mit.edu/)

> [!NOTE]
> Code for the **ICLR 2025** paper **[Contrastive Learning from Synthetic Audio Doppelgängers](https://arxiv.org/abs/2406.05923)**.

By randomly perturbing the parameters of a sound synthesizer ([SynthAX](https://github.com/PapayaResearch/synthax)), we generate synthetic positive pairs with causally manipulated variations in timbre, pitch, and temporal envelopes. These variations, difficult to achieve through augmentations of existing audio, provide a rich source of contrastive information. Despite the shift to randomly generated synthetic data, our method produces strong audio representations, outperforming real data on several standard audio classification tasks.

> [!TIP]
> You can hear examples on our [website](https://doppelgangers.media.mit.edu/).

> [!WARNING]
> The code for the evaluations from the paper will be found in a different [repository](https://github.com/PapayaResearch/doppelgangers-experiments) (coming soon).

## Installation

You can create the environment as follows

```bash
conda create -n doppelgangers python=3.10
conda activate doppelgangers
pip install -r requirements.txt
```

By default, we use CUDA 12.1, but you can change the requirements.

## Training
Training with audio doppelgängers is simple

```bash
python train.py embedding=resnet synth=voice data.synthetic.delta=0.25 general.epochs=200
```

It will generate directories containing logs and outputs.

## Configuration
> [!IMPORTANT]
> We use [Hydra](https://hydra.cc/) to configure `doppelgängers`. The configuration can be found in `conf/config.yaml`, with specific sub-configs in sub-directories of `conf/`.

The configs define all the parameters (e.g. embedding, synthesizer, transformations). By default, these are the ones used for the paper. The only `embedding` for now is ResNet, but you can choose a `synth` architecture and a `synthconfig`. This is also where you choose the `transform` if you train with real data. Other important parameters are the `data.synthetic.delta` value for the doppelgängers, the `data.batch_size`, whether `data.apply_transform` or not, the `data.duration` of the synthetic sounds, the `data.sample_rate` of the synthetic sounds, whether to use `data.temporal_jitter` or not, the number of `data.n_layers` of sounds to stack together, the number of `general.epochs`, or the initial random `system.seed`.

## Acknowledgements & Citing

If you use `doppelgängers` in your research, please cite the following paper:
```bibtex
@inproceedings{cherep2024contrastive,
  title={Contrastive Learning from Synthetic Audio Doppelgängers},
  author={Cherep, Manuel and Singh, Nikhil},
  booktitle={Thirteenth International Conference on Learning Representations},
  year={2025}
}
```

For the synthesizer component itself, please cite [SynthAX](https://github.com/PapayaResearch/synthax):
```bibtex
@conference{cherep2023synthax,
  title = {SynthAX: A Fast Modular Synthesizer in JAX},
  author = {Cherep, Manuel and Singh, Nikhil},
  booktitle = {Audio Engineering Society Convention 155},
  month = {May},
  year = {2023},
  url = {http://www.aes.org/e-lib/browse.cfm?elib=22261}
}
```

Manuel received the support of a fellowship from “la Caixa” Foundation (ID 100010434). The fellowship code is LCF/BQ/EU23/12010079. The authors acknowledge the MIT SuperCloud and Lincoln Laboratory Supercomputing Center for providing resources that have contributed to the research results reported within this paper.
