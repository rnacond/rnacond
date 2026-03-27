# Conditional generative models for scRNA data

Conditional generative scRNA models from textual cell descriptions.

To launch a training
```
python3 main.py --model [MODEL] --data [DATA] --num_layers 3 --batch_size 5 --trainer_log_every_n_steps 5 --shuffle_metadata False --block_size 1500  --learning_rate 0.005 --metadata_mask_prob 0.05 --trainer_max_epochs 50  --shuffle False  --sample False --verbose True
```
`[MODEL]` can be one of `decoder`, `encoder-decoder`, `encoder`, `clip`, `diffusion`.

`[DATA]` is a path to a `h5ad` file.

You can overwrite any other argument that is in the `yaml` config files of the models (which are in the folders of the models in `src/`).

You can pass any argument to the `lightning` trainer if its name in the command line starts with `trainer_`.

[Slides](https://drive.google.com/file/d/1vmQJMrHiTb8uT2yFM5g5_GVjld05HSTi/view?usp=sharing)

[Miro Board](https://miro.com/app/board/uXjVLLD-o_k=/?share_link_id=639489781277)

[Google Doc](https://docs.google.com/document/d/1Yk7ZRqrsGHYlCdbN6k_0w0-gCQboKnYNkXU-G-mbPRI/edit?usp=sharing)

[wandb](https://wandb.ai/deeplife?shareProfileType=copy)

[Notion](https://www.notion.so/Conditional-Generative-Models-Experiments-1982dd0947f98072a84dfdafb92b051d?showMoveTo=true&saveParent=true)
