import os
import torch
import wandb
from pathlib import Path

def get_average_embeddings(embeddings, labels):
    avg_embeddings = []
    labels_list = []

    for label in set(labels.cpu().numpy()):
        labels_list.append(label)
        avg_embeddings.append(
            embeddings[labels==label].mean(dim=0)
        )

    avg_embeddings = torch.stack(avg_embeddings)

    return avg_embeddings, labels_list

def maybe_setup_wandb(args):
    dir_path = Path(args.save_path)
    wandb_entity = os.environ.get("WANDB_ENTITY")
    wandb_project = os.environ.get("WANDB_PROJECT")

    if wandb_entity is None or wandb_project is None:
        print(f"{wandb_entity=}", f"{wandb_project=}")
        print("Not initializing WANDB")
        return

    run_name = '/'.join(dir_path.parts[1:])

    wandb.init(entity=wandb_entity, project=wandb_project, config=args, name=run_name, dir=str(dir_path))

    print("WANDB run", wandb.run.id, run_name)

