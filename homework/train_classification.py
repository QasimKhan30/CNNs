import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb
import torchvision

from .models import load_model, ClassificationLoss, save_model, Detector
from .metrics import AccuracyMetric


from .datasets.classification_dataset import load_data

# https://discuss.pytorch.org/t/does-pytorch-use-tensor-cores-by-default/167676/3
# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

"""
python3 -m homework.train_classification --model_name classifier
"""


def train(
    exp_dir: str = "logs",
    model_name: str = "classifier",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()


    train_data = load_data("classification_data/train", transform_pipeline='default', shuffle=True, batch_size=batch_size, num_workers=0)
    val_data = load_data("classification_data/val", transform_pipeline='default', shuffle=False)


    # logger.add_graph(model,torch.zeros(1,3,*size))
    # logger.add_images("train_images", torch.stack([train_data[i][0] for i in range(32)]))
    # logger.flush()
    # model_view = Detector()
    # model_graph = draw_graph(model=model_view, input_data=torch.zeros(1,3,64,64), depth=1)
    # model_graph.visual_graph



    # create loss function and optimizer
    loss_func = ClassificationLoss()
    # optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    # optimizer = torch.optim.SGD(params=model.parameters(),lr=lr,momentum=0.9, weight_decay=1e-2)
    optimizer = torch.optim.AdamW(params=model.parameters(),lr=lr,weight_decay=1e-3)

    global_step = 0

    # create metric calculation objects
    train_accuracy_metric = AccuracyMetric()
    val_accuracy_metric = AccuracyMetric()

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        train_accuracy_metric.reset()
        val_accuracy_metric.reset()

        model.train()

        for img, label in train_data:
            
            
            img, label = img.to(device), label.to(device)

            pred_logits = model(img)
            loss = loss_func(pred_logits,label)
            logger.add_scalar('train_loss',loss,global_step)

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1

            # Calculate training accuracy and append to metrics for each image
            predicted_labels = torch.argmax(pred_logits,dim=1)
            train_accuracy_metric.add(predicted_labels, label)

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for img, label in val_data:
                img, label = img.to(device), label.to(device)

                pred_logits = model(img)

                # Compute validation accuracy
                predicted_labels = torch.argmax(pred_logits,dim=1)
                
                val_accuracy_metric.add(predicted_labels, label)

        # compute accuracy for the epoch
        epoch_train_acc = train_accuracy_metric.compute()["accuracy"]
        epoch_val_acc = val_accuracy_metric.compute()["accuracy"]

        # log average train and val accuracy to tensorboard
        logger.add_scalar('train_accuracy',epoch_train_acc,epoch)
        logger.add_scalar('val_accuracy',epoch_val_acc,epoch)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )

        # if epoch_val_acc >= 0.90:
        #     print(f"Early stopping: validation accuracy {epoch_val_acc} >= 82%")
        #     break
    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=4)
    # parser.add_argument("--batch_size", type=int, default=128)
    # parser.add_argument("--hidden_dim", type=int, default=128)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
