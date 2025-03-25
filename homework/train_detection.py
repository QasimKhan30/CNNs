import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import torch.utils.tensorboard as tb

from torchview import draw_graph

from .models import load_model, save_model
from .metrics import DetectionMetric


from .datasets.road_dataset import load_data

# https://discuss.pytorch.org/t/does-pytorch-use-tensor-cores-by-default/167676/3
# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True
# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True

"""
python3 -m homework.train_detection --model_name detector
"""

def dice_loss(pred, target, smooth=1e-6):

    num_classes = pred.shape[1]
    target_one_hot = F.one_hot(target, num_classes=num_classes)  # (B, H, W, C)
    target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()    # (B, C, H, W)
    
    intersection = (pred * target_one_hot).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target_one_hot.sum(dim=(2,3))
    
    dice = (2 * intersection + smooth) / (union + smooth)
    loss = 1 - dice.mean()
    return loss

def get_seg_loss(seg_loss_func, seg_logits,seg_labels, seg_loss_weight=1.0,dice_loss_weight=1.0):
    seg_loss = seg_loss_func(seg_logits,seg_labels)
    
    pred_probs = torch.softmax(seg_logits,dim=1)
    
    # Compute dice loss
    seg_dice_loss = dice_loss(pred_probs,seg_labels)
    # Calculate total loss from two functions
    seg_loss = (seg_loss_weight * seg_loss) + (dice_loss_weight * seg_dice_loss)
    
    return seg_loss

def train(
    exp_dir: str = "logs",
    model_name: str = "detector",
    num_epoch: int = 50,
    lr: float = 1e-4,
    batch_size: int = 32,
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
    train_data = load_data("drive_data/train", transform_pipeline='default', shuffle=True, batch_size=batch_size, num_workers=20)
    val_data = load_data("drive_data/val", transform_pipeline='default', shuffle=False, batch_size=batch_size, num_workers=20)

    # create loss functions and optimizer    
    seg_class_weights = torch.tensor([0.1,1.0,1.0]).to(device)
    seg_loss_func = torch.nn.CrossEntropyLoss()
    depth_loss_func = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(params=model.parameters(),lr=lr,weight_decay=1e-2)

    global_step = 0

    # create metric calculation objects
    train_detection_metric = DetectionMetric(num_classes=3)
    val_detection_metric = DetectionMetric(num_classes=3)
    

    
    # training loop
    for epoch in range(num_epoch):
        
        # clear metrics at beginning of epoch
        train_detection_metric.reset()
        val_detection_metric.reset()
        
        train_seg_loss_total = 0.0
        train_depth_loss_total = 0.0
        num_batches = 0
        
        for batch in train_data:
            
            images = batch['image'].to(device)      # (B, 3, H, W)
            depth_labels = batch['depth'].to(device)      # (B, H, W)
            seg_labels = batch['track'].to(device)        # (B, H, W) with values in {0,1,2}
            
            # Reset gradients
            optimizer.zero_grad()

            # Use model to make predictions
            seg_logits,depth_pred = model(images)
            
            # Compute loss
            seg_loss = get_seg_loss(seg_loss_func,seg_logits,seg_labels,0.03,0.97)
            depth_loss = depth_loss_func(depth_pred,depth_labels)

            logger.add_scalar('seg_train_loss',seg_loss,global_step)
            logger.add_scalar('depth_train_loss',depth_loss,global_step)

            # Caluculate total weighted loss
            total_loss = (0.97 * seg_loss) + (0.03 * depth_loss)
            logger.add_scalar('total_loss',total_loss,global_step)
            
            # Backward Pass
            total_loss.backward()
            optimizer.step()

            global_step += 1

            predicted_labels = torch.argmax(seg_logits,dim=1)
            train_detection_metric.add(predicted_labels,seg_labels, depth_pred,depth_labels)
            
            train_seg_loss_total += seg_loss.item()
            train_depth_loss_total += depth_loss.item()
            num_batches += 1
            
            # Loss calculations per epoch
            avg_seg_loss = train_seg_loss_total / num_batches
            avg_depth_loss = train_depth_loss_total / num_batches
            train_metrics = train_detection_metric.compute()

        with torch.inference_mode():
            
            val_seg_loss_total = 0.0
            val_depth_loss_total = 0.0
            val_batches = 0
            model.eval()

            for batch in val_data:
                
                images = batch['image'].to(device)       # (B, 3, H, W)
                depth_labels = batch['depth'].to(device) # (B, H, W) with values in the range [0,1]
                seg_labels = batch['track'].to(device)   # (B, H, W) with values in {0,1,2}
                
            
                seg_logits,depth_pred = model(images)
                
                seg_loss = get_seg_loss(seg_loss_func, seg_logits,seg_labels)
                
                depth_loss = depth_loss_func(depth_pred,depth_labels)

                val_seg_loss_total += seg_loss.item()
                val_depth_loss_total += depth_loss.item()
                val_batches += 1

                seg_preds = torch.argmax(seg_logits, dim=1)
                val_detection_metric.add(seg_preds, seg_labels, depth_pred, depth_labels)

        # Loss calculation per epoch
        avg_val_seg_loss = val_seg_loss_total / val_batches
        avg_val_depth_loss = val_depth_loss_total / val_batches
        val_metrics = val_detection_metric.compute()
        
        if val_metrics['iou'] >= 0.77:
            print("Acceptable IOU achieved, stopping early")
            break

        # compute accuracy for each epoch
        epoch_train_acc = train_detection_metric.compute()["accuracy"]
        epoch_val_acc = val_detection_metric.compute()["accuracy"]

        # log average train and val accuracy to tensorboard
        logger.add_scalar('train_accuracy',epoch_train_acc,epoch)
        logger.add_scalar('val_accuracy',epoch_val_acc,epoch)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}:\n"
                f"Train:    seg_loss = {avg_seg_loss:.4f}, depth_loss = {avg_depth_loss:.4f}, iou = {train_metrics['iou']:.4f}, accuracy = {train_metrics['accuracy']:.4f}, "
                f"  depth_error = {train_metrics['abs_depth_error']:.4f}, tp_depth_error = {train_metrics['tp_depth_error']:.4f}"
                f"\n"
                f"Val:      seg_loss = {avg_val_seg_loss:.4f}, depth_loss = {avg_val_depth_loss:.4f}, iou = {val_metrics['iou']:.4f}, accuracy = {val_metrics['accuracy']:.4f}, "
                f"  depth_error = {val_metrics['abs_depth_error']:.4f}, tp_depth_error = {val_metrics['tp_depth_error']:.4f}"
                f"\n"
            )

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


    # pass all arguments to train
    train(**vars(parser.parse_args()))
