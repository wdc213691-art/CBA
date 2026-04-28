import os
import numpy as np
import matplotlib.pyplot as plt

import utils_
from models.networks import *

import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from misc.metric_tool import ConfuseMatrixMeter
from models.losses import BCEDiceLoss, cross_entropy
from misc.logger_tool import Logger, Timer
from utils_ import de_norm


class CDTrainer():

    def __init__(self, args, dataloaders):
        self.dataloaders = dataloaders
        self.n_class = args.n_class
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)

        self.defer_trigger = getattr(args, 'defer_trigger')

        if self.defer_trigger:
            self.trigger_black = torch.nn.Parameter(torch.full((1, 3, 16, 16), -1.0))
            print("可学习触发器已启用")

        use_gpu = torch.cuda.is_available() and len(args.gpu_ids) > 0
        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if use_gpu else "cpu")
        print("使用设备：", self.device)

        if self.defer_trigger:
            self.trigger_black = torch.nn.Parameter(self.trigger_black.data.to(self.device))

        self.lr = args.lr
        train_params = list(self.net_G.parameters())
        if self.defer_trigger:
            train_params.append(self.trigger_black)

        if args.optimizer == "sgd":
            self.optimizer_G = optim.SGD(train_params, lr=self.lr,
                                         momentum=0.9, weight_decay=5e-4)
        elif args.optimizer == "adam":
            self.optimizer_G = optim.Adam(train_params, lr=self.lr, weight_decay=0)
        elif args.optimizer == "adamw":
            self.optimizer_G = optim.AdamW(train_params, lr=self.lr,
                                           betas=(0.9, 0.999), weight_decay=0.01)

        self.exp_lr_scheduler_G = get_scheduler(self.optimizer_G, args)

        self.running_metric = ConfuseMatrixMeter(n_class=2)
        log_path = os.path.join(args.checkpoint_dir, 'log.txt')
        self.logger = Logger(log_path)
        self.logger.write_dict_str(args.__dict__)
        self.timer = Timer()
        self.batch_size = args.batch_size

        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = args.max_epochs
        self.global_step = 0
        self.steps_per_epoch = len(dataloaders['train'])
        self.total_steps = self.max_num_epochs * self.steps_per_epoch

        self.batch = None
        self.G_pred = None
        self.G_loss = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
        self.vis_dir = args.vis_dir

        self._pxl_loss = BCEDiceLoss

        val_acc_path = os.path.join(self.checkpoint_dir, 'val_acc.npy')
        self.VAL_ACC = np.load(val_acc_path) if os.path.exists(val_acc_path) else np.array([], np.float32)

        train_acc_path = os.path.join(self.checkpoint_dir, 'train_acc.npy')
        self.TRAIN_ACC = np.load(train_acc_path) if os.path.exists(train_acc_path) else np.array([], np.float32)

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)


    def _load_checkpoint(self, ckpt_name='last_ckpt.pt'):
        ckpt_path = os.path.join(self.checkpoint_dir, ckpt_name)

        if not os.path.exists(ckpt_path):
            print('未找到断点，从头开始训练...')
            return

        self.logger.write('正在加载断点...\n')
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.net_G.load_state_dict(ckpt['model_G_state_dict'])
        self.optimizer_G.load_state_dict(ckpt['optimizer_G_state_dict'])
        self.exp_lr_scheduler_G.load_state_dict(ckpt['exp_lr_scheduler_G_state_dict'])
        self.net_G.to(self.device)

        if self.defer_trigger and 'trigger_black' in ckpt:
            self.trigger_black.data = ckpt['trigger_black'].data.to(self.device)
            print('可学习触发器已从断点恢复。')

        self.epoch_to_start = ckpt['epoch_id'] + 1
        self.best_val_acc   = ckpt['best_val_acc']
        self.best_epoch_id  = ckpt['best_epoch_id']
        self.total_steps = (self.max_num_epochs - self.epoch_to_start) * self.steps_per_epoch

        self.logger.write('续训起始 epoch=%d, 历史最佳精度=%.4f (epoch %d)\n' %
                          (self.epoch_to_start, self.best_val_acc, self.best_epoch_id))
        self.logger.write('\n')

    def _timer_update(self):
        self.global_step = (self.epoch_id - self.epoch_to_start) * self.steps_per_epoch + self.batch_id
        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        remaining_hours = self.timer.estimated_remaining()
        elapsed_seconds = self.timer.get_stage_elapsed()
        imgs_per_sec = (self.global_step + 1) * self.batch_size / elapsed_seconds
        return imgs_per_sec, remaining_hours

    def _visualize_pred(self, args):
        if args.deep_supervision:
            logits = self.G_pred[-1]
        else:
            logits = self.G_pred
        pred_class = torch.argmax(logits, dim=1, keepdim=True)
        pred_vis = pred_class * 255
        return pred_vis

    def _save_trigger_vis(self):
        if not self.defer_trigger:
            return
        trigger_arr = self.trigger_black.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        trigger_arr = (trigger_arr + 1.0) / 2.0
        trigger_arr = np.clip(trigger_arr, 0.0, 1.0)
        save_path = os.path.join(self.checkpoint_dir, 'trigger_black_latest.png')
        plt.imsave(save_path, trigger_arr)

    def _save_checkpoint(self, ckpt_name):
        save_dict = {
            'epoch_id':                       self.epoch_id,
            'best_val_acc':                   self.best_val_acc,
            'best_epoch_id':                  self.best_epoch_id,
            'model_G_state_dict':             self.net_G.state_dict(),
            'optimizer_G_state_dict':         self.optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict':  self.exp_lr_scheduler_G.state_dict(),
        }
        if self.defer_trigger:
            save_dict['trigger_black'] = self.trigger_black
            self._save_trigger_vis()
        save_path = os.path.join(self.checkpoint_dir, ckpt_name)
        torch.save(save_dict, save_path)

    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_G.step()

    def _update_metric(self, args):
        gt_label = self.batch['L'].to(self.device).detach()
        if args.deep_supervision:
            pred_logits = self.G_pred[-1].detach()
        else:
            pred_logits = self.G_pred.detach()
        pred_class = torch.argmax(pred_logits, dim=1)
        score = self.running_metric.update_cm(
            pr=pred_class.cpu().numpy(),
            gt=gt_label.cpu().numpy()
        )
        return score

    def _collect_running_batch_states(self, args):
        running_acc = self._update_metric(args)
        total_batches = len(self.dataloaders['train'] if self.is_training else self.dataloaders['val'])
        imgs_per_sec, remaining_hours = self._timer_update()

        if self.batch_id % 100 == 1:
            log_msg = ('[%s] epoch %d/%d  batch %d/%d  '
                       '速度: %.1f img/s  剩余: %.2fh  '
                       '损失: %.5f  F1: %.5f\n') % (
                '训练' if self.is_training else '验证',
                self.epoch_id, self.max_num_epochs - 1,
                self.batch_id, total_batches,
                imgs_per_sec, remaining_hours,
                self.G_loss.item(), running_acc
            )
            self.logger.write(log_msg)

    def _collect_epoch_states(self):
        scores = self.running_metric.get_scores()
        self.epoch_acc = scores['mf1']
        self.logger.write('[%s] Epoch %d/%d  mF1=%.5f\n' %
                          ('训练' if self.is_training else '验证',
                           self.epoch_id, self.max_num_epochs - 1, self.epoch_acc))
        detail = '  '.join('%s: %.5f' % (k, v) for k, v in scores.items())
        self.logger.write(detail + '\n\n')

    def _update_checkpoints(self):
        self._save_checkpoint('last_ckpt.pt')
        self.logger.write('最新模型已更新。当前精度=%.4f  历史最佳=%.4f (epoch %d)\n'
                          % (self.epoch_acc, self.best_val_acc, self.best_epoch_id))
        self.logger.write('\n')

        if self.epoch_acc > self.best_val_acc:
            self.best_val_acc  = self.epoch_acc
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint('best_ckpt.pt')
            self.logger.write('★ 最佳模型已更新！\n\n')

    def _update_training_acc_curve(self):
        self.TRAIN_ACC = np.append(self.TRAIN_ACC, self.epoch_acc)
        np.save(os.path.join(self.checkpoint_dir, 'train_acc.npy'), self.TRAIN_ACC)

    def _update_val_acc_curve(self):
        self.VAL_ACC = np.append(self.VAL_ACC, self.epoch_acc)
        np.save(os.path.join(self.checkpoint_dir, 'val_acc.npy'), self.VAL_ACC)

    def _clear_cache(self):
        self.running_metric.clear()

    def _apply_learnable_trigger(self, x, poison_type, opacity, apply_to, current_img_type):
        if apply_to not in ('both', current_img_type):
            return x

        B, C, H, W = x.shape
        s = 16
        top  = H - s
        left = W - s

        poison_mask = (poison_type == 2).view(B, 1, 1, 1).float()
        alpha = opacity.view(B, 1, 1, 1)

        x_out = x.clone()
        roi = x_out[:, :, top:top+s, left:left+s]

        blended = (1.0 - alpha * poison_mask) * roi + alpha * poison_mask * self.trigger_black
        x_out[:, :, top:top+s, left:left+s] = blended
        return x_out

    def _forward_pass(self, args, batch):
        self.batch = batch
        img_A = batch['A'].to(self.device)
        img_B = batch['B'].to(self.device)

        if self.defer_trigger and self.is_training:
            poison_type = batch.get('poison_type')
            opacity     = batch.get('opacity')
            if poison_type is not None:
                poison_type = poison_type.to(self.device)
                opacity     = opacity.to(self.device)
                apply_to = getattr(args, 'apply_to', 'A')
                img_A = self._apply_learnable_trigger(img_A, poison_type, opacity, apply_to, 'A')

        if args.loss_SD:
            self.G_pred0, self.G_pred1, self.G_pred2, self.G_pred3, self.G_pred4 = self.net_G(img_A, img_B)
            self.G_pred = self.G_pred0
        else:
            self.G_pred = self.net_G(img_A, img_B)

    def _backward_G(self, args):
        gt = self.batch['L'].to(self.device).long()

        self.G_loss = 0
        if args.loss_SD:
            self.G_loss = self._pxl_loss(self.G_pred, gt)
            gt_float = gt.float()
            for scale, aux_pred in zip([1/2, 1/4, 1/8, 1/16],
                                       [self.G_pred1, self.G_pred2, self.G_pred3, self.G_pred4]):
                gt_scaled = F.interpolate(gt_float, scale_factor=scale, mode='bilinear')
                self.G_loss = self.G_loss + self._pxl_loss(aux_pred, gt_scaled)
        else:
            if args.deep_supervision:
                for aux_pred in self.G_pred:
                    self.G_loss = self.G_loss + self._pxl_loss(aux_pred, gt)
                self.G_loss = self.G_loss / len(self.G_pred)
            else:
                self.G_loss = self._pxl_loss(self.G_pred, gt)

        self.G_loss.backward()

    def train_models(self, args):
        self._load_checkpoint()

        for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):

            self._clear_cache()
            self.is_training = True
            self.net_G.train()

            num_batches = len(self.dataloaders['train'])
            current_lr = self.optimizer_G.param_groups[0]['lr']
            self.logger.write('epoch %d  lr=%.7f\n' % (self.epoch_id, current_lr))

            for self.batch_id, batch in tqdm(enumerate(self.dataloaders['train']), total=num_batches):
                self._forward_pass(args, batch)
                self.optimizer_G.zero_grad()
                self._backward_G(args)
                self.optimizer_G.step()

                if self.defer_trigger:
                    with torch.no_grad():
                        self.trigger_black.data.clamp_(-1.0, 1.0)

                self._collect_running_batch_states(args)

            self._collect_epoch_states()
            self._update_training_acc_curve()
            self._update_lr_schedulers()

            self.logger.write('\n开始验证...\n')
            self._clear_cache()
            self.is_training = False
            self.net_G.eval()

            for self.batch_id, batch in enumerate(self.dataloaders['val']):
                with torch.no_grad():
                    self._forward_pass(args, batch)
                self._collect_running_batch_states(args)

            self._collect_epoch_states()
            self._update_val_acc_curve()
            self._update_checkpoints()

