import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    confusion_matrix, precision_recall_curve, precision_score
)
from sklearn.preprocessing import StandardScaler
from prettytable import PrettyTable

from models import binary_cross_entropy, cross_entropy_logits, entropy_logits, RandomLayer
from domain_adaptator import ReverseLayerF


# =========================
# 辅助：batch AUC margin
# =========================
def batch_auc_margin_loss(logits, labels, margin=0.15):
    """
    logits: [B, 2] —— 分类器的原始logits（未softmax）
    labels: [B]     —— 0/1
    约束：正例平均分 > 负例平均分 + margin
    """
    with torch.no_grad():
        has_pos = (labels == 1).any().item()
        has_neg = (labels == 0).any().item()
    if not (has_pos and has_neg):
        return logits.new_zeros(1).squeeze(0)

    s = F.softmax(logits, dim=1)[:, 1]   # 正类概率
    pos = s[labels == 1]
    neg = s[labels == 0]
    gap = pos.mean() - neg.mean()
    return F.relu(margin - gap)


# =========================
# 温度标定
# =========================
class TemperatureScaler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.log_t = torch.nn.Parameter(torch.zeros(1))  # T = exp(log_t)

    def forward(self, logits):
        return logits / torch.exp(self.log_t)

    def fit(self, logits_val, labels_val, max_iter=200, lr=0.05):
        self.train()
        opt = torch.optim.LBFGS([self.log_t], lr=lr, max_iter=max_iter)
        nll = torch.nn.CrossEntropyLoss()

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = nll(self.forward(logits_val), labels_val.long())
            loss.backward()
            return loss

        opt.step(closure)
        self.eval()


# =========================
# Trainer
# =========================
class Trainer(object):
    def __init__(self, model, optim, device, train_dataloader, val_dataloader, test_dataloader,
                 opt_da=None, discriminator=None, experiment=None, alpha=1, **config):
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.current_epoch = 0

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.is_da = config["DA"]["USE"]
        self.alpha = alpha  # ⭐ 使用传入的alpha参数（默认1），与inductive_mode_32一致
        self.n_class = config["DECODER"]["BINARY"]

        # DA 组件
        if opt_da:
            self.optim_da = opt_da
        if self.is_da:
            self.da_method = config["DA"]["METHOD"]
            self.domain_dmm = discriminator
            if config["DA"]["RANDOM_LAYER"] and not config["DA"]["ORIGINAL_RANDOM"]:
                self.random_layer = nn.Linear(
                    in_features=config["DECODER"]["IN_DIM"] * self.n_class,
                    out_features=config["DA"]["RANDOM_DIM"],
                    bias=False
                ).to(self.device)
                torch.nn.init.normal_(self.random_layer.weight, mean=0, std=1)
                for param in self.random_layer.parameters():
                    param.requires_grad = False
            elif config["DA"]["RANDOM_LAYER"] and config["DA"]["ORIGINAL_RANDOM"]:
                self.random_layer = RandomLayer(
                    [config["DECODER"]["IN_DIM"], self.n_class],
                    config["DA"]["RANDOM_DIM"]
                )
                if torch.cuda.is_available():
                    self.random_layer.cuda()
            else:
                self.random_layer = False

        self.da_init_epoch = config["DA"]["INIT_EPOCH"]
        self.init_lamb_da = config["DA"]["LAMB_DA"]
        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.use_da_entropy = config["DA"]["USE_ENTROPY"]

        self.nb_training = len(self.train_dataloader)
        self.step = 0
        self.experiment = experiment

        self.best_model = None
        self.best_epoch = None
        self.best_auroc = 0.0
        self.best_sel = -1e9  # 1.0*AUROC + 0.*AUPRC

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.train_da_loss_epoch = []
        self.val_loss_epoch, self.val_auroc_epoch = [], []
        self.test_metrics = {}
        self.config = config
        self.output_dir = config["RESULT"]["OUTPUT_DIR"]

        valid_metric_header = ["# Epoch", "AUROC", "AUPRC", "Val_loss"]
        test_metric_header = ["# Best Epoch", "AUROC", "AUPRC", "F1", "Sensitivity",
                              "Specificity", "Accuracy", "Threshold", "Test_loss"]
        train_metric_header = (["# Epoch", "Train_loss"] if not self.is_da
                               else ["# Epoch", "Train_loss", "Model_loss", "epoch_lamb_da", "da_loss"])

        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)

        self.original_random = config["DA"]["ORIGINAL_RANDOM"]
        self.scaler = StandardScaler()
        self.temp_scaler = None

        # ⭐ 新增：检测当前使用的交互层类型
        self.inter_type = config.get("BCN", {}).get("TYPE", "ban").lower()
        self.use_simple_loss = (self.inter_type == "ban")  # BAN使用简单损失，CrossMamba使用复杂损失

    # 旧的 λ 调度函数
    def da_lambda_decay(self):
        delta_epoch = self.current_epoch - self.da_init_epoch
        non_init_epoch = self.epochs - self.da_init_epoch
        p = (self.current_epoch + delta_epoch * self.nb_training) / (non_init_epoch * self.nb_training)
        grow_fact = 2.0 / (1.0 + np.exp(-10 * p)) - 1
        return self.init_lamb_da * grow_fact

    def train(self):
        float2str = lambda x: '%0.4f' % x
        # Calculate remaining epochs if resuming from checkpoint
        start_epoch = self.current_epoch
        for i in range(start_epoch, self.epochs):
            self.current_epoch = i + 1

            if not self.is_da:
                train_loss = self.train_epoch()
                train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss]))
                if self.experiment:
                    self.experiment.log_metric("train_epoch model loss", train_loss, epoch=self.current_epoch)
            else:
                train_loss, model_loss, da_loss, epoch_lamb = self.train_da_epoch()
                train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [
                    train_loss, model_loss, epoch_lamb, da_loss
                ]))
                self.train_model_loss_epoch.append(model_loss)
                self.train_da_loss_epoch.append(da_loss)
                if self.experiment:
                    self.experiment.log_metric("train_epoch total loss", train_loss, epoch=self.current_epoch)
                    self.experiment.log_metric("train_epoch model loss", model_loss, epoch=self.current_epoch)
                    if self.current_epoch >= self.da_init_epoch:
                        self.experiment.log_metric("train_epoch da loss", da_loss, epoch=self.current_epoch)

            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)

            # 验证
            auroc, auprc, val_loss = self.test(dataloader="val")
            if self.experiment:
                self.experiment.log_metric("valid_epoch model loss", val_loss, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch auroc", auroc, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch auprc", auprc, epoch=self.current_epoch)

            val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [auroc, auprc, val_loss]))
            self.val_table.add_row(val_lst)
            self.val_loss_epoch.append(val_loss)
            self.val_auroc_epoch.append(auroc)

            # 选最优：0.5*(AUROC + AUPRC)
            sel = auroc
            if sel >= self.best_sel:
                self.best_sel = sel
                self.best_model = copy.deepcopy(self.model)
                self.best_auroc = auroc
                self.best_epoch = self.current_epoch

            print(f'Validation at Epoch {self.current_epoch} with validation loss {val_loss} '
                  f'AUROC {auroc} AUPRC {auprc}')

            # Save checkpoint after each epoch
            self.save_checkpoint(self.current_epoch)

        # ---------- 温度标定 + 验证集选阈值 ----------
        self._fit_temperature_on_val()
        thr, f1_val = self._select_threshold_on_val(calibrate=True)
        print(f"[VAL] fixed threshold={thr:.4f}, F1={f1_val:.4f}")

        # ---------- 用 best_model 在测试集评估 ----------
        # auroc, auprc, f1, sensitivity, specificity, accuracy, test_loss, thred_optim, precision, cm1, y_pred = \
        #     self.test(dataloader="test", calibrate=True, fixed_threshold=thr)
            

        # test_lst = ["epoch " + str(self.best_epoch)] + list(map(float2str, [
        #     auroc, auprc, f1, sensitivity, specificity, accuracy, thred_optim, test_loss
        # ]))
        # self.test_table.add_row(test_lst)

        # print(f'Test at Best Model of Epoch {self.best_epoch} with test loss {test_loss} '
        #       f'AUROC {auroc} AUPRC {auprc} f1-score {f1} Specificity {specificity} '
        #       f'Accuracy {accuracy} Thred_optim {thred_optim}')


        # ---------- 用 best_model 在测试集评估 ----------
        auroc, auprc, f1, sensitivity, specificity, accuracy, test_loss, thred_optim, precision, cm1, y_pred = \
            self.test(dataloader="test", calibrate=True, fixed_threshold=thr)

        # 取测试集的 (y_true, y_scores)，与上面同口径：best_model + calibrate=True
        y_true, y_scores, _ = self._scores_and_labels(
            self.test_dataloader, use_best=True, calibrate=True
        )

        # 复刻 baseline 的“ROC 假精度 + 测试集内选阈值 + 跳前5个阈值”
        f1_base, thr_base, cm_base = self._baseline_like_f1(
            y_true, y_scores, ignore_first_k=5, assume_pi_half=True
        )
        # 同样的 ROC-F1，但用真实先验 π，量化 baseline 的乐观幅度
        f1_base_realpi, thr_base_realpi, _ = self._baseline_like_f1(
            y_true, y_scores, ignore_first_k=5, assume_pi_half=False
        )

        test_lst = ["epoch " + str(self.best_epoch)] + list(map(float2str, [
            auroc, auprc, f1, sensitivity, specificity, accuracy, thred_optim, test_loss
        ]))
        self.test_table.add_row(test_lst)

        # 打印两套口径
        print(
            f'Test at Best Model of Epoch {self.best_epoch} with test loss {test_loss} '
            f'AUROC {auroc} AUPRC {auprc} f1-score {f1} Sensitivity {sensitivity} '
            f'Specificity {specificity} Accuracy {accuracy} Thred_optim {thred_optim}'
        )
        print(
            f"[METRIC] proper_F1={f1:.4f} @thr={thred_optim:.4f} | "
            f"baseline_F1={f1_base:.4f} @thr={thr_base:.4f} "
            f"(real-π ROC-F1={f1_base_realpi:.4f})"
)


        self.test_metrics["auroc"] = auroc
        self.test_metrics["auprc"] = auprc
        self.test_metrics["test_loss"] = test_loss
        self.test_metrics["sensitivity"] = sensitivity
        self.test_metrics["specificity"] = specificity
        self.test_metrics["accuracy"] = accuracy
        self.test_metrics["thred_optim"] = thred_optim
        self.test_metrics["best_epoch"] = self.best_epoch
        self.test_metrics["F1"] = f1
        self.test_metrics["Precision"] = precision

        self.save_result()

        if self.experiment:
            self.experiment.log_metric("valid_best_auroc", self.best_auroc)
            self.experiment.log_metric("valid_best_epoch", self.best_epoch)
            self.experiment.log_metric("test_auroc", auroc)
            self.experiment.log_metric("test_auprc", auprc)
            self.experiment.log_metric("test_sensitivity", sensitivity)
            self.experiment.log_metric("test_specificity", specificity)
            self.experiment.log_metric("test_accuracy", accuracy)
            self.experiment.log_metric("test_threshold", thred_optim)
            self.experiment.log_metric("test_f1", f1)
            self.experiment.log_metric("test_precision", precision)

        return self.test_metrics

    def save_checkpoint(self, epoch):
        """Save checkpoint at the end of each epoch"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'best_model_state_dict': self.best_model.state_dict() if self.best_model is not None else None,
            'best_epoch': self.best_epoch,
            'best_auroc': self.best_auroc,
            'train_loss_epoch': self.train_loss_epoch,
            'train_model_loss_epoch': self.train_model_loss_epoch,
            'train_da_loss_epoch': self.train_da_loss_epoch,
            'val_loss_epoch': self.val_loss_epoch,
            'val_auroc_epoch': self.val_auroc_epoch,
            'config': self.config,
            'train_table_str': self.train_table.get_string(),
            'val_table_str': self.val_table.get_string(),
        }
        if self.is_da and hasattr(self, 'optim_da'):
            checkpoint['optimizer_da_state_dict'] = self.optim_da.state_dict()

        checkpoint_path = os.path.join(self.output_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint saved at epoch {epoch} to {checkpoint_path}')

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint to resume training"""
        if not os.path.exists(checkpoint_path):
            print(f'No checkpoint found at {checkpoint_path}, starting from scratch')
            return False

        print(f'Loading checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.current_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])

        if checkpoint['best_model_state_dict'] is not None:
            self.best_model = copy.deepcopy(self.model)
            self.best_model.load_state_dict(checkpoint['best_model_state_dict'])

        self.best_epoch = checkpoint['best_epoch']
        self.best_auroc = checkpoint['best_auroc']
        self.train_loss_epoch = checkpoint['train_loss_epoch']
        self.train_model_loss_epoch = checkpoint['train_model_loss_epoch']
        self.train_da_loss_epoch = checkpoint['train_da_loss_epoch']
        self.val_loss_epoch = checkpoint['val_loss_epoch']
        self.val_auroc_epoch = checkpoint['val_auroc_epoch']

        # Reconstruct PrettyTable objects from saved strings
        if 'train_table_str' in checkpoint and 'val_table_str' in checkpoint:
            # For now, just print a note that we loaded from strings
            print("Note: PrettyTable data loaded from checkpoint")
        elif 'train_table' in checkpoint and 'val_table' in checkpoint:
            # Old format compatibility
            self.train_table = checkpoint['train_table']
            self.val_table = checkpoint['val_table']

        if self.is_da and 'optimizer_da_state_dict' in checkpoint and hasattr(self, 'optim_da'):
            self.optim_da.load_state_dict(checkpoint['optimizer_da_state_dict'])

        print(f'Resumed from epoch {self.current_epoch}, best epoch: {self.best_epoch}, best AUROC: {self.best_auroc:.4f}')
        return True

    def save_result(self):
        if self.config["RESULT"]["SAVE_MODEL"]:
            torch.save(self.best_model.state_dict(),
                       os.path.join(self.output_dir, f"best_model_epoch_{self.best_epoch}.pth"))
            torch.save(self.model.state_dict(),
                       os.path.join(self.output_dir, f"model_epoch_{self.current_epoch}.pth"))

        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config
        }
        if self.is_da:
            state["train_model_loss"] = self.train_model_loss_epoch
            state["train_da_loss"] = self.train_da_loss_epoch
            state["da_init_epoch"] = self.da_init_epoch

        os.makedirs(self.output_dir, exist_ok=True)
        torch.save(state, os.path.join(self.output_dir, f"result_metrics.pt"))

        with open(os.path.join(self.output_dir, "valid_markdowntable.txt"), 'w') as fp:
            fp.write(self.val_table.get_string())
        with open(os.path.join(self.output_dir, "test_markdowntable.txt"), 'w') as fp:
            fp.write(self.test_table.get_string())
        with open(os.path.join(self.output_dir, "train_markdowntable.txt"), "w") as fp:
            fp.write(self.train_table.get_string())

    # =========================
    # 熵权重（CDAN-E）
    # =========================
    def _compute_entropy_weights(self, logits):
        p = torch.softmax(logits, dim=1)
        ent = -torch.sum(p * torch.log(p + 1e-12), dim=1)  # [B]
        w = 1.0 + torch.exp(-ent)
        return (w / (w.mean() + 1e-12)).detach()

    # =========================
    # 单域训练（不启用DA）
    # =========================
    def train_epoch(self):
        self.model.train()
        loss_epoch = 0.0
        num_batches = len(self.train_dataloader)

        for i, (v_d, v_d_3d, sm, v_p, esm, labels, teacher, drug_3d_data) in enumerate(tqdm(self.train_dataloader)):  # ⭐ 增加v_d_3d和drug_3d_data
            self.step += 1

            sm = torch.tensor(sm, dtype=torch.float32).reshape(sm.shape[0], 1, 384)
            esm = torch.tensor(esm, dtype=torch.float32).reshape(esm.shape[0], 1, 1280)
            teacher = torch.tensor(teacher, dtype=torch.float32)

            v_d, v_d_3d, sm, v_p, esm, labels, teacher = (  # ⭐ 增加v_d_3d
                v_d.to(self.device), v_d_3d.to(self.device), sm.to(self.device), v_p.to(self.device),
                esm.to(self.device), labels.float().to(self.device), teacher.to(self.device)
            )

            self.optim.zero_grad(set_to_none=True)
            _, _, f, score = self.model(v_d, v_d_3d, sm, v_p, esm, self.device)  # ⭐ 增加v_d_3d参数

            if self.n_class == 1:
                _, loss = binary_cross_entropy(score, labels)
            else:
                _, loss = cross_entropy_logits(score, labels)

            z = F.mse_loss(f, teacher)
            loss = loss + 1.0 * z

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 梯度裁剪
            self.optim.step()

            loss_epoch += float(loss.item())
            if self.experiment:
                self.experiment.log_metric("train_step model loss", float(loss.item()), step=self.step)

        loss_epoch /= max(1, num_batches)
        print(f'Training at Epoch {self.current_epoch} with training loss {loss_epoch}')
        return loss_epoch

    # =========================
    # 简单DA训练（BAN配置专用，与inductive_mode_32一致）
    # =========================
    def _train_da_epoch_simple(self):
        """
        简单的DA训练策略，完全匹配inductive_mode_32的逻辑
        用于BAN配置，避免Step-D/Step-G分离带来的训练不稳定
        """
        self.model.train()
        total_loss_epoch = 0
        model_loss_epoch = 0
        da_loss_epoch = 0
        epoch_lamb_da = 0

        if self.current_epoch >= self.da_init_epoch:
            epoch_lamb_da = self.da_lambda_decay()
            if self.experiment:
                self.experiment.log_metric("DA loss lambda", epoch_lamb_da, epoch=self.current_epoch)

        num_batches = len(self.train_dataloader)

        for i, (batch_s, batch_t) in enumerate(tqdm(self.train_dataloader)):
            self.step += 1

            # 解包数据（与inductive_mode_32完全一致）
            v_d, v_d_3d, sm, v_p, esm, labels, teacher = batch_s[0].to(self.device), batch_s[1].to(self.device), batch_s[2].to(self.device), batch_s[3].to(self.device), batch_s[4].to(self.device), batch_s[5].float().to(self.device), batch_s[6].to(self.device)
            v_d_t, v_d_3d_t, smt, v_p_t, esmt, labelst = batch_t[0].to(self.device), batch_t[1].to(self.device), batch_t[2].to(self.device), batch_t[3].to(self.device), batch_t[4].to(self.device), batch_t[5].float().to(self.device)

            # 转换和reshape
            teacher = torch.tensor(teacher, dtype=torch.float32)
            sm = torch.tensor(sm, dtype=torch.float32)
            sm = torch.reshape(sm, (sm.shape[0], 1, 384))

            smt = torch.tensor(smt, dtype=torch.float32)
            smt = torch.reshape(smt, (smt.shape[0], 1, 384))

            esm = torch.tensor(esm, dtype=torch.float32)
            esm = torch.reshape(esm, (esm.shape[0], 1, 1280))

            esmt = torch.tensor(esmt, dtype=torch.float32)
            esmt = torch.reshape(esmt, (esmt.shape[0], 1, 1280))

            self.optim.zero_grad()
            self.optim_da.zero_grad()

            device = self.device
            v_d, v_p, f, score = self.model(v_d, v_d_3d, sm, v_p, esm, device)

            # 计算模型损失（简单版本）
            if self.n_class == 1:
                n, model_loss = binary_cross_entropy(score, labels)
            else:
                n, model_loss = cross_entropy_logits(score, labels)

            z = F.mse_loss(f, teacher)
            model_loss = model_loss + 1.0 * z

            # 如果达到DA epoch，计算DA loss
            if self.current_epoch >= self.da_init_epoch:
                v_d_t, v_p_t, f_t, t_score = self.model(v_d_t, v_d_3d_t, smt, v_p_t, esmt, device)

                if self.da_method == "CDAN":
                    reverse_f = ReverseLayerF.apply(f, self.alpha)
                    softmax_output = torch.nn.Softmax(dim=1)(score)
                    softmax_output = softmax_output.detach()

                    if self.original_random:
                        random_out = self.random_layer.forward([reverse_f, softmax_output])
                        adv_output_src_score = self.domain_dmm(random_out.view(-1, random_out.size(1)))
                    else:
                        feature = torch.bmm(softmax_output.unsqueeze(2), reverse_f.unsqueeze(1))
                        feature = feature.view(-1, softmax_output.size(1) * reverse_f.size(1))
                        if self.random_layer:
                            random_out = self.random_layer.forward(feature)
                            adv_output_src_score = self.domain_dmm(random_out)
                        else:
                            adv_output_src_score = self.domain_dmm(feature)

                    reverse_f_t = ReverseLayerF.apply(f_t, self.alpha)
                    softmax_output_t = torch.nn.Softmax(dim=1)(t_score)
                    softmax_output_t = softmax_output_t.detach()

                    if self.original_random:
                        random_out_t = self.random_layer.forward([reverse_f_t, softmax_output_t])
                        adv_output_tgt_score = self.domain_dmm(random_out_t.view(-1, random_out_t.size(1)))
                    else:
                        feature_t = torch.bmm(softmax_output_t.unsqueeze(2), reverse_f_t.unsqueeze(1))
                        feature_t = feature_t.view(-1, softmax_output_t.size(1) * reverse_f_t.size(1))
                        if self.random_layer:
                            random_out_t = self.random_layer.forward(feature_t)
                            adv_output_tgt_score = self.domain_dmm(random_out_t)
                        else:
                            adv_output_tgt_score = self.domain_dmm(feature_t)

                    if self.use_da_entropy:
                        entropy_src = self._compute_entropy_weights(score)
                        entropy_tgt = self._compute_entropy_weights(t_score)
                        src_weight = entropy_src / torch.sum(entropy_src)
                        tgt_weight = entropy_tgt / torch.sum(entropy_tgt)
                    else:
                        src_weight = None
                        tgt_weight = None

                    # 使用当前batch的实际大小
                    current_batch_size = v_d.size(0)
                    n_src, loss_cdan_src = cross_entropy_logits(
                        adv_output_src_score,
                        torch.zeros(current_batch_size).to(self.device),
                        src_weight
                    )
                    n_tgt, loss_cdan_tgt = cross_entropy_logits(
                        adv_output_tgt_score,
                        torch.ones(current_batch_size).to(self.device),
                        tgt_weight
                    )
                    da_loss = loss_cdan_src + loss_cdan_tgt
                else:
                    raise ValueError(f"The da method {self.da_method} is not supported")

                loss = model_loss + da_loss  # ⭐ 不使用epoch_lamb_da权重（与inductive_mode_32一致）
            else:
                loss = model_loss

            loss.backward()
            # ⭐ 使用梯度裁剪（与inductive_mode_32一致）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if self.is_da and self.current_epoch >= self.da_init_epoch:
                torch.nn.utils.clip_grad_norm_(self.domain_dmm.parameters(), max_norm=1.0)
            self.optim.step()
            self.optim_da.step()

            total_loss_epoch += loss.item()
            model_loss_epoch += model_loss.item()

            if self.experiment:
                self.experiment.log_metric("train_step model loss", model_loss.item(), step=self.step)
                self.experiment.log_metric("train_step total loss", loss.item(), step=self.step)

            if self.current_epoch >= self.da_init_epoch:
                da_loss_epoch += da_loss.item()
                if self.experiment:
                    self.experiment.log_metric("train_step da loss", da_loss.item(), step=self.step)

        total_loss_epoch = total_loss_epoch / num_batches
        model_loss_epoch = model_loss_epoch / num_batches
        da_loss_epoch = da_loss_epoch / num_batches

        if self.current_epoch < self.da_init_epoch:
            print('Training at Epoch ' + str(self.current_epoch) + ' with model training loss ' + str(total_loss_epoch))
        else:
            print('Training at Epoch ' + str(self.current_epoch) + ' model training loss ' + str(model_loss_epoch)
                  + ", da loss " + str(da_loss_epoch) + ", total training loss " + str(total_loss_epoch) + ", DA lambda " +
                  str(epoch_lamb_da))

        return total_loss_epoch, model_loss_epoch, da_loss_epoch, epoch_lamb_da

    # =========================
    # DA 训练（CDAN / CDAN-E）
    # =========================
    def train_da_epoch(self):
        # ⭐ BAN配置使用简单的DA训练策略（与inductive_mode_32一致）
        if self.use_simple_loss:
            return self._train_da_epoch_simple()

        # CrossMamba配置使用复杂的Step-D/Step-G策略
        self.model.train()
        total_loss_epoch = 0.0
        model_loss_epoch = 0.0
        da_loss_epoch = 0.0
        epoch_lamb_da_last = 0.0

        num_batches = len(self.train_dataloader)
        lambda_max = float(self.config["DA"].get("LAMB_MAX", 0.05))  # 可在 YAML 里设置

        # 封装：DGLGraph.local_var() 兼容
        def _local_var_safe(x):
            return x.local_var() if hasattr(x, "local_var") else x

        for i, (batch_s, batch_t) in enumerate(tqdm(self.train_dataloader)):
            # λ 爬坡并限幅
            if self.current_epoch >= self.da_init_epoch:
                progress = (self.current_epoch - self.da_init_epoch) + (i + 1) / max(1, num_batches)
                total = max(1, self.epochs - self.da_init_epoch)
                p = float(np.clip(progress / total, 0.0, 1.0))
                epoch_lamb_da = lambda_max * (2.0 / (1.0 + np.exp(-5 * p)) - 1.0)
            else:
                epoch_lamb_da = 0.0
            epoch_lamb_da_last = epoch_lamb_da

            # 取数据并整形到 device
            v_d, v_d_3d, sm, v_p, esm, labels, teacher, drug_3d_data = batch_s  # ⭐ 增加v_d_3d和drug_3d_data
            v_d_t, v_d_3d_t, smt, v_p_t, esmt, _, drug_3d_data_t = batch_t  # ⭐ 增加v_d_3d_t和drug_3d_data_t

            sm   = torch.as_tensor(sm,  dtype=torch.float32, device=self.device).reshape(sm.shape[0],  1, 384)
            esm  = torch.as_tensor(esm, dtype=torch.float32, device=self.device).reshape(esm.shape[0], 1, 1280)
            smt  = torch.as_tensor(smt, dtype=torch.float32, device=self.device).reshape(smt.shape[0],  1, 384)
            esmt = torch.as_tensor(esmt,dtype=torch.float32, device=self.device).reshape(esmt.shape[0], 1, 1280)

            v_d, v_d_3d, v_p = v_d.to(self.device), v_d_3d.to(self.device), v_p.to(self.device)  # ⭐ 增加v_d_3d
            v_d_t, v_d_3d_t, v_p_t = v_d_t.to(self.device), v_d_3d_t.to(self.device), v_p_t.to(self.device)  # ⭐ 增加v_d_3d_t
            labels = labels.float().to(self.device)
            teacher = teacher.float().to(self.device)

            # ========= 源域监督（始终进行）=========
            self.optim.zero_grad(set_to_none=True)
            _, _, f_s, s_logit = self.model(v_d, v_d_3d, sm, v_p, esm, self.device)  # ⭐ 增加v_d_3d参数

            # ⭐ 根据交互层类型选择损失函数策略
            if self.use_simple_loss:
                # ========= BAN配置：使用简单损失（与inductive_mode_32一致）=========
                if self.n_class == 1:
                    _, cls_loss = binary_cross_entropy(s_logit, labels)
                else:
                    _, cls_loss = cross_entropy_logits(s_logit, labels)

                # 辅助损失：固定权重1.0的MSE
                aux_loss = 1.0 * F.mse_loss(f_s, teacher)
            else:
                # ========= CrossMamba配置：使用复杂损失（原有策略）=========
                if self.n_class == 1:
                    pi_b = labels.mean().clamp(1e-3, 1-1e-3)
                    w = torch.where(labels > 0.5, 0.5 / pi_b, 0.5 / (1.0 - pi_b))  # 正例更稀缺 ⇒ 权重大
                    _, cls_loss = cross_entropy_logits(s_logit, labels, w)  # 第三个参数为逐样本权重
                else:
                    _, cls_loss = cross_entropy_logits(s_logit, labels, margin=0.0)

                # 辅助损失（DA 前后不同权重）
                if self.current_epoch < self.da_init_epoch:
                    w_mse = 1.00
                else:
                    w_mse = 0.50
                aux_loss = w_mse * F.mse_loss(f_s, teacher)

            # 若未到 INIT_EPOCH：不做 Step-D / Step-G
            if self.current_epoch < self.da_init_epoch:
                loss = cls_loss + aux_loss
                loss.backward()
                # ⭐ BAN配置不使用梯度裁剪，CrossMamba使用梯度裁剪
                if not self.use_simple_loss:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim.step()

                self.step += 1
                total_loss_epoch += float(loss.item())
                model_loss_epoch += float((cls_loss + aux_loss).item())
                # da_loss_epoch 保持 0
                if self.experiment:
                    self.experiment.log_metric("train_step model loss", float((cls_loss + aux_loss).item()), step=self.step)
                    self.experiment.log_metric("train_step total loss", float(loss.item()), step=self.step)
                continue

            # ========= Step-D：只训练域判别器（INIT_EPOCH 之后）=========
            self.optim_da.zero_grad(set_to_none=True)
            with torch.no_grad():
                _, _, f_s_D, s_logit_D = self.model(_local_var_safe(v_d),   _local_var_safe(v_d_3d),   sm,  v_p,  esm,  self.device)  # ⭐ 增加v_d_3d
                _, _, f_t_D, t_logit_D = self.model(_local_var_safe(v_d_t), _local_var_safe(v_d_3d_t), smt, v_p_t, esmt, self.device)  # ⭐ 增加v_d_3d_t

            ps_det = torch.softmax(s_logit_D, dim=1).detach()
            pt_det = torch.softmax(t_logit_D, dim=1).detach()

            def _cond_feat(f, p):
                if self.original_random:
                    return self.random_layer.forward([f, p]).view(-1, self.random_layer.output_dim)
                else:
                    cf = torch.bmm(p.unsqueeze(2), f.unsqueeze(1)).view(-1, p.size(1) * f.size(1))
                    return self.random_layer.forward(cf) if self.random_layer else cf

            feat_s_D = _cond_feat(f_s_D.detach(), ps_det)
            feat_t_D = _cond_feat(f_t_D.detach(), pt_det)

            self.domain_dmm.train()
            dom_s = self.domain_dmm(feat_s_D)
            dom_t = self.domain_dmm(feat_t_D)

            # 对称权重
            bs_s, bs_t = dom_s.size(0), dom_t.size(0)
            y_s = torch.zeros(bs_s, device=self.device)
            y_t = torch.ones(bs_t,  device=self.device)
            _, loss_Ds = cross_entropy_logits(dom_s, y_s)
            _, loss_Dt = cross_entropy_logits(dom_t, y_t)

            w_s = bs_t / (bs_s + 1e-8)
            w_t = bs_s / (bs_t + 1e-8)
            loss_D = w_s * loss_Ds + w_t * loss_Dt
            loss_D.backward()
            self.optim_da.step()

            if self.experiment:
                with torch.no_grad():
                    acc_s = (dom_s.argmax(1) == 0).float().mean().item()
                    acc_t = (dom_t.argmax(1) == 1).float().mean().item()
                    self.experiment.log_metric("D_acc_src", acc_s, step=self.step)
                    self.experiment.log_metric("D_acc_tgt", acc_t, step=self.step)
                    self.experiment.log_metric("D_loss_raw", float((loss_Ds + loss_Dt).item()), step=self.step)

            # ========= Step-G：经 GRL 愚弄判别器 + 源域监督 =========
            # 再前向拿可反传张量
            self.optim.zero_grad(set_to_none=True)
            _, _, f_s_G, s_logit_G = self.model(v_d, v_d_3d, sm, v_p, esm, self.device)  # ⭐ 增加v_d_3d
            _, _, f_t_G, t_logit_G = self.model(v_d_t, v_d_3d_t, smt, v_p_t, esmt, self.device)  # ⭐ 增加v_d_3d_t

            ps_g = torch.softmax(s_logit_G.detach(), dim=1)
            pt_g = torch.softmax(t_logit_G.detach(), dim=1)


            rev_fs = ReverseLayerF.apply(f_s_G, self.alpha)
            rev_ft = ReverseLayerF.apply(f_t_G, self.alpha)
            feat_s_G = _cond_feat(rev_fs, ps_g)   # 不 detach
            feat_t_G = _cond_feat(rev_ft, pt_g)

            # 冻结判别器，避免累积梯度
            dmm_requires = [p.requires_grad for p in self.domain_dmm.parameters()]
            for p in self.domain_dmm.parameters():
                p.requires_grad_(False)

            dom_s_adv = self.domain_dmm(feat_s_G)
            dom_t_adv = self.domain_dmm(feat_t_G)

            if self.use_da_entropy:
                tgt_w = self._compute_entropy_weights(t_logit_G)  # 用 G 分支的 logits
            else:
                tgt_w = None

            _, adv_s = cross_entropy_logits(dom_s_adv, torch.zeros(dom_s_adv.size(0), device=self.device))
            if self.use_da_entropy:
                _, adv_t = cross_entropy_logits(dom_t_adv, torch.ones(dom_t_adv.size(0), device=self.device), tgt_w)
            else:
                _, adv_t = cross_entropy_logits(dom_t_adv, torch.ones(dom_t_adv.size(0), device=self.device))

            # 对称权重
            bs_s_adv, bs_t_adv = dom_s_adv.size(0), dom_t_adv.size(0)
            w_s_adv = bs_t_adv / (bs_s_adv + 1e-8)
            w_t_adv = bs_s_adv / (bs_t_adv + 1e-8)

            da_adv_loss = epoch_lamb_da * (w_s_adv * adv_s + w_t_adv * adv_t)

            loss = cls_loss + aux_loss + da_adv_loss
            loss.backward()
            # ⭐ BAN配置不使用梯度裁剪，CrossMamba使用梯度裁剪
            if not self.use_simple_loss:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optim.step()

            # 恢复判别器 flags
            for p, rg in zip(self.domain_dmm.parameters(), dmm_requires):
                p.requires_grad_(rg)

            # 统计
            self.step += 1
            total_loss_epoch += float(loss.item())
            model_loss_epoch += float((cls_loss + aux_loss).item())
            da_loss_epoch += float(da_adv_loss.item())

            if self.experiment:
                self.experiment.log_metric("train_step model loss", float((cls_loss + aux_loss).item()), step=self.step)
                self.experiment.log_metric("train_step total loss", float(loss.item()), step=self.step)
                self.experiment.log_metric("train_step da loss", float(da_adv_loss.item()), step=self.step)
                self.experiment.log_metric("DA lambda (batch)", float(epoch_lamb_da), step=self.step)

        total_loss_epoch /= max(1, num_batches)
        model_loss_epoch /= max(1, num_batches)
        da_loss_epoch    /= max(1, num_batches)

        if self.current_epoch < self.da_init_epoch:
            print(f"Training at Epoch {self.current_epoch} with model training loss {total_loss_epoch}")
        else:
            print(f"Training at Epoch {self.current_epoch} model training loss {model_loss_epoch}, "
                  f"da loss {da_loss_epoch}, total training loss {total_loss_epoch}, "
                  f"DA lambda {epoch_lamb_da_last}")

        return total_loss_epoch, model_loss_epoch, da_loss_epoch, epoch_lamb_da_last

    # =========================
    # 评估（val/test）
    # =========================
    def test(self, dataloader="test", calibrate=False, fixed_threshold=None):
        test_loss = 0.0
        y_label, y_pred_prob = [], []

        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")

        num_batches = len(data_loader)
        with torch.no_grad():
            self.model.eval()
            for v_d, v_d_3d, sm, v_p, esm, labels, drug_3d_data in data_loader:  # ⭐ 增加v_d_3d和drug_3d_data
                sm  = torch.tensor(sm,  dtype=torch.float32).reshape(sm.shape[0], 1, 384).to(self.device)
                esm = torch.tensor(esm, dtype=torch.float32).reshape(esm.shape[0], 1, 1280).to(self.device)
                v_d, v_d_3d, v_p = v_d.to(self.device), v_d_3d.to(self.device), v_p.to(self.device)  # ⭐ 增加v_d_3d
                labels = labels.float().to(self.device)

                # val 用当前模型；test 用 best_model
                if dataloader == "val":
                    _, _, _, score = self.model(v_d, v_d_3d, sm, v_p, esm, self.device)  # ⭐ 增加v_d_3d
                else:
                    _, _, _, score = self.best_model(v_d, v_d_3d, sm, v_p, esm, self.device)  # ⭐ 增加v_d_3d

                # 温度标定（只在 test 且 calibrate=True）
                if dataloader == "test" and calibrate and (getattr(self, "temp_scaler", None) is not None):
                    score = self.temp_scaler(score)

                # 评估不加 margin
                if self.n_class == 1:
                    n, loss = binary_cross_entropy(score, labels)
                else:
                    n, loss = cross_entropy_logits(score, labels, margin=0.0)

                test_loss += float(loss.item())
                y_label.extend(labels.detach().cpu().tolist())
                y_pred_prob.extend(n.detach().cpu().tolist())

        y_true = np.asarray(y_label, dtype=np.int64)
        y_scores = np.asarray(y_pred_prob, dtype=np.float64)

        auroc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.5
        auprc = average_precision_score(y_true, y_scores)
        test_loss = test_loss / max(1, num_batches)

        # 验证路径：只返回三元组
        if dataloader == "val":
            return auroc, auprc, test_loss

        # 测试路径：阈值（若没给 fixed_threshold，就在测试集内部取最大F1 —— 一般不用）
        if fixed_threshold is None:
            prec_arr, rec_arr, thr_pr = precision_recall_curve(y_true, y_scores)
            f1_all = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-12)
            if thr_pr.size > 0:
                best_idx = int(np.nanargmax(f1_all[:-1]))
                thred_optim = float(thr_pr[best_idx])
            else:
                thred_optim = 0.5
        else:
            thred_optim = float(fixed_threshold)

        y_pred_bin = (y_scores >= thred_optim).astype(int)
        cm1 = confusion_matrix(y_true, y_pred_bin, labels=[0, 1])
        tn, fp, fn, tp = cm1.ravel()

        precision1 = precision_score(y_true, y_pred_bin, zero_division=0)
        recall_pos = tp / (tp + fn + 1e-12)   # sensitivity
        f1 = 2 * precision1 * recall_pos / (precision1 + recall_pos + 1e-12)
        accuracy = (tn + tp) / (tn + fp + fn + tp + 1e-12)
        sensitivity = recall_pos
        specificity = tn / (tn + fp + 1e-12)

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_scores)
        if self.experiment:
            self.experiment.log_curve("test_roc curve", fpr, tpr)
            self.experiment.log_curve("test_pr curve", rec_arr, prec_arr)

        return (
            auroc, auprc, f1,
            sensitivity, specificity, accuracy,
            test_loss, thred_optim, precision1,
            cm1, y_scores
        )

    # =========================
    # 校准相关
    # =========================
    def _fit_temperature_on_val(self):
        assert self.best_model is not None, "best_model 为空：请先跑完一个 val 轮次再拟合温度"
        self.best_model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for v_d, v_d_3d, sm, v_p, esm, labels, drug_3d_data in self.val_dataloader:
                sm  = torch.tensor(sm,  dtype=torch.float32).reshape(sm.shape[0], 1, 384).to(self.device)
                esm = torch.tensor(esm, dtype=torch.float32).reshape(esm.shape[0], 1, 1280).to(self.device)
                v_d, v_d_3d, v_p = v_d.to(self.device), v_d_3d.to(self.device), v_p.to(self.device)
                labels = labels.float().to(self.device)

                _, _, _, score = self.best_model(v_d, v_d_3d, sm, v_p, esm, self.device)  # logits [B,2]
                all_logits.append(score.detach())
                all_labels.append(labels.detach())

        logits_val = torch.cat(all_logits, dim=0)
        labels_val = torch.cat(all_labels, dim=0)

        self.temp_scaler = TemperatureScaler().to(self.device)
        self.temp_scaler.fit(logits_val, labels_val)
        print("[TempScale] Fitted T =", float(self.temp_scaler.log_t.exp()))

    def _scores_and_labels(self, loader, use_best=False, calibrate=False):
        """收集一个 loader 上的 (y_true, y_score, avg_loss)。use_best=True 用 self.best_model。"""
        self.model.eval()
        model = self.best_model if use_best and (self.best_model is not None) else self.model
        y_true, y_score = [], []
        tot_loss, n_batch = 0.0, 0
        with torch.no_grad():
            for v_d, v_d_3d, sm, v_p, esm, labels, drug_3d_data in loader:
                sm  = torch.tensor(sm,  dtype=torch.float32).reshape(sm.shape[0], 1, 384).to(self.device)
                esm = torch.tensor(esm, dtype=torch.float32).reshape(esm.shape[0], 1, 1280).to(self.device)
                v_d, v_d_3d, v_p = v_d.to(self.device), v_d_3d.to(self.device), v_p.to(self.device)
                labels = labels.float().to(self.device)
                _, _, _, logits = model(v_d, v_d_3d, sm, v_p, esm, self.device)
                if calibrate and (getattr(self, "temp_scaler", None) is not None):
                    logits = self.temp_scaler(logits)
                score = torch.softmax(logits, dim=1)[:, 1]
                _, loss = cross_entropy_logits(logits, labels)  # 评估期不加 margin
                tot_loss += float(loss.item())
                n_batch += 1
                y_true.extend(labels.detach().cpu().tolist())
                y_score.extend(score.detach().cpu().tolist())
        return np.asarray(y_true, int), np.asarray(y_score, float), (tot_loss / max(1, n_batch))

    def _select_threshold_on_val(self, calibrate=False):
        """在验证集上选取最大 F1 的阈值，并保存为 self.fixed_threshold。"""
        yv, pv, _ = self._scores_and_labels(self.val_dataloader, use_best=True, calibrate=calibrate)
        prec, rec, thr = precision_recall_curve(yv, pv)
        f1_all = 2 * prec * rec / (prec + rec + 1e-12)
        if thr.size == 0:
            self.fixed_threshold = 0.5
            return self.fixed_threshold, float(np.nanmax(f1_all))
        best_idx = int(np.nanargmax(f1_all[:-1]))
        self.fixed_threshold = float(thr[best_idx])
        return self.fixed_threshold, float(f1_all[best_idx])


    def _baseline_like_f1(self, y_true, y_score, ignore_first_k=5, assume_pi_half=True):
        """
        - 先从 ROC 得到 (fpr, tpr, thresholds)
        - 用 tpr,fpr 近似 precision（可选：假设 π=0.5，或用真实先验 π）
        - 在测试集内部选使 F1 最大的阈值（还跳过前 k 个点）
        返回: f1_max, thr, cm
        """
        fpr, tpr, thr = roc_curve(y_true, y_score)
        if assume_pi_half:
            prec_approx = tpr / (tpr + fpr + 1e-12)  # baseline 的写法（乐观）
        else:
            pi = float(np.mean(y_true))               # 用真实先验更合理；与 baseline 不同
            prec_approx = (pi * tpr) / (pi * tpr + (1.0 - pi) * fpr + 1e-12)

        f1 = 2 * prec_approx * tpr / (prec_approx + tpr + 1e-12)

        # 对齐 baseline 的 "thresholds[5:]" hack
        k = min(ignore_first_k, len(thr) - 1)
        if k < len(f1):
            idx_rel = np.argmax(f1[k:])
            best_idx = k + int(idx_rel)
        else:
            best_idx = int(np.argmax(f1))

        thr_best = float(thr[best_idx]) if len(thr) > 0 else 0.5
        y_pred_bin = (y_score >= thr_best).astype(int)
        cm = confusion_matrix(y_true, y_pred_bin, labels=[0,1])

        return float(f1[best_idx]), thr_best, cm
