import torch
import faiss

import torch.nn as nn
import torch.nn.functional as F



# open-set noisy label contrastive
class CECL(nn.Module):
    def __init__(self, cfg, base_encoder):
        super().__init__()
        self.encoder_q = base_encoder(num_class=cfg.num_class, feat_dim=cfg.low_dim, name=cfg.arch,
                                      pretrained=False)
        # momentum encoder
        self.encoder_k = base_encoder(num_class=cfg.num_class, feat_dim=cfg.low_dim, name=cfg.arch,
                                      pretrained=False)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(cfg.moco_queue, cfg.low_dim))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_Y", torch.zeros(cfg.moco_queue).long())
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("prototypes", torch.zeros(cfg.num_class, cfg.low_dim))
        self.register_buffer("isID", torch.zeros(cfg.moco_queue))

    @torch.no_grad()
    def _momentum_update_key_encoder(self, cfg):
        """
        update momentum encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * cfg.moco_m + param_q.data * (1. - cfg.moco_m)

    @torch.no_grad()
    def _dequeue_and_enqueue1(self, keys, Y, cfg):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        Y = concat_all_gather(Y)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert cfg.moco_queue % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :] = keys
        self.queue_Y[ptr:ptr + batch_size] = Y

        ptr = (ptr + batch_size) % cfg.moco_queue  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue2(self, keys, Y, isid, cfg):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        Y = concat_all_gather(Y)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert cfg.moco_queue % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :] = keys
        self.queue_Y[ptr:ptr + batch_size] = Y
        self.isID[ptr:ptr + batch_size][isid] = 1
        self.isID[ptr:ptr + batch_size][~isid] = 0

        ptr = (ptr + batch_size) % cfg.moco_queue  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x, y):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        y_gather = concat_all_gather(y)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], y_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, y, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        y_gather = concat_all_gather(y)

        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], y_gather[idx_this]


    @torch.no_grad()
    def get_faiss_result(self, feature_dim, need_searched, all_features, id_index, k=100):
        all_features = all_features[id_index.bool()]
        need_searched, all_features = need_searched.numpy(), all_features.numpy()
        faiss_index = faiss.IndexFlatL2(feature_dim)
        faiss_index.add(all_features)
        D, I = faiss_index.search(need_searched, k)
        Mean = D.mean(axis=1)
        return torch.tensor(Mean)


    def reset_prototypes(self, prototypes):
        self.prototypes = prototypes

    def forward(self, img_q, img_k=None, Y=None, nc=None, c=None, cfg=None, mode='s0'):            
        if mode == 'test':
            output, q = self.encoder_q(img_q)
            return output
            # output: logits, q: normarlized feature

        bs = Y.size(0)

        output, q = self.encoder_q(img_q)
        output_x, q_x = output[:bs], q[:bs]
        # output_x_w, q_x_w = output[bs: 2*bs], q[bs: 2*bs]
        # output_x_s, q_x_s = output[2*bs:], q[2*bs:]

        if mode == 's0':
            N, C = output_x.size()

            for feat, label in zip(concat_all_gather(q_x), concat_all_gather(Y)):
                self.prototypes[label] = self.prototypes[label] * cfg.proto_m + (1 - cfg.proto_m) * feat

            # normalize prototypes
            self.prototypes = F.normalize(self.prototypes, p=2, dim=1)

            # compute key features
            with torch.no_grad():  # no gradient
                # update the momentum encoder
                self._momentum_update_key_encoder(cfg)
                # shuffle for making use of BN
                img_k, Y, idx_unshuffle = self._batch_shuffle_ddp(img_k, Y)
                _, k_x = self.encoder_k(img_k)
                # undo shuffle
                k_x, Y = self._batch_unshuffle_ddp(k_x, Y, idx_unshuffle)

            features = torch.cat((q_x, k_x, self.queue.clone().detach()), dim=0)
            train_targets = torch.cat((Y, Y, self.queue_Y.clone().detach()), dim=0)

            # dequeue and enqueue
            self._dequeue_and_enqueue2(k_x, Y, torch.ones(N).bool().cuda(), cfg)


            moco_queue = {'feature': features, 'target': train_targets}

            return output_x, moco_queue


        if mode == 's1':
            N, C = output_x.size()

            with torch.no_grad():
                selected = torch.zeros(N).bool().cuda()
                nc = nc.clone()
                selected[nc] = True
                c = c.clone()
                not_c = (~nc) * c
                index_list = torch.arange(N).cuda()[not_c]
                not_c_features = q_x[index_list]
                not_c_labels = Y[index_list]
                cos_sim = F.cosine_similarity(not_c_features, self.prototypes[not_c_labels], dim=1)
                selected[index_list[cos_sim > cfg.sim_thres]] = True
            for feat, label in zip(concat_all_gather(q_x[selected]), concat_all_gather(Y[selected])):
                self.prototypes[label] = self.prototypes[label] * cfg.proto_m + (1 - cfg.proto_m) * feat

            self.prototypes = F.normalize(self.prototypes, p=2, dim=1)

            # compute key features
            with torch.no_grad():  # no gradient
                # update the momentum encoder
                self._momentum_update_key_encoder(cfg)
                # shuffle for making use of BN
                img_k, Y, idx_unshuffle = self._batch_shuffle_ddp(img_k, Y)
                _, k_x = self.encoder_k(img_k)
                # undo shuffle
                k_x, Y = self._batch_unshuffle_ddp(k_x, Y, idx_unshuffle)

            queue_features = torch.cat((q_x, k_x, self.queue.clone().detach()), dim=0)
            queue_targets = torch.cat((Y, Y, self.queue_Y.clone().detach()), dim=0)
            queue_isid = torch.cat((selected, selected, self.isID.clone().detach()), dim=0)
            # to calculate SupCon Loss using pseudo_labels and target

            # dequeue and enqueue
            self._dequeue_and_enqueue2(k_x, Y, selected, cfg)

            moco_queue = {
                'feature': queue_features,
                'target': queue_targets.long(),
                'IDindex': queue_isid
            }
            return output_x, selected, moco_queue


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

