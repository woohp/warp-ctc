import torch
from torch.autograd import Function
from torch.nn import Module
from torch.nn.modules.loss import _assert_no_grad
from . import _warp_ctc

__all__ = []


class _CTC(Function):
    def forward(self, acts, labels, act_lens, label_lens):
        acts = acts.contiguous()
        is_gpu = acts.is_cuda
        if is_gpu:
            acts = acts.cpu()
        # loss_func = _warp_ctc.gpu_ctc if acts.is_cuda else _warp_ctc.cpu_ctc
        loss_func = _warp_ctc.cpu_ctc
        grads = torch.zeros(acts.size()).type_as(acts)
        minibatch_size = acts.size(1)
        costs = torch.zeros(minibatch_size).cpu()
        loss_func(
            acts._cdata,
            grads._cdata,
            labels._cdata,
            label_lens._cdata,
            act_lens._cdata,
            minibatch_size,
            costs._cdata
        )
        self.grads = grads
        self.costs = torch.FloatTensor([costs.sum()])
        if is_gpu:
            self.grads = self.grads.cuda()
        return self.costs

    def backward(self, grad_output):
        return self.grads, None, None, None


class CTCLoss(Module):
    def __init__(self):
        super(CTCLoss, self).__init__()

    def forward(self, acts, labels, act_lens, label_lens):
        """
        acts: Tensor of (seqLength x batch x outputDim) containing output from network
        labels: 1 dimensional Tensor containing all the targets of the batch in one sequence
        act_lens: Tensor of size (batch) containing size of each output sequence from the network
        act_lens: Tensor of (batch) containing label length of each example
        """
        assert len(labels.size()) == 1  # labels must be 1 dimensional
        _assert_no_grad(labels)
        _assert_no_grad(act_lens)
        _assert_no_grad(label_lens)
        return _CTC()(acts, labels, act_lens, label_lens)
