import torch

adv_batch_size=0
adv_target_size=0

class PGDAttacker:
    def __init__(self, epsilon, k, a, random_start, loss_func, restriction, target):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start
        self.loss_func=loss_func
        self.restriction=restriction
        self.target=target

    def attack(self, original_x, y, model, device):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        original_x=original_x.clone()
        x = original_x.detach()
        if self.rand:
            x+=self.epsilon*(2*torch.rand_like(x)-1)
            x=torch.clamp(x,0,1)

        for i in range(self.k):
            x.requires_grad=True
            _,pred=model(x)
            if self.target== 'det':
                loss, _=self.loss_func(pred,y.to(device))
            elif self.target== 'cls':
                loss = self.loss_func.cls(pred, y.to(device))
            elif self.target== 'obj':
                loss = self.loss_func.obj(pred, y.to(device))
            elif self.target=='cls_obj':
                loss=self.loss_func.cls_obj(pred,y.to(device))
            else:
                raise NotImplementedError
            g = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]

            x=x+self.a*torch.sign(g)

            if self.restriction=='linf':
                x = torch.clamp(x, original_x - self.epsilon, original_x + self.epsilon)
            elif self.restriction == 'l2':
                dist = (x - original_x)
                dist = dist.view(x.shape[0], -1)
                dist_norm = torch.norm(dist, dim=1, keepdim=True)
                mask = (dist_norm > self.epsilon).unsqueeze(2).unsqueeze(3)
                # dist = F.normalize(dist, p=2, dim=1)
                dist = dist / dist_norm
                dist *= self.epsilon
                dist = dist.view(x.shape)
                x = (original_x + dist) * mask.float() + x * (1 - mask.float())
            else:
                raise NotImplementedError
            x = torch.clamp(x, 0, 1).detach()  # ensure valid pixel range

        return x