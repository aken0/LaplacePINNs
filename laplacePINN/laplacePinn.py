import numpy as np 
import torch
import asdfghjkl
#from torch.nn.utils import parameters_to_vector

def loss_fn_Jac(outputs,targets):
    return outputs[:, 0].sum()
def _flatten_after_batch(tensor: torch.Tensor):
    if tensor.ndim == 1:
        return tensor.unsqueeze(-1)
    else:
        return tensor.flatten(start_dim=1)
def _get_batch_grad(model):
    batch_grads = list()
    for module in model.modules():
        if hasattr(module, 'op_results'):
            res = module.op_results['batch_grads']
            if 'weight' in res:
                batch_grads.append(_flatten_after_batch(res['weight']))
            if 'bias' in res:
                batch_grads.append(_flatten_after_batch(res['bias']))
            if len(set(res.keys()) - {'weight', 'bias'}) > 0:
                raise ValueError(f'Invalid parameter keys {res.keys()}')
    return torch.cat(batch_grads, dim=1)

def batch_gradient(model, loss_fn, X, targets,Functional=None):
    with asdfghjkl.extend(model, 'batch_grads'):
        model.zero_grad()
        if Functional:
            f=Functional(X[:,0,None],X[:,1,None],model)
        else:
            f = model(X)
        loss = loss_fn(f, targets)
        loss.backward()
    return f

def asdl_hessian(model, loss,X,y,Functional=None):
    params = [p for p in model.parameters() if p.requires_grad]
    if Functional:
        output=Functional(X[:,0,None],X[:,1,None],model)
    else:
        output=model(X)
    loss=loss(output,y)
    return loss.detach(),asdfghjkl.hessian(loss,params)

class Laplace_s3:
    def __init__(self,model, X_f,X_b,f_u,b_u,optim=True,Epochs=1000,use_fisher=False,use_pinv=False):

        ###constructor
        self.model=model
        self._device = next(model.parameters()).device
        self.prior_precision = 1.
        X=torch.vstack([X_f,X_b])
        self.n_data=len(X)
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.n_params=len(torch.cat([p.flatten() for p in model.parameters() if p.requires_grad]))
        ###fit 
        self.model.eval()

        with torch.no_grad():
            try:
                out = self.model(X[:1].to(self._device))
            except (TypeError, AttributeError):
                out = self.model(X.to(self._device))
        self.n_outputs = out.shape[-1]
        setattr(self.model, 'output_size', self.n_outputs)

        self.model.zero_grad()
        X_b, X_f  = X_b.to(self._device), X_f.to(self._device)
        lossfunc=torch.nn.MSELoss(reduction='sum')
        self.factor=.5

        #use fisher approximation for hessian
        if use_fisher:
            batch_gradient(model,loss_fn_Jac,X_f,None,f_u)
            Js_f=_get_batch_grad(model)
            Js_f=Js_f.view(Js_f.shape[0],1,-1)
            batch_gradient(model,loss_fn_Jac,X_b,None,b_u)
            Js_b=_get_batch_grad(model)
            Js_b=Js_b.view(Js_b.shape[0],1,-1)
            fs=f_u(X_f[:,0,None].requires_grad_(),X_f[:,1,None].requires_grad_(),model)
            bs=b_u(X_b[:,0,None].requires_grad_(),X_b[:,1,None].requires_grad_(),model)
            loss_f = lossfunc(fs, torch.zeros_like(fs))
            loss_b = lossfunc(bs, torch.zeros_like(bs))
            self.H_f = torch.einsum('mkp,mkq->pq', Js_f, Js_f).detach()
            self.H_b = torch.einsum('mkp,mkq->pq', Js_b, Js_b).detach()
        #use full hessian
        else:
            loss_f,H_f=asdl_hessian(model,lossfunc,X_f,torch.zeros(X_f.shape[0],1).to(self._device),f_u)
            loss_b,H_b=asdl_hessian(model,lossfunc,X_b,torch.zeros(X_b.shape[0],1).to(self._device),b_u)
            self.H_f=self.factor*H_f.detach()
            self.H_b=self.factor*H_b.detach()

        self.loss_f=self.factor*loss_f.detach()
        self.loss_b=self.factor*loss_b.detach()
        self.loss=torch.add(self.loss_f,self.loss_b)
        self.H = self.H_f.detach() + self.H_b.detach()

        ###hyperparam optim
        if optim:
            #initialize prior hyperparams
            log_prior = torch.ones(1, requires_grad=True,device=self._device)
            log_sigma_f, log_sigma_b=torch.ones(1, requires_grad=True,device=self._device), torch.ones(1, requires_grad=True,device=self._device)
            hyper_optimizer = torch.optim.Adam([log_prior, log_sigma_f,log_sigma_b], lr=1e-2)
            self.posterior_precision=self.H + torch.diag(1.0*torch.ones(self.n_params, device=self._device))
            for i in range(Epochs):
                hyper_optimizer.zero_grad()
                self.sigma_f=log_sigma_f.exp()
                self.sigma_b=log_sigma_b.exp()

                self.prior_precision=log_prior.exp()
                c=len(X_f) * torch.log(self.sigma_f*np.sqrt(2*np.pi)) + len(X_b) * torch.log(self.sigma_b*np.sqrt(2*np.pi))
                H_factor_f=  1 / (2*self.sigma_f.square())
                H_factor_b=  1 / (2*self.sigma_b.square())

                H_f_scale=len(X_f)
                H_b_scale=len(X_b)
                self.log_likelihood= -H_factor_f *H_f_scale * self.loss_f - H_factor_b*H_b_scale *self.loss_b - c

                self.posterior_precision=((H_factor_f*self.H_f + H_factor_b*self.H_b) + torch.diag(self.prior_precision * torch.ones(self.n_params, device=self._device)))
                self.post_det=self.posterior_precision.slogdet()[1]
                self.prior_det=(self.prior_precision * torch.ones(self.n_params, device=self._device)).log().sum()
                self.log_det_ratio =  self.post_det - self.prior_det
                self.prior_mean=torch.zeros_like(self.mean)
                delta = (self.mean - self.prior_mean)
                self.scatter= (delta * self.prior_precision) @ delta
                neg_marglik= -(self.log_likelihood - 0.5 * (self.log_det_ratio + self.scatter))

                neg_marglik.backward()
                hyper_optimizer.step() 

            if use_pinv:
                self.posterior_covariance=torch.linalg.pinv(self.posterior_precision)
            else:
                chol=torch.linalg.cholesky(self.posterior_precision)
                inv=torch.cholesky_inverse(chol)
                self.posterior_covariance=inv@inv.T
    def __call__(self,X):
        f_mu=self.model(X.to(self._device))
        #Calculate Jacobian of eval data
        asdfghjkl.batch_gradient(self.model,loss_fn_Jac,torch.as_tensor(X).requires_grad_().to(self._device),None)
        Js=_get_batch_grad(self.model)
        Js=Js.view(Js.shape[0],1,-1)
        #Linearized model variance
        f_var= torch.einsum('ncp,pq,nkq->nck', Js, self.posterior_covariance, Js)
        return f_mu,f_var
