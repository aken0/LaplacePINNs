import numpy as np 
import torch
import matplotlib.pyplot as plt
import scipy.io 
import laplacePinn 

def getLaplace(iters=30000,n_col=1000,n_boundary=500,verbose=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def diff(u,x,order=1):
        grad = torch.autograd.grad(u.sum(),x,create_graph=True)[0]
        for _ in range(1,order):
            grad = torch.autograd.grad(grad.sum(),x,create_graph=True)[0]
        return grad

    def f_u(x,t,net):
        input = torch.hstack([x,t]).requires_grad_()#.to(_device)
        u = net(input)
        u_x = diff(u,x)
        u_t = diff(u,t)
        u_xx = diff(u_x,x)
        return (u_t + u*u_x - (0.01/np.pi)*u_xx)

    def b_u(x,t,net):
        _device = next(net.parameters()).device
        input = torch.hstack([x,t]).requires_grad_()#.to(_device)
        u=net(input)
        u_star=torch.as_tensor([-torch.sin(i*np.pi) if j==0 else 0. for i,j in zip(x,t)])[:,None].to(_device)
        return u-u_star


    data = scipy.io.loadmat('./burgers_shock.mat')
    t_ = torch.tensor(data['t'].flatten()).float()
    x_ = torch.tensor(data['x'].flatten()).float()
    Exact = np.real(data['usol']).T

    X,T=torch.meshgrid(x_,t_,indexing='xy')
    X_exact=X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_exact = Exact.flatten()[:,None]              

    t = torch.linspace(0,1,int(n_boundary/2+1)).float()
    x = torch.linspace(-1,1,int(n_boundary+1)).float()
    bc_l=torch.vstack([-1*torch.ones_like(t),t]).T
    bc_r=torch.vstack([torch.ones_like(t),t]).T
    ic=torch.vstack([x,torch.zeros_like(x)]).T
    data_grid=torch.vstack([bc_l,bc_r,ic])
    x=data_grid[:,0]
    t=data_grid[:,1]
    u_star=torch.as_tensor([-np.sin(i[0]*np.pi) if i[1]==0 else 0. for i in data_grid])[:,None]

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 16), torch.nn.Tanh(),
        torch.nn.Linear(16, 16), torch.nn.Tanh(),
        torch.nn.Linear(16, 16), torch.nn.Tanh(),
        torch.nn.Linear(16, 1)).to(device)

    mse = torch.nn.MSELoss(reduction='sum') 
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-2)

    iters=iters
    N_col=n_col

    x=x.flatten()[:,None]
    t=t.flatten()[:,None]
    data_grid_used=data_grid.to(device).requires_grad_()

    x_bc = torch.as_tensor(x,device=device).float()
    t_bc = torch.as_tensor(t,device=device).float()
    u_bc = torch.as_tensor(u_star,device=device).float().requires_grad_()


    ###Grid
    #x11=torch.linspace(-1,-.01,int(np.sqrt(N_col)/2),device=device)[1:]   
    #x12=torch.linspace(.01,1,int(np.sqrt(N_col)/2),device=device)[:-1]   
    #x1=torch.cat([x11,x12])
    #equidistant grid
    x1=torch.linspace(-1,1,int(np.sqrt(N_col)/2)*2,device=device)[1:-1]   
    x2=torch.linspace(0,1,int(np.sqrt(N_col)),device=device)[1:-1]   
    A,B=(torch.meshgrid(x1,x2,indexing='xy'))
    stack=(torch.dstack([A.flatten(),B.flatten()]).requires_grad_().squeeze())
    x_col=stack[:,0].requires_grad_()[:,None]
    t_col=stack[:,1].requires_grad_()[:,None]
    dots=torch.hstack([x_col,t_col])

    zeros = torch.zeros((x_col.shape[0],1)).to(device)

    for i in range(iters):
        optimizer.zero_grad(set_to_none=True) 
        #mse_u
        mse_u=mse(model(data_grid_used),u_bc)
        #mse_f
        f_out = f_u(dots[:,0,None], dots[:,1,None], model) 
        mse_f = mse(f_out, zeros)
        loss = torch.add(mse_u ,  mse_f)
        loss.backward() 
        optimizer.step() 
        if ((i+1)%1000==0 and verbose):
            with torch.autograd.no_grad():
                print(i+1,"Training Loss:",loss.data)
                
    #laplace
    la=laplacePinn.Laplace_s3(model,X_f=dots,X_b=data_grid_used,f_u=f_u,b_u=b_u,optim=True,Epochs=10000,use_pinv=True)
    post_mean,post_variance=la(torch.as_tensor(X_star).requires_grad_())


    pred=model(torch.tensor(X_exact).to(device)).cpu().detach().numpy()
    u_err=(u_exact-pred)
    return u_err,post_mean.cpu().detach(),post_variance.detach().cpu(),la.posterior_covariance.detach().cpu(),dots.detach().cpu()
