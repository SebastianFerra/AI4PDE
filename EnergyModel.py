from config import *


class EnergyModel:
    # ---------------------------------------------------------------------------------------------------------------------------
    def __init__(self, energy, dim, param_chi=None, param_phi_0=None, param_gamma=None):
        self.type = energy
        self.dim = dim
        if self.type == 'floryhuggins':
            self.chi = param_chi
            self.phi_0 = param_phi_0
            self.gamma = param_gamma
    def getStoredEnergy(self, u, x):

        if self.type == 'floryhuggins':
            if self.dim == 2:
                return self.FH2D(u, x)
    def FH2D(self,u,x):
        duxdxy = grad(u[:, 0].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        duydxy = grad(u[:, 1].unsqueeze(1), x, torch.ones(x.size()[0], 1, device=dev), create_graph=True, retain_graph=True)[0]
        Fxx = duxdxy[:, 0].unsqueeze(1) + 1
        Fxy = duxdxy[:, 1].unsqueeze(1) + 0
        Fyx = duydxy[:, 0].unsqueeze(1) + 0
        Fyy = duydxy[:, 1].unsqueeze(1) + 1
        frob_mod = Fxx ** 2 + Fxy ** 2 + Fyx ** 2 + Fyy ** 2
        detF = Fxx * Fyy - Fxy * Fyx
        J = detF
        phi = self.phi_0/J
        H = lambda J: (J-self.phi_0)* torch.log(1-phi)+self.phi_0*self.chi*(1-phi)
        energy =  0.5*self.gamma*frob_mod + H(J)
        return energy