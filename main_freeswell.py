import torch
import define_structure as des
from MultiLayerNet import *
import Utility as util
import config_prob as cf
from IntegrationLoss import *
from EnergyModel import *
import EnergyModel as md
import numpy as np
import time

import pickle
mpl.rcParams['figure.dpi'] = 100
# fix random seeds
axes = {'labelsize' : 'large'}
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 17}
legend = {'fontsize': 'medium'}
lines = {'linewidth': 3,
         'markersize' : 7}
mpl.rc('font', **font)
mpl.rc('axes', **axes)
mpl.rc('legend', **legend)
mpl.rc('lines', **lines)

class DeepEnergyMethod:
    def __init__(self,model,numIntType, energy, dim =2):
        self.model = MultiLayerNet(model[0],model[1],model[2])
        self.model.to(dev)
        self.energy = energy
        self.intLoss = IntegrationLoss(numIntType, dim)
        self.dim = dim
        self.lossArray = []
    
    def train_model(self,shape , dxdydz, data, neumannBC, dirichletBC, iteration, learning_rate):
        x = torch.from_numpy(data).float()
        x = x.to(dev)
        x.requires_grad = True


        # No dirichlet BC, just for generality
        dirBC_coordinates = {}  # declare a dictionary
        dirBC_values = {}  # declare a dictionary
        dirBC_penalty = {}
        for i, keyi in enumerate(dirichletBC):
            dirBC_coordinates[i] = torch.from_numpy(dirichletBC[keyi]['coord']).float().to(dev)
            dirBC_values[i] = torch.from_numpy(dirichletBC[keyi]['known_value']).float().to(dev)
            dirBC_penalty[i] = torch.tensor(dirichletBC[keyi]['penalty']).float().to(dev)
        # Neumann BC
        neuBC_coordinates = {}
        neuBC_values = {}   
        neuBC_penalty = {}
        for i, keyi in enumerate(neumannBC):
            neuBC_coordinates[i] = torch.from_numpy(neumannBC[keyi]['coord']).float().to(dev)
            neuBC_coordinates[i].requires_grad_(True)
            neuBC_values[i] = torch.from_numpy(neumannBC[keyi]['known_value']).float().to(dev)
            neuBC_penalty[i] = torch.tensor(neumannBC[keyi]['penalty']).float().to(dev)
        optimizer = torch.optim.LBFGS(self.model.parameters(), lr=learning_rate, max_iter=20)
        start_time = time.time()
        energy_loss_array = []
        boundary_loss_array = []
        for t in range(iteration):
            # Zero gradients, perform a backward pass, and update the weights.
            def closure():
                it_time = time.time()
                # ----------------------------------------------------------------------------------
                # Internal Energy
                # ----------------------------------------------------------------------------------
                u_pred = self.getU(x)
                u_pred.double()
                storedEnergy = self.energy.getStoredEnergy(u_pred, x)
                internal2 = self.intLoss.lossInternalEnergy(storedEnergy, dx=dxdydz[0], dy=dxdydz[1], shape=shape)
                external2 = torch.zeros(len(neuBC_coordinates))
                for i, vali in enumerate(neuBC_coordinates):
                    neu_u_pred = self.getU(neuBC_coordinates[i])
                    fext = torch.bmm((neu_u_pred + neuBC_coordinates[i]).unsqueeze(1), neuBC_values[i].unsqueeze(2))
                    external2[i] = self.intLoss.lossExternalEnergy(fext, dx=dxdydz[1])
                bc_u_crit = torch.zeros((len(dirBC_coordinates)))
                for i, vali in enumerate(dirBC_coordinates):
                    dir_u_pred = self.getU(dirBC_coordinates[i])
                    bc_u_crit[i] = self.loss_squared_sum(dir_u_pred, dirBC_values[i])
                energy_loss = internal2 - torch.sum(external2)
                boundary_loss = torch.sum(bc_u_crit)
                loss = energy_loss + boundary_loss
                optimizer.zero_grad()
                loss.backward()
                print('Iter: %d Loss: %.9e Energy: %.9e Boundary: %.9e Time: %.3e'
                      % (t + 1, loss.item(), energy_loss.item(), boundary_loss.item(), time.time() - it_time))
                energy_loss_array.append(energy_loss.data)
                boundary_loss_array.append(boundary_loss.data)
                self.lossArray.append(loss.data)
                return loss
            optimizer.step(closure)
        elapsed = time.time() - start_time
        print('Training time: %.4f' % elapsed)
    def getU(self, x):
        u = self.model(x)
        Ux = x[:, 0] * u[:, 0]
        Uy = x[:, 0] * u[:, 1]
        Ux = Ux.reshape(Ux.shape[0], 1)
        Uy = Uy.reshape(Uy.shape[0], 1)
        u_pred = torch.cat((Ux, Uy), -1)
        return u_pred
    def evaluate_model(self,x,y,z):
            chi = self.energy.chi
            phi_0 = self.energy.phi_0
            gamma = self.energy.gamma
            Nx = len(x)
            Ny = len(y)
            xGrid, yGrid = np.meshgrid(x, y)
            x1D = xGrid.flatten()
            y1D = yGrid.flatten()
            xy = np.concatenate((np.array([x1D]).T, np.array([y1D]).T), axis=-1)
            xy_tensor = torch.from_numpy(xy).float()
            xy_tensor = xy_tensor.to(dev)
            xy_tensor.requires_grad_(True)
            # u_pred_torch = self.model(xy_tensor)
            u_pred_torch = self.getU(xy_tensor)
            duxdxy = grad(u_pred_torch[:, 0].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device=dev),
                            create_graph=True, retain_graph=True)[0]
            duydxy = grad(u_pred_torch[:, 1].unsqueeze(1), xy_tensor, torch.ones(xy_tensor.size()[0], 1, device=dev),
                            create_graph=True, retain_graph=True)[0]
            F11 = duxdxy[:, 0].unsqueeze(1) + 1
            F12 = duxdxy[:, 1].unsqueeze(1) + 0
            F21 = duydxy[:, 0].unsqueeze(1) + 0
            F22 = duydxy[:, 1].unsqueeze(1) + 1
            detF = F11 * F22 - F12 * F21
            invF11 = F22 / detF
            invF22 = F11 / detF
            invF12 = -F12 / detF
            invF21 = -F21 / detF
            u_pred = u_pred_torch.detach().cpu().numpy()
            surUx = u_pred[:, 0].reshape(Ny, Nx, 1)
            surUy = u_pred[:, 1].reshape(Ny, Nx, 1)
            surUz = np.zeros([Nx, Ny, 1])
            U = (np.float64(surUx), np.float64(surUy), np.float64(surUz))
            return U, surUx, surUy

    @staticmethod
    def loss_sum(tinput):
        return torch.sum(tinput) / tinput.data.nelement()

    # --------------------------------------------------------------------------------
    # purpose: loss square sum for the boundary part
    # --------------------------------------------------------------------------------
    @staticmethod
    def loss_squared_sum(tinput, target):
        row, column = tinput.shape
        loss = 0
        for j in range(column):
            loss += torch.sum((tinput[:, j] - target[:, j]) ** 2) / tinput[:, j].data.nelement()
        return loss
if __name__ == '__main__':
    # ----------------------------------------------------------------------
    #                   STEP 1: SETUP DOMAIN - COLLECT CLEAN DATABASE
    # ----------------------------------------------------------------------
    dom, boundary_neumann, boundary_dirichlet = des.setup_domain()
    x, y, datatest = des.get_datatest()
    # ----------------------------------------------------------------------
    #                   STEP 2: SETUP MODEL
    # ----------------------------------------------------------------------
    mat = md.EnergyModel('floryhuggins', 2, cf.chi, cf.phi_0, cf.gamma)
    dem = DeepEnergyMethod([cf.D_in, cf.H, cf.D_out], 'simpson', mat, 2)
    # dem = DeepEnergyMethod([cf.D_in, cf.H, cf.D_out], 'simpson', mat, 2)
    # ----------------------------------------------------------------------
    #                   STEP 3: TRAINING MODEL
    # ----------------------------------------------------------------------
    start_time = time.time()
    shape = [cf.Nx, cf.Ny]
    dxdy = [cf.hx, cf.hy]
    cf.iteration = 1000
    cf.filename_out = "freeswell"
    dem.train_model(shape, dxdy, dom, boundary_neumann, boundary_dirichlet, cf.iteration, cf.lr)
    end_time = time.time() - start_time
    print("End time: %.5f" % end_time)
    z = np.array([0])
    U, imgx,imgy = dem.evaluate_model(x, y, z)
    img = np.sqrt(imgx ** 2 + imgy ** 2)
    plt.imshow(img, cmap='jet', interpolation='nearest')
    plt.colorbar()
    plt.show()
    # ----------------------------------------------------------------------
    pickle.dump(img, open(cf.filename_out + '.pkl', 'wb'))