 

from ngsolve import *
from netgen.geom2d import SplineGeometry
import torch
from netgen.occ import *
import netgen.meshing as ngm
# from ngsPETSc import NonLinearSolver
# from mpi4py.MPI import COMM_WORLD
import numpy as np
from random import uniform
import pickle
from time import time

## Generate geometry
# Problem 2
n_samples = 30
for i in range(n_samples):
    print(f"Sample {i}")
    w = 3.0 + uniform(-0.5, 5)
    l = 15 + uniform(-5, 5)

    rect = SplineGeometry()
    pnts = [(0,0), (l,0), (l,w), (0,w)]
    p1,p2,p3,p4 = [rect.AppendPoint(*pnt) for pnt in pnts]
    curves = [[["line",p1,p2],"bottom"],
            [["line",p2,p3],"right"],
            [["line",p3,p4],"top"],
            [["line",p4,p1],"left"]]
    [rect.Append(c,bc=bc, leftdomain=1, rightdomain=0) for c,bc in curves]

    # save the geometry

    ## get problem parameters 

    phi0 = 0.3 + uniform(-0.1, 0.1)
    chi = 0.2 + uniform(-0.1, 0.1)
    G = 0.15
    GRID_SZ = 32
    h = 1
    ord = 1
    N = 1000
    KBTV = 136.6
    form = "EDP" # EDP //functional

    ## Generate mesh and geometry ### add parallel stuff
    def mesher(geom, h):
        mesh = Mesh(geom.GenerateMesh(maxh=h))
        print(mesh.GetBoundaries())
        return mesh
    mesh = mesher(rect, h)
    Draw(mesh)

    x = np.linspace(0, l, num= GRID_SZ)
    y = np.linspace(0,w, num = GRID_SZ)
    xx, yy = np.meshgrid(x, y, indexing='xy')
    interp_points = np.dstack((xx, yy)).reshape((-1, 2))
    mesh_interp_points = np.array([mesh(*p) for p in interp_points])



    def Norm(vec):
        return InnerProduct(vec, vec)**0.5

    def div_custom(A):
        return CF((div(A[0]), div(A[1])))

    def Gel_energy_EDP(F): ## |F|^2 + H => gamma F:Gradv + H'*J'
        # ddet(A(t))/dt = det(A(t))*trace(A^-1(t)*Grad (v))
        gamma = G/KBTV
        J = Det(F)
        phi = phi0/J
        dv = Grad(v)
        invF = Inv(F)
        H_prime = -phi/N + log(1-phi) + phi + chi*phi**2
        edp = gamma * InnerProduct(F,dv) + H_prime * J * Trace(invF*dv)
        return edp

    def Gel_energy_mixed(u,v,Ptup,Ttup,Paux,Taux,lam,mu): ## |F|^2 + H => gamma F:Gradv + H'*J'

        gamma = G/KBTV
        J = Det(F)
        phi = phi0/J
        invF = Inv(F)
        H_prime = -phi/N +log(1-phi) + phi + chi*phi**2
        # P = F+H_prime*J*F^{-T}
        tens_eq = InnerProduct( gamma * F + H_prime * J * invF.trans - Paux , Taux)
                        
        div_eq = InnerProduct(div_custom(Ptup),v) 
        # agregar int((u1,u2,0) * tau.n) = 0 (this only on BC z = 0)
        return tens_eq + div_eq + lam * u + mu * v 


    ## Generate spaces and forms
    fesU = VectorH1(mesh, order=ord)
    fesP1 = HDiv(mesh, order=ord+1, dirichlet = "left|right|top|bottom")
    fesP = FESpace([fesP1, fesP1])
    R = NumberSpace(mesh)
    fes = fesU * fesP * R * R 
    u,P1,P2,mu1,mu2= fes.TrialFunction()
    P = (P1,P2)
    v,T1,T2,lam1,lam2 = fes.TestFunction()
    T = (T1,T2)
    BF = BilinearForm(fes)
    F = Id(2) + Grad(u)
    lam = CF((lam1,lam2))
    mu = CF((mu1,mu2))

    ## Assemble forms
    def Assemble_Bilinear_Form(BF, u,v=None,P=None,T=None,lam=None,mu=None, form = "Mixed"):
        if form == "EDP":
            BF += Gel_energy_EDP(F).Compile() * dx
            return BF
        elif form == "Mixed":
            Paux = CoefficientFunction((P[0][0],P[0][1],P[1][0],P[1][1]), dims = (2,2))
            Taux = CoefficientFunction((T[0][0],T[0][1],T[1][0],T[1][1]), dims = (2,2))
            BF += Gel_energy_mixed(u,v,P,T,Paux,Taux,lam,mu).Compile() * dx
            return BF

    BF = Assemble_Bilinear_Form(BF, u,v,P,T,lam,mu)

    def Solver_freeswell(BF, gfu, tol=1e-8, maxiter=250, damp = 0.5):
        """
        Solves the problem
        """
        res = gfu.vec.CreateVector()
        w = gfu.vec.CreateVector()
        history = GridFunction(fes, multidim = 0)
        # here we may need to add another loop
    
        for iter in range(maxiter):
            Draw(gfu.components[0], mesh, "u")
            # Prints before the iteration: number of it, residual, energy
            print("Energy: ", BF.Energy(gfu.vec), "Residual: ", sqrt(abs(InnerProduct(res,res))), "Iteration: ", iter)
            BF.Apply(gfu.vec, res)
            BF.AssembleLinearization(gfu.vec)
            inv = BF.mat.Inverse(freedofs = fes.FreeDofs())        
            w.data = damp * inv * res
            gfu.vec.data -= w
            history.AddMultiDimComponent(gfu.vec)
            if sqrt(abs(InnerProduct(w,res))) < tol:
                print("Converged")
                break
        return gfu, history


    gfu = GridFunction(fes)
    gfu.vec[:] = 0
    t1 =  time()
    S, history = Solver_freeswell(BF, gfu, damp = 0.5)
    img_S = np.array([S(p) for p in mesh_interp_points]).reshape((-1, GRID_SZ))
    torch.save(torch.tensor(img_S).float(), f"DATA/_{i}S.pt")

# pickle the results, history and mesh for later use
#pickle.dump(history, open(f"Sol_Problem{problem[-1]}/history_{form}.p", "wb"))


