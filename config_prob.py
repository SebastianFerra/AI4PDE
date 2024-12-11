
# ------------------------------ network settings ---------------------------------------------------
iteration = 20
D_in = 2
H = 30
D_out = 2
lr = 0.01
# ------------------------------ material parameter -------------------------------------------------
model_energy = 'FH2D'
phi_0 = 0.2
chi = 0.1
gamma = 0.0010980966325036604
# ----------------------------- define structural parameters ---------------------------------------
Length = 15.0
Height = 5.0
Depth = 1.0
known_left_ux = 0
known_left_uy = 0
bc_left_penalty = 1.0

known_right_tx = 0
known_right_ty = 0
bc_right_penalty = 1.0

known_top_tx = 0
known_top_ty = 0
bc_top_penalty = 1.0

known_bottom_ux = 0
known_bottom_uy = 0
bc_bottom_penalty = 1.0

# ------------------------------ define domain and collocation points -------------------------------
Nx = 200 # 120  # 120
Ny = 50 # 30  # 60
x_min, y_min = (0.0, 0.0)
hx = Length / (Nx - 1)
hy = Height / (Ny - 1)
shape = [Nx, Ny]
dxdy = [hx, hy]

# ------------------------------ data testing -------------------------------------------------------
num_test_x = 200
num_test_y = 50