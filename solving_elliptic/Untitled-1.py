

def build_matrices(boundary_points,x,y,length_scale, b_matrix, nu):
    n_boundary = boundary_points.shape[0]
    n_points = x.shape[0]
    theta_11 = vmap_kernel(boundary_points, boundary_points, length_scale)
    theta_22_tensor = L_b_xy_tensor(x, y,x,y, length_scale, nu, b_matrix, b_matrix)
    theta_12_tensor = L_b_y_partial(boundary_points[:, 0],boundary_points[:, 1],  x, y, length_scale, nu, b_matrix)

    theta_22_dst_tensor = double_dst_2d(theta_22_tensor)
    theta_12_dst_tensor  = vmap_dst_2d(theta_12_tensor)

    theta_12_dst = jnp.reshape(theta_12_dst_tensor, (n_boundary*4,n_points*n_points))
    theta_22_dst = jnp.reshape(theta_22_dst_tensor, shape= ( n_points*n_points, n_points*n_points))
    theta = jnp.block([[theta_11, theta_12_dst], [theta_12_dst.T, theta_22_dst]])

    return theta


def solve(theta, boundary_condition, coef_forcing, s_decay, reg_bc = 1e-8, reg = 1e-10):
    n_points = coef_forcing.shape[0]
    n_boundary = boundary_condition.shape[0]
    rhs = jnp.hstack([boundary_condition, jnp.reshape(coef_forcing, (n_points)*(n_points))])

    # Adaptive nugget
    decay_nugget =  (jnp.arange(0,n_points)[None]**2 + jnp.arange(0,n_points)[:,None]**2) + 1e-10
    decay_nugget = decay_nugget.reshape((n_points)*(n_points))**(s_decay)
    nugget = jnp.hstack([jnp.ones(n_boundary)*1e-10, decay_nugget*reg])

    alpha = scipy.linalg.solve(theta + jnp.diag(nugget), rhs, assume_a='pos')
    return alpha


def create_interior(x_eval, y_eval):
    # Use meshgrid to create the grid of pairs
    A, B = jnp.meshgrid(x_eval, y_eval, indexing='ij')

    # Combine A and B into pairs
    interior_eval = jnp.stack((A, B), axis=-1).reshape(-1, 2)
    return interior_eval

def predict(alpha, x_eval, y_eval, x,y, boundary_points, length_scale, nu, b_matrix):
    n_eval = x_eval.shape[0]
    n_points =x.shape[0]
    interior_eval = create_interior(x_eval, y_eval)
    K_interior = L_b_y_tensor(x_eval, y_eval, x, y, length_scale, nu, b_matrix).reshape(n_eval*n_eval, n_points, n_points)
    K_interior = vmap_dst_2d(K_interior)
    K_interior = K_interior.reshape(-1, (n_points)*(n_points))

    K_eval_bc = vmap_kernel(interior_eval, boundary_points, length_scale)
    K_eval = jnp.hstack([K_eval_bc, K_interior])


    pred = jnp.dot(K_eval, alpha)
    pred_grid = jnp.reshape(pred, (n_eval, n_eval))

    return pred_grid, pred


def Gauss_Newton(f, boundary_condition, b_matrix_linear,boundary_points, x,y,length_scale, nu, n_iter):
    b_matrix = jnp.copy(b_matrix_linear)
    r_n = f
    n_points = f.shape[0]
    
    for i in range(n_iter):
        print("Current iteration {}".format(i))

        # Build the matrix and sovle the system
        theta, theta_11, theta_12_dst, theta_22_dst = build_matrices(boundary_points,x,y,length_scale, b_matrix, nu)
        alpha = solve(theta, boundary_condition, r_n, 0.5, reg_bc = 1e-10, reg = 1e-8)

        # Evaluate on the grid
        pred_grid, pred = predict(alpha, x, y, x,y, boundary_points, length_scale, nu, b_matrix)

        # Evaluate the DST transform of  negative laplacian 
        K_laplacian_dst = jnp.vstack([theta_12_dst, theta_22_dst]).T
        L_dst = K_laplacian_dst@alpha

        # Compute the discrepancy with f
        error = jnp.linalg.norm(L_dst + dst_2d(tau(pred_grid)).reshape(n_points*n_points) - f.reshape(n_points*n_points))/jnp.linalg.norm(f.reshape(n_points*n_points))
        _, error_truth = compute_2d_error(pred_grid, u_grid, x, y)
        print(error, error_truth)


        if i < n-1:
            # Compute the linearization
            linearization = tau_prime(pred_grid)*pred_grid - tau(pred_grid)
            # Project on the sine basis
            linearization = dst_2d(linearization)
            # Compute the residual
            r_n = f + linearization

            # Compute the new coefficent b
            b_matrix = b_matrix_linear + tau_prime(pred_grid)
    return alpha, b_matrix
    