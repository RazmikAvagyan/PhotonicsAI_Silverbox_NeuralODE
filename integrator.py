def rk4_step(model, y, v, u, h):
    """
    Performs a single classical 4th-Order Runge-Kutta (RK4) step for continuous dynamic systems.

    System Equations:
        dy/dt = v           (velocity: rate of change of position)
        dv/dt = model       (acceleration dv/dt: rate of change of velocity, predicted by MLP)
    """
    # Evaluate state derivatives at current point t
    k1_v = v
    k1_dv = model(y, v, u)  # v' (acceleration)

    # First midpoint estimate at t + h/2
    y2 = y + 0.5 * h * k1_v
    v2 = v + 0.5 * h * k1_dv
    k2_v = v2
    k2_dv = model(y2, v2, u)  # dv/dt (acceleration)

    # Second midpoint estimate at t + h/2
    y3 = y + 0.5 * h * k2_v
    v3 = v + 0.5 * h * k2_dv
    k3_v = v3
    k3_dv = model(y3, v3, u)  # dv/dt (acceleration)

    # Full-step endpoint estimate at t + h
    y4 = y + h * k3_v
    v4 = v + h * k3_dv
    k4_v = v4
    k4_dv = model(y4, v4, u)  # dv/dt (acceleration)

    # Simpson's weighted combination to compute next state at t + h
    y_next = y + (h / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    v_next = v + (h / 6.0) * (k1_dv + 2.0 * k2_dv + 2.0 * k3_dv + k4_dv)

    return y_next, v_next