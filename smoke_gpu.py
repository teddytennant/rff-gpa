"""H200 smoke for the MC ELBO train step.

A few SGD steps on one random sequence. Not the paper's Transformer, and not
a calibration result. JAX_PLATFORMS is left to the job script.
"""

import jax

from rff_gpa.elbo import init_train_params, train_step


def main():
    devices = jax.devices()
    print("jax", jax.__version__, devices, flush=True)
    if not any(d.platform == "gpu" for d in devices):
        raise SystemExit("expected a GPU device")

    key = jax.random.PRNGKey(0)
    length, dim, value_dim, n_classes = 8, 16, 16, 4
    n_freq = 32
    steps = 20
    k_q, k_k, k_v, k_label, k_init = jax.random.split(key, 5)
    q = jax.random.normal(k_q, (length, dim))
    k = jax.random.normal(k_k, (length, dim))
    v = jax.random.normal(k_v, (length, value_dim))
    label = jax.random.randint(k_label, (), 0, n_classes)
    params = init_train_params(k_init, dim, value_dim, n_classes)

    losses = []
    for step in range(steps):
        step_key = jax.random.fold_in(key, step)
        params, loss = train_step(
            q, k, v, label, params, step_key, n_freq, 1e-2, n_samples=2
        )
        loss_f = float(loss)
        losses.append(loss_f)
        print(f"step {step} loss {loss_f:.6f}", flush=True)

    print(f"initial {losses[0]:.6f} final {losses[-1]:.6f}", flush=True)
    if not all(x == x and abs(x) < 1e12 for x in losses):
        raise SystemExit("non-finite loss")
    print("rff-gpa gpu smoke done", flush=True)


if __name__ == "__main__":
    main()
