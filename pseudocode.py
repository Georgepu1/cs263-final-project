# torch architecture


for _ in range(epochs):
    for (x1, y1), (x2, y2), ... in zip(ds_loader1, ds_loader2, ...):
        # Note can also set the data to a decide (cuda)
        m.zero_grad()
        m_outputs = [m(x) for x in [x1, x2, ...]]
        loss = c(m_outputs[0], y1) + \
            c(m_outputs[0], y2) + ...  # multitask loss
        loss.backward()
        # intermediate variabels stores embedding of x and computes
        # m_output.grad w.r.t. this and calculate the MSE of the m_output.grad(emb_x)
        # norm and 1.0 and use relu on; before you do the square, pass it through a relu
        # so everything les than 1.0 wont be counted to the square.
        # MSE (m_output.grad(x), 1.0) calulate gradient of M output w.r.t. x's embedding space
        # To avoid overfitting, calculating regularization term can use a varied version of x
        # instead of the original (e.g. add gaussian noise around embeddings of x); can
        # also minimize discrepancy on the two for robustness of model
        # Lipschitz-regularized loss
        o.step()
        # print statements/early stopping
