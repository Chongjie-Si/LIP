import torch

def lip(W_noisy: torch.Tensor,
              X: torch.Tensor,
              Y: torch.Tensor,
              k: int) -> torch.Tensor:
    """
    Apply the LIP refinement to a weight matrix learned from noisy labels,
    using PyTorch tensors.

    Args:
        W_noisy (torch.Tensor): shape (q, l), noisy weight matrix.
        X       (torch.Tensor): shape (n, q), feature matrix.
        Y       (torch.Tensor): shape (n, l), label matrix.
        k       (int): number of principal singular components to retain.

    Returns:
        torch.Tensor: shape (q, l), refined weight matrix W*.
    """
    # 1. Full SVD decomposition on the same device as W_noisy
    U, S, Vh = torch.linalg.svd(W_noisy, full_matrices=False)
    V = Vh.transpose(-2, -1)  # shape (l, l)

    # 2. Principal Subspace Preservation (PSP)
    U_k = U[:, :k]               # (q, k)
    S_k = torch.diag(S[:k])      # (k, k)
    V_k = V[:, :k]               # (l, k)
    W_k = U_k @ S_k @ V_k.t()    # (q, l)

    # 3. Residual subspace
    U_l = U[:, k:]               # (q, r)
    V_l = V[:, k:]               # (l, r)

    # 4. Label Ambiguity Purification (LAP)
    residual = Y - X @ W_k       # (n, l)
    A = U_l.t() @ (X.t() @ residual) @ V_l  # (r, r)
    B = U_l.t() @ (X.t() @ (X @ U_l))       # (r, r)

    # Extract diagonal entries and compute refined singulars
    diag_idx = torch.arange(A.shape[0], device=A.device)
    sigma_refined = A[diag_idx, diag_idx] / B[diag_idx, diag_idx]
    S_l_refined = torch.diag(sigma_refined)  # (r, r)

    # 5. Reconstruct final weight
    W_refined = W_k + U_l @ S_l_refined @ V_l.t()  # (q, l)
    return W_refined
