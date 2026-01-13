import numpy as np
from opt_einsum import contract_path

print(
    contract_path(
        "abc,def,ij,kl,nm,ikad,jlbe,mncf->",
        np.random.random((3, 3, 3)),
        np.random.random((3, 3, 3)),
        np.random.random((4, 4)),
        np.random.random((4, 4)),
        np.random.random((4, 4)),
        np.random.random((4, 4, 3, 3)),
        np.random.random((4, 4, 3, 3)),
        np.random.random((4, 4, 3, 3)),
    ),
)

print(
    contract_path(
        "abc,def,ij,kl,nm,ikad,jnbe,mlcf->",
        np.random.random((3, 3, 3)),
        np.random.random((3, 3, 3)),
        np.random.random((4, 4)),
        np.random.random((4, 4)),
        np.random.random((4, 4)),
        np.random.random((4, 4, 3, 3)),
        np.random.random((4, 4, 3, 3)),
        np.random.random((4, 4, 3, 3)),
    ),
)
