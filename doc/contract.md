# Contraction

Two types of baryon two-point correlation function can be illustrated as

![](./contract1.drawio.svg)
![](./contract2.drawio.svg)

All contraction can be defined as these two types by swapping $i,j$ and/or $k,l$.

## `numpy.einsum` and similar implementation

We use `opt_einsum` to optimize the contraction order.

For type 1
```log
([(4, 7), (2, 4), (2, 3), (3, 4), (0, 3), (0, 2), (0, 1)],   Complete contraction:  abc,def,ij,kl,nm,ikad,jlbe,mncf->
         Naive scaling:  12
     Optimized scaling:  6
      Naive FLOP count:  2.389e+7
  Optimized FLOP count:  5.850e+3
   Theoretical speedup:  4.083e+3
  Largest intermediate:  1.440e+2 elements
--------------------------------------------------------------------------------
scaling        BLAS                current                             remaining
--------------------------------------------------------------------------------
   4    GEMV/EINSUM            mncf,nm->cf          abc,def,ij,kl,ikad,jlbe,cf->
   5           GEMM          ikad,ij->kadj             abc,def,kl,jlbe,cf,kadj->
   5           TDOT          jlbe,kl->jbek                abc,def,cf,kadj,jbek->
   6           TDOT        jbek,kadj->bead                     abc,def,cf,bead->
   5           TDOT          bead,abc->edc                          def,cf,edc->
   4           TDOT            edc,def->cf                               cf,cf->
   2            DOT                cf,cf->                                    ->)
```
Actually we need 2925 multiplication and 2925 addition.

For type 2
```log
([(2, 5), (2, 5), (2, 3), (0, 2), (1, 3), (1, 2), (0, 1)],   Complete contraction:  abc,def,ij,kl,nm,ikad,jnbe,mlcf->
         Naive scaling:  12
     Optimized scaling:  7
      Naive FLOP count:  2.389e+7
  Optimized FLOP count:  1.906e+4
   Theoretical speedup:  1.253e+3
  Largest intermediate:  4.320e+2 elements
--------------------------------------------------------------------------------
scaling        BLAS                current                             remaining
--------------------------------------------------------------------------------
   5           GEMM          ikad,ij->kadj        abc,def,kl,nm,jnbe,mlcf,kadj->
   5           TDOT          mlcf,kl->mcfk           abc,def,nm,jnbe,kadj,mcfk->
   5           TDOT          jnbe,nm->jbem              abc,def,kadj,mcfk,jbem->
   6           TDOT        kadj,abc->kdjbc                 def,mcfk,jbem,kdjbc->
   7           TDOT      kdjbc,mcfk->djbmf                      def,jbem,djbmf->
   6           TDOT        djbmf,jbem->dfe                             def,dfe->
   3     DOT/EINSUM              dfe,def->                                    ->)
```
Actually we need 9531 multiplication and 9531 addition.

## `pycontract`

We need `ikad,jlbe,mncf` as data, and we make `epsion`, `Gamma_A`, `Gamma_B` and `P` as functions since they are sparse matrices.

### FLOP

It's easy to extract two `epsilon`s into two $N_c$ loops.

It's easy to extract one `Gamma` or `P` into one $N_s$ loop.

For type 1
$$
4\times4\times3\times3\times\left(4\times1+7\right)=1584 \text{ multiplications} \\
4\times4\times3\times3\times\left(4\times1+4\right)=1152 \text{ additions} \\
$$

For type 2
$$
4\times4\times3\times3\times\left(4\times5+3\right)=3312 \text{ multiplications} \\
4\times4\times3\times3\times\left(4\times4+1\right)=2448 \text{ additions} \\
$$

Consider that multiplication actually uses more float-pointing operations than addition for complex numbers, here we get approximately 2x and 3x for type 1 and type 2 contraction, respectively.

### Shared memory

Naively, we need 3 `complex<double> propag[4][4][3][3]` local variables per lattice site. But 3x2304 bytes (1728 32-bit registers) are too much for a single GPU thread (Maximum 255 registers per thread for A100).

![](./contract3.drawio.svg)

We use a block of 64 threads to perform calculation on 4 lattice site. We handle 2304*4 bytes on shared memory, and every 16 threads perform the calculation on one lattice site. (108 32-bit registers per threads)

![](./contract4.drawio.svg)

Here we actually get more than 3x bandwidth while using the shared memory to save the register usage.


## Benchmark

For type 1
```log
PyQUDA INFO: Time for cupy.einsum: 0.069742 sec
PyQUDA INFO: Time for pycontract.baryonTwoPoint: 0.009307 sec
```
Approximately 7.5x


For type 2
```log
PyQUDA INFO: Time for cupy.einsum: 0.150068 sec
PyQUDA INFO: Time for pycontract.baryonTwoPoint: 0.011815 sec
```
Approximately 12.7x