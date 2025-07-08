# Contraction for LQCD correlation functions

Requires [PyQUDA-Utils](https://pypi.org/project/PyQUDA-Utils/) >= 0.10.22

## Install
```bash
git clone https://github.com/CLQCD/contract.git
cd contract
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMAND=ON -DGPU_ARCH=60
cmake --build . -j8 && cmake --install .
python -m pyquda_plugins -i contract.h -l contract -I $(pwd)/install/include -L $(pwd)/install/lib
cd ../..
```

## Check
```bash
cd contract
mkdir -p .cache
mpirun -n 4 python baryon_2pt.py
```