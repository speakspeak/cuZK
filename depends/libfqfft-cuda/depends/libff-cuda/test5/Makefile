alt:
	nvcc -arch=sm_35 -rdc=true --expt-extended-lambda ./test5.cu ../depends/libstl-cuda/memory.cu ../common/utils.cu ../curves/alt_bn128/alt_bn128_pp.cu ../curves/alt_bn128/alt_bn128_pairing.cu ../curves/alt_bn128/alt_bn128_g2.cu ../curves/alt_bn128/alt_bn128_g1.cu ../curves/alt_bn128/alt_bn128_init.cu ../mini-mp-cuda/mini-mp-cuda.cu -o test

bls:
	nvcc -arch=sm_35 -rdc=true --expt-extended-lambda ./test5.cu ../depends/libstl-cuda/memory.cu ../common/utils.cu ../curves/bls12_381/bls12_381_pp.cu ../curves/bls12_381/bls12_381_pairing.cu ../curves/bls12_381/bls12_381_g2.cu ../curves/bls12_381/bls12_381_g1.cu ../curves/bls12_381/bls12_381_init.cu ../mini-mp-cuda/mini-mp-cuda.cu -o test

mnt:
	nvcc -arch=sm_35 -rdc=true --expt-extended-lambda ./test5.cu ../depends/libstl-cuda/memory.cu ../common/utils.cu ../curves/mnt4/mnt4_pp.cu ../curves/mnt4/mnt4_pairing.cu ../curves/mnt4/mnt4_g2.cu ../curves/mnt4/mnt4_g1.cu ../curves/mnt4/mnt4_init.cu ../mini-mp-cuda/mini-mp-cuda.cu -o test


