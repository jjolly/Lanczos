lanczos-cuda: main.cu lanczos.cu readsparsematrix.cu printvec.cu deletesparsematrix.cu createfullmatrix.cu randomzahl.cu printfilematrix.cu deletefullmatrix.cu createsparsematrix.cu
	nvcc -g -O0 -o $@ $^
lanczos: main.c lanczos.h lanczos.c axpy.c copyvector.c createfullmatrix.c createsparsematrix.c createvector.c deletefullmatrix.c deletesparsematrix.c deletevector.c dotproduct.c fillfullmatrix.c fuelledia.c fuellevector.c norm.c printfilematrix.c printfullmatrix.c printsparsematrix.c printvec.c randomzahl.c readsparsematrix.c readvec.c scalevector.c sparsematrixvector.c
	gcc -o $@ $^ -lm -I.
clean:
	for i in lanczos lanczos-cuda; do \
		if [ -e $$i ]; then rm -v $$i; fi; true; \
	done
