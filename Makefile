execute:
	rm -Rf build/ src/cudanoncuda.cpp src/cudanoncuda.so
	python setup.py build_ext --inplace
