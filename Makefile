clean:
	rm -f -r ./build
	rm -f *.so

.PHONY: build
build: clean
	python3 setup.py build_ext --inplace

