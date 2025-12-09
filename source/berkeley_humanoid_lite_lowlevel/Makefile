.PHONY: build rebuild run clean

build:
	cmake -S ./ -B ./build/
	cmake --build ./build/ --target main

rebuild:
	cmake --build ./build/ --target clean
	cmake --build ./build/ --target main

run: build
	./build/main

clean:
	rm -rf ./build/