# Targets run inside docker (no host python needed). `make test-local` uses host python if pytest exists.
REPO := $(CURDIR)

.PHONY: test test-local build-images doctor

test: build-images
	docker run --rm -v $(REPO):/repo -w /repo -e PYTHONPATH=/repo -e PYTHONDONTWRITEBYTECODE=1 \
	  --entrypoint python r2s-cli:latest -m pytest -q -p no:cacheprovider

test-local:
	python3 -m pytest -q

# Builds r2s-cli (the only image buildable today). The isaac service reuses r2s-isaaclab3-sim61 (see README).
build-images:
	docker build -t r2s-cli:latest -f docker/r2s-cli/Dockerfile .

doctor:
	./r2s doctor
