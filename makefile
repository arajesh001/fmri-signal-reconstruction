IMAGE_NAME = fmri-pipeline

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run -it -v $(shell pwd):/app $(IMAGE_NAME)

test:
	docker run -it -v $(shell pwd):/app $(IMAGE_NAME) pytest tests/

clean:
	docker rmi $(IMAGE_NAME)