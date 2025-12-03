# Makefile
.PHONY: build run stop test clean

build:
	docker build -t california-housing-ml .

run:
	docker run -d -p 8000:8000 --name housing-ml-service california-housing-ml
	@echo "Service running at http://localhost:8000"

stop:
	docker stop housing-ml-service || true
	docker rm housing-ml-service || true

logs:
	docker logs -f housing-ml-service

test:
	curl http://localhost:8000/health

clean: stop
	docker rmi california-housing-ml || true

restart: stop run