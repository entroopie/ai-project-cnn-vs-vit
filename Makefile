IMAGE  := rice_miniproject_base
CONT   := rice-run
SRC    := $(PWD)
DATA   := $(PWD)/Rice_Image_Dataset
MODELS := $(PWD)/models
RESULTS := $(PWD)/results

.PHONY: build run stop clean

build:
	docker build -t $(IMAGE) .

run:
	mkdir -p $(MODELS) $(RESULTS)
	docker rm -f $(CONT) 2>/dev/null || true
	docker run --rm --name $(CONT) \
		--gpus all \
		-v $(SRC)/main.py:/app/main.py:ro \
		-v $(SRC)/cnn.py:/app/cnn.py:ro \
		-v $(SRC)/vit.py:/app/vit.py:ro \
		-v $(SRC)/results_writer.py:/app/results_writer.py:ro \
		-v $(DATA):/app/Rice_Image_Dataset:ro \
		-v $(MODELS):/app/models \
		-v $(RESULTS):/app/results \
		$(IMAGE)

stop:
	docker stop $(CONT) 2>/dev/null || true

clean:
	docker rmi $(IMAGE) 2>/dev/null || true
