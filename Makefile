.PHONY: help install test test-integration run-server run-example clean lint format

# Default target
help:
	@echo "Available commands:"
	@echo "  install           - Install package and dependencies"
	@echo "  install-dev       - Install package in development mode"
	@echo "  test             - Run basic tests"
	@echo "  test-integration - Run integration tests (requires http-trickle)"
	@echo "  test-all         - Run all tests"
	@echo "  run-server       - Run the example HTTP server"
	@echo "  run-example      - Run the simple streaming example"
	@echo "  run-workflow     - Run the automated test workflow"
	@echo "  lint             - Run linting checks"
	@echo "  format           - Format code with black"
	@echo "  clean            - Clean up build artifacts"
	@echo "  setup-http-trickle - Setup http-trickle for testing"

# Installation
install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e .[dev]

# Testing
test:
	pytest tests/test_basic_streaming.py -v

test-integration:
	pytest tests/test_integration.py -v -m integration

test-all:
	pytest tests/ -v

# Running examples
run-server:
	python examples/http_server_example.py

run-example:
	python examples/async_processor_example.py

run-workflow:
	python scripts/test_workflow.py

# Development tools
lint:
	flake8 pytrickle/ examples/ tests/ --max-line-length=120
mypy pytrickle/ --ignore-missing-imports

format:
	black pytrickle/ examples/ tests/ scripts/ --line-length=120

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/

# Setup http-trickle for testing
setup-http-trickle:
	@echo "Setting up http-trickle for testing..."
	@if [ ! -d "$(HOME)/repos/http-trickle" ]; then \
		echo "Cloning http-trickle..."; \
		mkdir -p $(HOME)/repos; \
		git clone https://github.com/livepeer/http-trickle.git $(HOME)/repos/http-trickle; \
	fi
	@echo "Building http-trickle..."
	@cd $(HOME)/repos/http-trickle && (make all || make trickle-server || echo "http-trickle build completed")
	@echo "http-trickle setup complete!"

# Start trickle server (for manual testing)
start-trickle-server:
	@echo "Starting trickle server on localhost:3389..."
	@cd $(HOME)/repos/http-trickle && make trickle-server addr=0.0.0.0:3389

# Publish test video (requires video file)
publish-test-video:
	@echo "Publishing test video..."
	@cd $(HOME)/repos/http-trickle && make publisher-ffmpeg in=bbb_sunflower_1080p_30fps_normal.mp4 stream=sample url=http://127.0.0.1:3389

# Play processed output
play-output:
	@echo "Playing processed output stream..."
	@cd $(HOME)/repos/http-trickle && go run cmd/read2pipe/*.go --url http://127.0.0.1:3389/ --stream sample-output | ffplay -probesize 64 -

# Full demo workflow
demo: setup-http-trickle
	@echo "Starting full demo workflow..."
	@echo "This will start all servers and run a complete test"
	python scripts/test_workflow.py

# Development setup
dev-setup: install-dev setup-http-trickle
	@echo "Development environment setup complete!"
	@echo "You can now run 'make demo' to test the full workflow" 