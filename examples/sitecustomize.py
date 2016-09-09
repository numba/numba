import coverage
coverage.process_startup()
import os
print(os.environ['COVERAGE_PROCESS_START'])