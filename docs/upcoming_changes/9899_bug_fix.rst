Prevent cache invalidation due to file system differences on various systems
----------------------------------------------------------------------------

Depending on a filesystem on host you might or might not get millisecond precision for file timestamps.
This leads to unwanted consequences and cache invalidation.

Concrete use case: compiling and building caches in CI (within Docker container) and using them later in k8s pod.

This patch adds logic to ignore milliseconds precision while comparing file and cache timestamps.
