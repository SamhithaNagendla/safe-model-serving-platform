# Architecture

The request router maps a stable routing key to a champion or challenger. A challenger failure is
isolated through champion fallback. When shadow mode is enabled, the shadow model executes in a
separate future after the primary prediction has been persisted. SQLite stores request-level
results and delayed labels so metrics survive process restarts and can be compared by version.
