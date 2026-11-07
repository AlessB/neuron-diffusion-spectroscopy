Minimal stand-ins for graphtools' non-numerical dependencies (pygsp, tasklogger, deprecated, future),
which are not installed here and cannot be pip-installed. They contain no numerical code: logging is a
no-op, pygsp.graphs.Graph is an empty base class. With these on sys.path, the UNMODIFIED graphtools and
m_phate.kernel source from /home/claude/refcode can be imported and run (numba absent -> pure numpy path).
