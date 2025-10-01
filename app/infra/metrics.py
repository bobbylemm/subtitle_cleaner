from prometheus_client import Counter, Histogram, Gauge

# Request metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)

# Processing metrics
PROCESSING_TIME = Histogram(
    "subtitle_processing_duration_seconds",
    "Subtitle processing duration in seconds",
    ["operation", "language"],
)

FILE_SIZE = Histogram(
    "subtitle_file_size_bytes",
    "Size of processed subtitle files in bytes",
    ["format"],
)

SEGMENT_COUNT = Histogram(
    "subtitle_segment_count",
    "Number of segments in subtitle files",
    ["operation"],
)

# Error metrics
ERROR_COUNT = Counter(
    "subtitle_errors_total",
    "Total subtitle processing errors",
    ["error_type", "operation"],
)

# Active processing
ACTIVE_PROCESSING = Gauge(
    "subtitle_active_processing",
    "Number of subtitle files currently being processed",
)


def setup_metrics():
    """Initialize metrics collectors"""
    # Metrics are automatically registered when created
    pass